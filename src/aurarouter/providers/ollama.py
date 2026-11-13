import json
from collections.abc import AsyncIterator

import httpx

from aurarouter._logging import get_logger
from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.providers.base import BaseProvider
from aurarouter.ratelimit.parsers import parse_rate_limit_headers
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.Ollama")

_OLLAMA_CONTEXT_KEYWORDS = (
    "context length",
    "context window",
    "too many tokens",
    "prompt is too long",
    "exceeds the available context",
    "num_ctx",
)


def _safe_response_text(resp: httpx.Response | None) -> str:
    if resp is None:
        return ""
    try:
        txt = resp.text
        return txt if isinstance(txt, str) else ""
    except Exception:
        return ""


def _classify_ollama_http_error(exc: httpx.HTTPStatusError) -> Exception:
    """Translate Ollama / proxy HTTP errors into typed AuraRouter routing exceptions (QA 3.4)."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", 0)
    raw_headers = getattr(resp, "headers", None)
    hdr_dict = dict(raw_headers) if isinstance(raw_headers, (dict, httpx.Headers)) else {}
    body_text = _safe_response_text(resp)

    if status == 429:
        snap = parse_rate_limit_headers(hdr_dict, error_body=body_text)
        return RateLimitExceededError(str(exc), snapshot=snap, provider="ollama")

    if status in (400, 413, 422):
        lower = f"{body_text} {exc}".lower()
        if any(kw in lower for kw in _OLLAMA_CONTEXT_KEYWORDS):
            return ContextLengthExceededError(str(exc), provider="ollama")

    if isinstance(status, int) and status >= 500:
        return ProviderTransientError(str(exc), status_code=status, provider="ollama")

    return exc


def _format_ollama_messages(messages: list[dict], system_prompt: str = "") -> list[dict]:
    """Convert internal message dicts (supporting per-message ``images``) to Ollama /api/chat format."""
    from aurarouter.media import extract_base64_images

    chat_messages: list[dict] = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        entry = dict(m)
        if m.get("images"):
            b64_imgs = extract_base64_images(m["images"])
            if b64_imgs:
                entry["images"] = b64_imgs
            else:
                entry.pop("images", None)
        chat_messages.append(entry)
    return chat_messages


class OllamaProvider(BaseProvider):
    """Local Ollama HTTP API provider."""

    def generate(self, prompt: str, json_mode: bool = False,
                 response_schema: dict | None = None,
                 images: list[dict] | None = None) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode,
                                        response_schema=response_schema,
                                        images=images).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ) -> GenerateResult:
        endpoints = self._get_endpoints()

        payload: dict = {
            "model": self.config["model_name"],
            "prompt": prompt,
            "stream": False,
            "options": self.config.get("parameters", {}),
        }
        if images:
            from aurarouter.media import extract_base64_images
            b64_imgs = extract_base64_images(images)
            if b64_imgs:
                payload["images"] = b64_imgs
        if response_schema is not None:
            payload["format"] = response_schema
        elif json_mode:
            payload["format"] = "json"

        timeout = self.config.get("timeout", 120.0)
        last_error: Exception | None = None

        with self._http_client(timeout, client_factory=httpx.Client) as client:
            for url in endpoints:
                try:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return GenerateResult(
                        text=data.get("response", ""),
                        input_tokens=data.get("prompt_eval_count", 0) or 0,
                        output_tokens=data.get("eval_count", 0) or 0,
                        finish_reason=data.get("done_reason"),
                    )
                except httpx.HTTPStatusError as e:
                    typed = _classify_ollama_http_error(e)
                    if isinstance(typed, ContextLengthExceededError):
                        raise typed from e
                    last_error = typed
                    logger.warning(f"Ollama endpoint {url} returned HTTP error: {typed}")
                    continue
                except httpx.RequestError as e:
                    last_error = e
                    logger.warning(f"Ollama endpoint {url} failed: {e}")
                    continue

        if last_error:
            raise last_error
        return GenerateResult(text="")

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via Ollama /api/chat endpoint."""
        chat_messages = _format_ollama_messages(messages, system_prompt)

        payload = {
            "model": self.config["model_name"],
            "messages": chat_messages,
            "stream": False,
        }

        params = self.config.get("parameters", {})
        if params:
            payload["options"] = params
        if json_mode:
            payload["format"] = "json"

        chat_endpoints = self._resolve_chat_endpoints()
        timeout = self.config.get("timeout", 120)
        last_error: Exception | None = None
        data: dict | None = None

        with self._http_client(timeout, client_factory=httpx.Client) as client:
            for endpoint in chat_endpoints:
                try:
                    resp = client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    last_error = None
                    break
                except httpx.HTTPStatusError as e:
                    typed = _classify_ollama_http_error(e)
                    if isinstance(typed, ContextLengthExceededError):
                        raise typed from e
                    last_error = typed
                    logger.warning(f"Ollama chat endpoint {endpoint} returned HTTP error: {typed}")
                    continue
                except httpx.RequestError as e:
                    last_error = e
                    logger.warning(f"Ollama chat endpoint {endpoint} failed: {e}")
                    continue

        if last_error is not None:
            raise last_error
        if data is None:
            raise RuntimeError("Empty response from Ollama /api/chat")

        text = data.get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("Empty response from Ollama /api/chat")

        return GenerateResult(
            text=text,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            model_id=self.config.get("model_name", ""),
            provider="ollama",
            context_limit=self.get_context_limit(),
            finish_reason=data.get("done_reason"),
        )

    def generate_stream_sync(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ):
        """[Deprecated] Use async generate_stream. This is kept for sync fabric compatibility if needed."""
        endpoints = self._get_endpoints()
        payload: dict = {
            "model": self.config["model_name"],
            "prompt": prompt,
            "stream": True,
            "options": self.config.get("parameters", {}),
        }
        if images:
            from aurarouter.media import extract_base64_images
            b64_imgs = extract_base64_images(images)
            if b64_imgs:
                payload["images"] = b64_imgs
        if response_schema is not None:
            payload["format"] = response_schema
        elif json_mode:
            payload["format"] = "json"

        timeout = self.config.get("timeout", 120.0)
        last_error: Exception | None = None

        for url in endpoints:
            tokens_emitted = False
            try:
                with self._http_client(timeout, client_factory=httpx.Client) as client:
                    with client.stream("POST", url, json=payload) as resp:
                        status = getattr(resp, "status_code", 200)
                        if isinstance(status, int) and status >= 400:
                            resp.read()
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line.strip():
                                continue
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                tokens_emitted = True
                                yield token
                            if chunk.get("done"):
                                in_tok = chunk.get("prompt_eval_count", 0) or 0
                                out_tok = chunk.get("eval_count", 0) or 0
                                self._record_stream_result(
                                    GenerateResult(text="", input_tokens=in_tok, output_tokens=out_tok),
                                    simulated_stream=False,
                                )
                                return
                return
            except httpx.HTTPStatusError as e:
                typed = _classify_ollama_http_error(e)
                if tokens_emitted or isinstance(typed, ContextLengthExceededError):
                    raise typed from e
                last_error = typed
                logger.warning(f"Ollama streaming endpoint {url} returned HTTP error: {typed}")
                continue
            except httpx.RequestError as e:
                if tokens_emitted:
                    raise
                last_error = e
                logger.warning(f"Ollama streaming endpoint {url} failed: {e}")
                continue

        if last_error:
            raise last_error

    async def generate_stream(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama /api/generate via NDJSON."""
        endpoints = self._get_endpoints()
        payload: dict = {
            "model": self.config["model_name"],
            "prompt": prompt,
            "stream": True,
            "options": self.config.get("parameters", {}),
        }
        if images:
            from aurarouter.media import extract_base64_images
            b64_imgs = extract_base64_images(images)
            if b64_imgs:
                payload["images"] = b64_imgs
        if response_schema is not None:
            payload["format"] = response_schema
        elif json_mode:
            payload["format"] = "json"

        timeout = self.config.get("timeout", 120.0)
        last_error: Exception | None = None

        for url in endpoints:
            tokens_emitted = False
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        status = getattr(resp, "status_code", 200)
                        if isinstance(status, int) and status >= 400:
                            await resp.aread()
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                tokens_emitted = True
                                yield token
                            if chunk.get("done"):
                                in_tok = chunk.get("prompt_eval_count", 0) or 0
                                out_tok = chunk.get("eval_count", 0) or 0
                                self._record_stream_result(
                                    GenerateResult(text="", input_tokens=in_tok, output_tokens=out_tok),
                                    simulated_stream=False,
                                )
                                return
                return
            except httpx.HTTPStatusError as e:
                typed = _classify_ollama_http_error(e)
                if tokens_emitted or isinstance(typed, ContextLengthExceededError):
                    raise typed from e
                last_error = typed
                logger.warning(f"Ollama streaming endpoint {url} returned HTTP error: {typed}")
                continue
            except httpx.RequestError as e:
                if tokens_emitted:
                    raise
                last_error = e
                logger.warning(f"Ollama streaming endpoint {url} failed: {e}")
                continue

        if last_error:
            raise last_error

    async def generate_stream_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama /api/chat via NDJSON."""
        chat_messages = _format_ollama_messages(messages, system_prompt)

        payload: dict = {
            "model": self.config["model_name"],
            "messages": chat_messages,
            "stream": True,
        }
        params = self.config.get("parameters", {})
        if params:
            payload["options"] = params
        if json_mode:
            payload["format"] = "json"

        chat_endpoints = self._resolve_chat_endpoints()
        timeout = self.config.get("timeout", 120.0)
        last_error: Exception | None = None

        for endpoint in chat_endpoints:
            tokens_emitted = False
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", endpoint, json=payload) as resp:
                        status = getattr(resp, "status_code", 200)
                        if isinstance(status, int) and status >= 400:
                            await resp.aread()
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                tokens_emitted = True
                                yield token
                            if chunk.get("done"):
                                in_tok = chunk.get("prompt_eval_count", 0) or 0
                                out_tok = chunk.get("eval_count", 0) or 0
                                self._record_stream_result(
                                    GenerateResult(text="", input_tokens=in_tok, output_tokens=out_tok),
                                    simulated_stream=False,
                                )
                                return
                return
            except httpx.HTTPStatusError as e:
                typed = _classify_ollama_http_error(e)
                if tokens_emitted or isinstance(typed, ContextLengthExceededError):
                    raise typed from e
                last_error = typed
                logger.warning(f"Ollama chat streaming endpoint {endpoint} returned HTTP error: {typed}")
                continue
            except httpx.RequestError as e:
                if tokens_emitted:
                    raise
                last_error = e
                logger.warning(f"Ollama chat streaming endpoint {endpoint} failed: {e}")
                continue

        if last_error:
            raise last_error

    def _resolve_chat_endpoints(self) -> list[str]:
        """Resolve all /api/chat endpoints from config."""
        endpoints = self._get_endpoints()
        if not endpoints:
            return ["http://localhost:11434/api/chat"]
        resolved: list[str] = []
        for ep in endpoints:
            base = ep.rsplit("/api/", 1)[0]
            resolved.append(f"{base}/api/chat")
        return resolved

    def _resolve_chat_endpoint(self) -> str:
        """Resolve the primary /api/chat endpoint from config."""
        return self._resolve_chat_endpoints()[0]

    def get_context_limit(self) -> int:
        """Return context limit from config, or query Ollama /api/show."""
        limit = self.config.get("context_limit", 0)
        if limit > 0:
            return limit
        try:
            endpoints = self._get_endpoints()
            base = endpoints[0].rsplit("/api/", 1)[0] if endpoints else "http://localhost:11434"
            resp = httpx.post(
                f"{base}/api/show",
                json={"name": self.config["model_name"]},
                timeout=10,
            )
            if resp.status_code == 200:
                info = resp.json()
                params = info.get("model_info", {})
                for key, value in params.items():
                    if "context_length" in key:
                        return int(value)
        except Exception as ex:  # noqa: F841
            logger.debug("providers.ollama.get_context_limit_error", exc_info=True)
        return 0

    def _get_endpoints(self) -> list[str]:
        endpoints = self.config.get("endpoints", [])
        if not endpoints:
            # Fallback to single endpoint for backward compatibility
            endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
            return [endpoint]
        return endpoints

    def get_telemetry(self):
        """Query Ollama /api/ps for loaded model state."""
        try:
            from aurarouter.auragrid.contracts import ModelState, ModelTelemetry
        except ImportError:
            return None
        try:
            endpoints = self._get_endpoints()
            base = endpoints[0].rsplit("/api/", 1)[0] if endpoints else "http://localhost:11434"
            resp = httpx.get(f"{base}/api/ps", timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            model_name = self.config.get("model_name", "unknown")
            for m in data.get("models", []):
                if m.get("name", "").startswith(model_name):
                    vram_mb = m.get("size_vram", 0) / (1024 * 1024) if m.get("size_vram") else None
                    return ModelTelemetry(
                        model_id=model_name,
                        provider_name=self.__class__.__name__,
                        state=ModelState.WARM,
                        vram_usage_mb=vram_mb,
                    )
            return ModelTelemetry(
                model_id=model_name,
                provider_name=self.__class__.__name__,
                state=ModelState.COLD,
            )
        except Exception as ex:  # noqa: F841
            logger.debug("providers.ollama.get_telemetry_error", exc_info=True)
            return ModelTelemetry(
                model_id=self.config.get("model_name", "unknown"),
                provider_name=self.__class__.__name__,
                state=ModelState.UNKNOWN,
            )
