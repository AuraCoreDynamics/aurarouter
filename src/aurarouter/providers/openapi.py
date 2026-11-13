"""OpenAPI-compatible chat/completions provider.

Works with vLLM, SGLang, text-generation-inference, LocalAI, LM Studio,
Ollama's OpenAI-compatible endpoint, and any server implementing
the ``/v1/chat/completions`` endpoint.

For OpenAI Cloud (GPT-4o, GPT-4.1, GPT-5, o1/o3/o4-mini, `/v1/responses`,
and Azure OpenAI), prefer the dedicated ``aurarouter-openai`` MCP provider
package.  If an ``openapi`` config points at a known cloud host without an
explicit ``hosting_tier``, AuraRouter automatically classifies it as
``cloud`` for sovereignty, privacy, and budget enforcement.

Config::

    my_vllm:
      provider: openapi
      endpoint: http://localhost:8000/v1
      model_name: meta-llama/Llama-3-8B
      api_key: optional-key      # or env_key: VLLM_API_KEY
      extra_headers:
        X-Custom-Header: value
      parameters:
        temperature: 0.7
        max_tokens: 2048
        top_p: 0.95
        extra_body:
          repetition_penalty: 1.05
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from aurarouter.media import (
    format_anthropic_content,
    format_gemini_parts,
    format_openai_messages,
)

import httpx

from aurarouter._logging import get_logger
from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.providers.base import BaseProvider
from aurarouter.ratelimit.models import RateLimitSnapshot
from aurarouter.ratelimit.parsers import parse_rate_limit_headers
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.OpenAPI")

_CONTEXT_OVERFLOW_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "prompt is too long",
    "exceeds the available context size",
    "context window",
)

_PASSTHROUGH_SAMPLING_KEYS = (
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "user",
    "n",
)


def _has_snapshot_data(snap: RateLimitSnapshot | None) -> bool:
    if snap is None:
        return False
    return snap.is_daily_quota or any(
        getattr(snap, f) is not None
        for f in snap.__dataclass_fields__
        if f != "is_daily_quota"
    )


def _extract_response_snapshot(resp: httpx.Response) -> RateLimitSnapshot | None:
    raw_headers = getattr(resp, "headers", None)
    if not isinstance(raw_headers, (dict, httpx.Headers)):
        return None
    snap = parse_rate_limit_headers(dict(raw_headers))
    return snap if _has_snapshot_data(snap) else None


def _safe_response_text(resp: httpx.Response) -> str:
    try:
        text = resp.text
        return text if isinstance(text, str) else ""
    except Exception:
        return ""


def _classify_http_status_error(exc: httpx.HTTPStatusError) -> Exception:
    resp = exc.response
    status = getattr(resp, "status_code", 0)
    raw_headers = getattr(resp, "headers", None)
    hdr_dict = dict(raw_headers) if isinstance(raw_headers, (dict, httpx.Headers)) else {}
    body_text = _safe_response_text(resp)

    if status == 429:
        snap = parse_rate_limit_headers(hdr_dict, error_body=body_text)
        return RateLimitExceededError(str(exc), snapshot=snap, provider="openapi")

    if status in (400, 413, 422):
        lower = body_text.lower()
        if any(kw in lower for kw in _CONTEXT_OVERFLOW_KEYWORDS):
            return ContextLengthExceededError(str(exc), provider="openapi")

    if isinstance(status, int) and status >= 500:
        return ProviderTransientError(str(exc), status_code=status, provider="openapi")

    return exc


def _extract_logprobs(choice: dict[str, Any]) -> tuple[list[int] | None, list[float] | None]:
    lp_data = choice.get("logprobs") or {}
    content_lps = lp_data.get("content")
    if not content_lps:
        return None, None
    tokens_out: list[int] = []
    logprobs_out: list[float] = []
    for item in content_lps:
        tokens_out.append(item.get("token_id", 0))
        logprobs_out.append(item.get("logprob", 0.0))
    return tokens_out, logprobs_out


class OpenAPIProvider(BaseProvider):
    """Provider for OpenAI-API-compatible ``/v1/chat/completions`` endpoints."""

    def _resolve_dialect(self) -> str:
        explicit = str(self.config.get("api_dialect") or "").strip().lower()
        if explicit in ("openai", "anthropic", "gemini"):
            return explicit
        ep = str(self.config.get("endpoint") or "").lower()
        if "api.anthropic.com" in ep or ep.rstrip("/").endswith("/messages"):
            return "anthropic"
        if "generativelanguage.googleapis.com" in ep or ":generatecontent" in ep:
            return "gemini"
        return "openai"

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        extra_hdrs = self.config.get("extra_headers")
        if isinstance(extra_hdrs, dict):
            for k, v in extra_hdrs.items():
                headers[str(k)] = str(v)
        api_key = self.resolve_api_key() or ""
        dialect = self._resolve_dialect()
        if dialect == "anthropic":
            if api_key and "x-api-key" not in headers:
                headers["x-api-key"] = api_key
            headers.setdefault("anthropic-version", str(self.config.get("anthropic_version", "2023-06-01")))
        elif dialect == "gemini":
            if api_key and "x-goog-api-key" not in headers and "Authorization" not in headers:
                headers["x-goog-api-key"] = api_key
        elif api_key and "Authorization" not in headers and "api-key" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
    ) -> dict[str, Any]:
        params = self.config.get("parameters") or {}
        model_name = self.config.get("model_name", "")

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }

        if "max_completion_tokens" in params:
            payload["max_completion_tokens"] = params["max_completion_tokens"]
        else:
            payload["max_tokens"] = params.get("max_tokens", 2048)

        if "temperature" in params or "max_completion_tokens" not in params:
            payload["temperature"] = params.get("temperature", 0.7)

        for key in _PASSTHROUGH_SAMPLING_KEYS:
            if key in params and params[key] is not None:
                payload[key] = params[key]

        extra_body = params.get("extra_body") or self.config.get("extra_body")
        if isinstance(extra_body, dict):
            for k, v in extra_body.items():
                if k not in payload:
                    payload[k] = v

        if stream and bool(self.config.get("include_stream_usage", False)):
            payload["stream_options"] = {"include_usage": True}

        if return_tokens and not stream:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "modifications",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _build_dialect_request(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
    ) -> tuple[str, str, dict[str, Any]]:
        """Build ``(dialect, url, payload)`` for openai, anthropic, or gemini endpoints."""
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        model_name = self.config.get("model_name", "")
        dialect = self._resolve_dialect()
        params = self.config.get("parameters") or {}

        if dialect == "anthropic":
            url = (
                endpoint
                if endpoint.rstrip("/").endswith("/messages")
                else endpoint.rstrip("/") + "/messages"
            )
            system_parts: list[str] = []
            anth_messages: list[dict[str, Any]] = []
            for msg in messages:
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")
                imgs = msg.get("images")
                if role == "system":
                    if content:
                        system_parts.append(str(content))
                    continue
                if isinstance(content, list) and not imgs:
                    anth_content: Any = content
                else:
                    anth_content = format_anthropic_content(str(content), imgs)
                anth_messages.append({"role": role, "content": anth_content})
            payload: dict[str, Any] = {
                "model": model_name,
                "max_tokens": params.get("max_tokens", 2048),
                "messages": anth_messages,
            }
            if system_parts:
                payload["system"] = "\n\n".join(system_parts)
            if "temperature" in params:
                payload["temperature"] = params["temperature"]
            if stream:
                payload["stream"] = True
            return dialect, url, payload

        if dialect == "gemini":
            action = "streamGenerateContent" if stream else "generateContent"
            if ":generateContent" in endpoint or ":streamGenerateContent" in endpoint:
                url = (
                    endpoint.replace(":generateContent", ":streamGenerateContent")
                    if stream
                    else endpoint.replace(":streamGenerateContent", ":generateContent")
                )
            else:
                url = f"{endpoint.rstrip('/')}/models/{model_name}:{action}"
            if stream and "alt=sse" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}alt=sse"

            sys_parts: list[str] = []
            contents: list[dict[str, Any]] = []
            for msg in messages:
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")
                imgs = msg.get("images")
                if role == "system":
                    if content:
                        sys_parts.append(str(content))
                    continue
                gem_role = "model" if role == "assistant" else "user"
                contents.append({
                    "role": gem_role,
                    "parts": format_gemini_parts(str(content), imgs),
                })

            payload = {"contents": contents}
            if sys_parts:
                payload["systemInstruction"] = {
                    "parts": [{"text": "\n\n".join(sys_parts)}],
                }
            gen_cfg: dict[str, Any] = {}
            if "temperature" in params:
                gen_cfg["temperature"] = params["temperature"]
            if "max_tokens" in params:
                gen_cfg["maxOutputTokens"] = params["max_tokens"]
            if response_schema is not None:
                gen_cfg["responseMimeType"] = "application/json"
                gen_cfg["responseSchema"] = response_schema
            elif json_mode:
                gen_cfg["responseMimeType"] = "application/json"
            if gen_cfg:
                payload["generationConfig"] = gen_cfg
            return dialect, url, payload

        url = endpoint.rstrip("/") + "/chat/completions"
        oai_messages = format_openai_messages(messages)
        payload = self._build_payload(
            oai_messages,
            stream=stream,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
        )
        return dialect, url, payload

    def _parse_dialect_response(
        self,
        dialect: str,
        data: dict[str, Any],
        *,
        elapsed: float,
        rate_snap: RateLimitSnapshot | None,
        return_tokens: bool = False,
    ) -> GenerateResult:
        """Parse a non-streaming response payload according to *dialect*."""
        model_name = self.config.get("model_name", "")
        if dialect == "anthropic":
            blocks = data.get("content", [])
            text = "".join(
                b.get("text", "")
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
            usage = data.get("usage", {})
            return GenerateResult(
                text=text,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                model_id=data.get("model") or model_name,
                provider="openapi",
                context_limit=self.get_context_limit(),
                finish_reason=data.get("stop_reason"),
                rate_limit=rate_snap,
                ttft_s=elapsed,
            )

        if dialect == "gemini":
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Empty candidates in Gemini response")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            usage = data.get("usageMetadata", {})
            return GenerateResult(
                text=text,
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                model_id=model_name,
                provider="openapi",
                context_limit=self.get_context_limit(),
                finish_reason=candidates[0].get("finishReason"),
                rate_limit=rate_snap,
                ttft_s=elapsed,
            )

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Empty choices in OpenAPI response")

        choice = choices[0]
        if "message" in choice:
            text = (choice["message"] or {}).get("content") or ""
        else:
            text = choice.get("text") or ""

        usage = data.get("usage", {})
        tokens_out, logprobs_out = _extract_logprobs(choice) if return_tokens else (None, None)

        return GenerateResult(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model_id=data.get("model") or model_name,
            provider="openapi",
            context_limit=self.get_context_limit(),
            finish_reason=choice.get("finish_reason"),
            tokens=tokens_out,
            logprobs=logprobs_out,
            rate_limit=rate_snap,
            ttft_s=elapsed,
        )

    @staticmethod
    def _extract_stream_chunk(
        dialect: str,
        chunk: dict[str, Any],
        in_tok: int,
        out_tok: int,
    ) -> tuple[str, int, int, bool]:
        """Extract ``(token_text, in_tok, out_tok, usage_updated)`` from an SSE JSON chunk."""
        if dialect == "anthropic":
            usage_updated = False
            ev_type = chunk.get("type")
            if ev_type == "message_start":
                msg_usage = (chunk.get("message") or {}).get("usage") or {}
                if msg_usage.get("input_tokens"):
                    in_tok = msg_usage["input_tokens"]
                    usage_updated = True
            elif ev_type == "message_delta":
                delta_usage = chunk.get("usage") or {}
                if delta_usage.get("output_tokens"):
                    out_tok = delta_usage["output_tokens"]
                    usage_updated = True
            elif ev_type == "content_block_delta":
                delta = chunk.get("delta") or {}
                if delta.get("type") == "text_delta":
                    return str(delta.get("text") or ""), in_tok, out_tok, usage_updated
            return "", in_tok, out_tok, usage_updated

        if dialect == "gemini":
            usage_updated = False
            um = chunk.get("usageMetadata")
            if isinstance(um, dict):
                in_tok = um.get("promptTokenCount", in_tok) or in_tok
                out_tok = um.get("candidatesTokenCount", out_tok) or out_tok
                usage_updated = True
            candidates = chunk.get("candidates") or []
            if not candidates:
                return "", in_tok, out_tok, usage_updated
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            return text, in_tok, out_tok, usage_updated

        usage_updated = False
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            in_tok = usage.get("prompt_tokens", in_tok) or in_tok
            out_tok = usage.get("completion_tokens", out_tok) or out_tok
            usage_updated = True
        choices = chunk.get("choices")
        if not choices:
            return "", in_tok, out_tok, usage_updated
        delta = choices[0].get("delta", {})
        return str(delta.get("content") or ""), in_tok, out_tok, usage_updated

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
        images: list[dict] | None = None,
    ) -> str:
        return self.generate_with_usage(
            prompt,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
            images=images,
        ).text

    def generate_with_usage(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
        images: list[dict] | None = None,
    ) -> GenerateResult:
        model_name = self.config.get("model_name", "")
        timeout = float(self.config.get("timeout", 120.0))
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        dialect, url, payload = self._build_dialect_request(
            [msg],
            stream=False,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
        )

        headers = self._build_headers()
        logger.debug("POST %s model=%s dialect=%s", url, model_name, dialect)

        t0 = time.monotonic()
        with self._http_client(timeout, client_factory=httpx.Client) as client:
            resp = client.post(url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise _classify_http_status_error(exc) from exc
            elapsed = time.monotonic() - t0
            rate_snap = _extract_response_snapshot(resp)
            data = resp.json()

        return self._parse_dialect_response(
            dialect,
            data,
            elapsed=elapsed,
            rate_snap=rate_snap,
            return_tokens=return_tokens,
        )

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        return_tokens: bool = False,
        response_schema: dict | None = None,
    ) -> GenerateResult:
        """Multi-turn generation honoring configured API dialect."""
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        dialect, url, payload = self._build_dialect_request(
            all_messages,
            stream=False,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
        )
        headers = self._build_headers()

        timeout = float(self.config.get("timeout", 120.0))
        t0 = time.monotonic()
        with self._http_client(timeout, client_factory=httpx.Client) as client:
            resp = client.post(url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise _classify_http_status_error(exc) from exc
            elapsed = time.monotonic() - t0
            rate_snap = _extract_response_snapshot(resp)
            data = resp.json()

        return self._parse_dialect_response(
            dialect,
            data,
            elapsed=elapsed,
            rate_snap=rate_snap,
            return_tokens=return_tokens,
        )

    def _iter_stream_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        response_schema: dict | None = None,
    ):
        timeout = float(self.config.get("timeout", 120.0))
        dialect, url, payload = self._build_dialect_request(
            messages,
            stream=True,
            json_mode=json_mode,
            response_schema=response_schema,
        )
        headers = self._build_headers()

        with self._http_client(timeout, client_factory=httpx.Client) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                status = getattr(resp, "status_code", 200)
                if isinstance(status, int) and status >= 400:
                    resp.read()
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise _classify_http_status_error(exc) from exc

                rate_snap = _extract_response_snapshot(resp)
                in_tok = 0
                out_tok = 0
                self._record_stream_result(
                    GenerateResult(text="", rate_limit=rate_snap),
                    simulated_stream=False,
                )
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    token, in_tok, out_tok, usage_updated = self._extract_stream_chunk(
                        dialect, chunk, in_tok, out_tok
                    )
                    if usage_updated:
                        self._record_stream_result(
                            GenerateResult(
                                text="",
                                input_tokens=in_tok,
                                output_tokens=out_tok,
                                rate_limit=rate_snap,
                            ),
                            simulated_stream=False,
                        )
                    if token:
                        yield token

    async def _aiter_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> AsyncIterator[str]:
        timeout = float(self.config.get("timeout", 120.0))
        dialect, url, payload = self._build_dialect_request(
            messages,
            stream=True,
            json_mode=json_mode,
            response_schema=response_schema,
        )
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                status = getattr(resp, "status_code", 200)
                if isinstance(status, int) and status >= 400:
                    await resp.aread()
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise _classify_http_status_error(exc) from exc

                rate_snap = _extract_response_snapshot(resp)
                in_tok = 0
                out_tok = 0
                self._record_stream_result(
                    GenerateResult(text="", rate_limit=rate_snap),
                    simulated_stream=False,
                )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        return
                    chunk = json.loads(data_str)
                    token, in_tok, out_tok, usage_updated = self._extract_stream_chunk(
                        dialect, chunk, in_tok, out_tok
                    )
                    if usage_updated:
                        self._record_stream_result(
                            GenerateResult(
                                text="",
                                input_tokens=in_tok,
                                output_tokens=out_tok,
                                rate_limit=rate_snap,
                            ),
                            simulated_stream=False,
                        )
                    if token:
                        yield token

    def generate_stream_sync(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ):
        """Synchronously stream tokens via SSE honoring the configured API dialect."""
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        yield from self._iter_stream_sync(
            [msg], json_mode=json_mode, response_schema=response_schema
        )

    async def generate_stream(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE honoring the configured API dialect."""
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        async for token in self._aiter_stream(
            [msg], json_mode=json_mode, response_schema=response_schema
        ):
            yield token

    async def generate_stream_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE with full history honoring the configured API dialect."""
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)
        async for token in self._aiter_stream(
            all_messages, json_mode=json_mode, response_schema=response_schema
        ):
            yield token

    def list_models(self) -> list[dict[str, Any]]:
        """Query ``GET /v1/models`` and return discovered model entries."""
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        url = endpoint.rstrip("/") + "/models"
        headers = self._build_headers()
        headers.pop("Content-Type", None)
        timeout = float(self.config.get("timeout", 10.0))
        with self._http_client(timeout, client_factory=httpx.Client) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        raw_models = data.get("data", []) if isinstance(data, dict) else []
        out: list[dict[str, Any]] = []
        for item in raw_models:
            if isinstance(item, dict) and item.get("id"):
                out.append({
                    "id": item["id"],
                    "model_id": item["id"],
                    "owned_by": item.get("owned_by", "openapi"),
                })
        return out

    def get_telemetry(self):
        """Query /v1/models to check endpoint liveness."""
        try:
            from aurarouter.auragrid.contracts import ModelState, ModelTelemetry
        except ImportError:
            return None
        model_name = self.config.get("model_name", "unknown")
        try:
            endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
            url = endpoint.rstrip("/") + "/models"
            headers = self._build_headers()
            headers.pop("Content-Type", None)
            resp = httpx.get(url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                return ModelTelemetry(
                    model_id=model_name,
                    provider_name=self.__class__.__name__,
                    state=ModelState.WARM,
                )
        except Exception as ex:  # noqa: F841
            logger.debug("providers.openapi.get_telemetry_error", exc_info=True)
        return ModelTelemetry(
            model_id=model_name,
            provider_name=self.__class__.__name__,
            state=ModelState.UNKNOWN,
        )
