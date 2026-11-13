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

from aurarouter.media import format_openai_content, format_openai_messages

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

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        extra_hdrs = self.config.get("extra_headers")
        if isinstance(extra_hdrs, dict):
            for k, v in extra_hdrs.items():
                headers[str(k)] = str(v)
        api_key = self.resolve_api_key() or ""
        if api_key and "Authorization" not in headers and "api-key" not in headers:
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
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        model_name = self.config.get("model_name", "")
        timeout = float(self.config.get("timeout", 120.0))

        url = endpoint.rstrip("/") + "/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(
            [{"role": "user", "content": format_openai_content(prompt, images)}],
            stream=False,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
        )

        logger.debug("POST %s model=%s", url, model_name)

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

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Empty choices in OpenAPI response")

        # Support both chat and completion response formats.
        choice = choices[0]
        if "message" in choice:
            text = choice["message"].get("content") or ""
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

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        return_tokens: bool = False,
        response_schema: dict | None = None,
    ) -> GenerateResult:
        """Multi-turn generation via /v1/chat/completions with full history."""
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(format_openai_messages(messages))

        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        url = endpoint.rstrip("/") + "/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(
            all_messages,
            stream=False,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
        )

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

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Empty choices in OpenAPI response")

        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or choice.get("text") or ""
        usage = data.get("usage", {})
        tokens_out, logprobs_out = _extract_logprobs(choice) if return_tokens else (None, None)

        return GenerateResult(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model_id=data.get("model") or self.config.get("model_name", ""),
            provider="openapi",
            context_limit=self.get_context_limit(),
            finish_reason=choice.get("finish_reason"),
            tokens=tokens_out,
            logprobs=logprobs_out,
            rate_limit=rate_snap,
            ttft_s=elapsed,
        )

    def generate_stream_sync(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ):
        """Synchronously stream tokens via SSE from /v1/chat/completions."""
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        timeout = float(self.config.get("timeout", 120.0))
        url = endpoint.rstrip("/") + "/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(
            [{"role": "user", "content": format_openai_content(prompt, images)}],
            stream=True,
            json_mode=json_mode,
            response_schema=response_schema,
        )

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
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        in_tok = usage.get("prompt_tokens", in_tok) or in_tok
                        out_tok = usage.get("completion_tokens", out_tok) or out_tok
                        self._record_stream_result(
                            GenerateResult(
                                text="",
                                input_tokens=in_tok,
                                output_tokens=out_tok,
                                rate_limit=rate_snap,
                            ),
                            simulated_stream=False,
                        )
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

    async def generate_stream(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE from /v1/chat/completions."""
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        timeout = float(self.config.get("timeout", 120.0))
        url = endpoint.rstrip("/") + "/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(
            [{"role": "user", "content": format_openai_content(prompt, images)}],
            stream=True,
            json_mode=json_mode,
            response_schema=response_schema,
        )

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
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        in_tok = usage.get("prompt_tokens", in_tok) or in_tok
                        out_tok = usage.get("completion_tokens", out_tok) or out_tok
                        self._record_stream_result(
                            GenerateResult(
                                text="",
                                input_tokens=in_tok,
                                output_tokens=out_tok,
                                rate_limit=rate_snap,
                            ),
                            simulated_stream=False,
                        )
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

    async def generate_stream_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE from /v1/chat/completions with full history."""
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(format_openai_messages(messages))

        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        url = endpoint.rstrip("/") + "/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(
            all_messages,
            stream=True,
            json_mode=json_mode,
            response_schema=response_schema,
        )
        timeout = float(self.config.get("timeout", 120.0))

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
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
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
