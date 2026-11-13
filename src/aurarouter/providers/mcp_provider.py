"""McpProvider -- BaseProvider adapter for remote MCP provider servers.

Wraps a remote MCP server that implements the MCP Provider Protocol
(see :mod:`aurarouter.providers.protocol`) behind the standard
:class:`BaseProvider` interface, so the :class:`ComputeFabric` can
route to it transparently.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from aurarouter._logging import get_logger
from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.mcp_client.client import GridMcpClient
from aurarouter.providers.base import BaseProvider
from aurarouter.providers.protocol import (
    ERROR_TYPE_CONTEXT_LENGTH,
    ERROR_TYPE_KEY,
    ERROR_TYPE_RATE_LIMIT,
    ERROR_TYPE_TRANSIENT,
    RATE_LIMIT_ENVELOPE_KEY,
    TOOL_GENERATE,
    TOOL_GENERATE_WITH_HISTORY,
    TOOL_HEALTH_CHECK,
    TOOL_LIST_MODELS,
    TTFT_ENVELOPE_KEY,
    validate_provider_tools,
)
from aurarouter.ratelimit.models import RateLimitSnapshot
from aurarouter.ratelimit.parsers import parse_rate_limit_headers
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.McpProvider")


def _has_snapshot_data(snap: RateLimitSnapshot | None) -> bool:
    """Return True iff at least one rate-limit signal field is populated."""
    if snap is None:
        return False
    return snap.is_daily_quota or any(
        getattr(snap, f) is not None
        for f in snap.__dataclass_fields__
        if f != "is_daily_quota"
    )


def _merge_snapshots(
    primary: RateLimitSnapshot | None,
    secondary: RateLimitSnapshot | None,
) -> RateLimitSnapshot | None:
    """Coalesce normalized RateLimitSnapshot fields with parsed HTTP header fields."""
    if primary is None and secondary is None:
        return None
    if primary is None:
        return secondary if _has_snapshot_data(secondary) else None
    if secondary is None:
        return primary if _has_snapshot_data(primary) else None

    merged_kwargs: dict[str, Any] = {}
    for fname in RateLimitSnapshot.__dataclass_fields__:
        if fname == "is_daily_quota":
            merged_kwargs[fname] = bool(primary.is_daily_quota or secondary.is_daily_quota)
        else:
            p_val = getattr(primary, fname)
            merged_kwargs[fname] = p_val if p_val is not None else getattr(secondary, fname)

    merged = RateLimitSnapshot(**merged_kwargs)
    return merged if _has_snapshot_data(merged) else None


class McpProvider(BaseProvider):
    """Provider adapter that delegates to a remote MCP server.

    The remote server must implement at least ``provider.generate`` and
    ``provider.list_models``.  Optional tools like
    ``provider.generate_with_history`` are used when available.
    """

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)
        endpoint = model_config.get("mcp_endpoint") or model_config.get(
            "endpoint", ""
        )
        if not endpoint:
            raise ValueError(
                "McpProvider requires 'mcp_endpoint' or 'endpoint' in config"
            )
        self._endpoint = endpoint
        self._model_name = model_config.get("model_name", "")
        self._timeout = float(model_config.get("timeout", 120.0))
        self._client = GridMcpClient(
            base_url=self._endpoint,
            name=f"mcp-provider:{self._model_name or endpoint}",
            timeout=self._timeout,
        )
        self._remote_capabilities: set[str] = set()
        self._validated = False
        self._validation_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection & validation
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Connect to the remote MCP server and validate protocol compliance."""
        if self._validated and self._client.connected:
            return

        with self._validation_lock:
            if self._validated and self._client.connected:
                return

            if not self._client.connect():
                raise ConnectionError(
                    f"McpProvider could not connect to {self._endpoint}"
                )

            tools = self._client.get_tools()
            valid, errors = validate_provider_tools(tools)
            if not valid:
                raise RuntimeError(
                    f"MCP server at {self._endpoint} does not satisfy the "
                    f"provider protocol: {'; '.join(errors)}"
                )

            self._remote_capabilities = self._client.get_capabilities()
            self._validated = True

    # ------------------------------------------------------------------
    # Envelope normalization helpers (v2.0 §3.4)
    # ------------------------------------------------------------------

    def _unwrap_mcp_payload(self, raw: Any) -> Any:
        """Unwrap FastMCP content blocks or JSON-encoded dict payloads."""
        # FastMCP tools/call result often has {"content": [{"type": "text", "text": "{...}"}]}
        if isinstance(raw, dict) and "content" in raw and isinstance(raw["content"], list):
            blocks = raw["content"]
            if len(blocks) == 1 and isinstance(blocks[0], dict) and "text" in blocks[0]:
                raw = blocks[0]["text"]

        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict) and (
                        "text" in parsed or ERROR_TYPE_KEY in parsed
                    ):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return raw

    def _extract_snapshot(self, payload: dict) -> RateLimitSnapshot | None:
        """Extract RateLimitSnapshot from MCP payload dict or transport headers (QA 3.3)."""
        rl_raw = payload.get(RATE_LIMIT_ENVELOPE_KEY)
        if isinstance(rl_raw, RateLimitSnapshot):
            return rl_raw if _has_snapshot_data(rl_raw) else None
        if isinstance(rl_raw, dict):
            # Coalesce both normalized RateLimitSnapshot keys AND raw HTTP rate-limit headers
            direct_snap = RateLimitSnapshot.from_dict(rl_raw)
            hdr_snap = parse_rate_limit_headers(rl_raw)
            merged = _merge_snapshots(direct_snap, hdr_snap)
            if merged is not None:
                return merged

        # Fallback: inspect HTTP headers from the MCP transport call itself (e.g. IETF gateway)
        transport_headers = getattr(self._client, "last_response_headers", None)
        if isinstance(transport_headers, dict) and transport_headers:
            snap = parse_rate_limit_headers(transport_headers)
            if _has_snapshot_data(snap):
                return snap
        return None

    def _raise_if_error_envelope(self, payload: dict) -> None:
        """Raise typed provider exceptions when the MCP server returns an error envelope."""
        err_type = payload.get(ERROR_TYPE_KEY)
        if not err_type:
            return
        msg = str(payload.get("error") or payload.get("message") or err_type)
        if err_type == ERROR_TYPE_RATE_LIMIT:
            snapshot = self._extract_snapshot(payload)
            raise RateLimitExceededError(msg, snapshot=snapshot)
        if err_type == ERROR_TYPE_CONTEXT_LENGTH:
            max_tok = payload.get("max_tokens")
            raise ContextLengthExceededError(
                msg, max_tokens=int(max_tok) if isinstance(max_tok, int) else None
            )
        if err_type == ERROR_TYPE_TRANSIENT:
            code = payload.get("status_code")
            raise ProviderTransientError(
                msg, status_code=int(code) if isinstance(code, int) else None
            )
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
        images: list[dict] | None = None,
    ) -> str:
        """Single-shot generation via ``provider.generate``."""
        return self.generate_with_usage(
            prompt, json_mode=json_mode, response_schema=response_schema, images=images
        ).text

    def _generate_with_usage_detail(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ) -> tuple[GenerateResult, bool]:
        """Internal helper returning ``(GenerateResult, has_explicit_ttft)``."""
        self._ensure_connected()

        kwargs: dict = {
            "prompt": prompt,
            "model": self._model_name,
            "json_mode": json_mode,
        }
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        if images:
            kwargs["images"] = images

        t0 = time.monotonic()
        raw_result = self._client.call_tool(TOOL_GENERATE, **kwargs)
        elapsed = time.monotonic() - t0
        result = self._unwrap_mcp_payload(raw_result)

        if isinstance(result, dict):
            self._raise_if_error_envelope(result)
            snapshot = self._extract_snapshot(result)
            ttft = result.get(TTFT_ENVELOPE_KEY)
            has_explicit_ttft = ttft is not None
            return (
                GenerateResult(
                    text=result.get("text", ""),
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    model_id=result.get("model_id", self._model_name),
                    provider="mcp",
                    context_limit=result.get("context_limit", 0),
                    rate_limit=snapshot,
                    ttft_s=float(ttft) if has_explicit_ttft else elapsed,
                ),
                has_explicit_ttft,
            )

        if isinstance(result, str):
            return (
                GenerateResult(
                    text=result,
                    model_id=self._model_name,
                    provider="mcp",
                    ttft_s=elapsed,
                ),
                False,
            )

        return (
            GenerateResult(
                text=str(result),
                model_id=self._model_name,
                provider="mcp",
                ttft_s=elapsed,
            ),
            False,
        )

    def generate_with_usage(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
        images: list[dict] | None = None,
    ) -> GenerateResult:
        """Generate and return a :class:`GenerateResult` with usage and rate-limit metadata."""
        res, _ = self._generate_with_usage_detail(
            prompt, json_mode=json_mode, response_schema=response_schema, images=images
        )
        return res

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        return_tokens: bool = False,
        response_schema: dict | None = None,
    ) -> GenerateResult:
        """Multi-turn generation, delegating to remote if supported."""
        self._ensure_connected()

        if TOOL_GENERATE_WITH_HISTORY not in self._remote_capabilities:
            return super().generate_with_history(
                messages,
                system_prompt=system_prompt,
                json_mode=json_mode,
                return_tokens=return_tokens,
            )

        kwargs: dict[str, Any] = {
            "messages": messages,
            "system_prompt": system_prompt,
            "model": self._model_name,
            "json_mode": json_mode,
        }
        if response_schema is not None:
            kwargs["response_schema"] = response_schema

        t0 = time.monotonic()
        raw_result = self._client.call_tool(TOOL_GENERATE_WITH_HISTORY, **kwargs)
        elapsed = time.monotonic() - t0
        result = self._unwrap_mcp_payload(raw_result)

        if isinstance(result, dict):
            self._raise_if_error_envelope(result)
            snapshot = self._extract_snapshot(result)
            ttft = result.get(TTFT_ENVELOPE_KEY)
            return GenerateResult(
                text=result.get("text", ""),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                model_id=result.get("model_id", self._model_name),
                provider="mcp",
                context_limit=result.get("context_limit", 0),
                gist=result.get("gist"),
                rate_limit=snapshot,
                ttft_s=float(ttft) if ttft is not None else elapsed,
            )

        return GenerateResult(
            text=str(result),
            model_id=self._model_name,
            provider="mcp",
            ttft_s=elapsed,
        )

    async def generate_stream(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming adapter that preserves RateLimitSnapshot and avoids Vegas TTFT pollution."""
        res, has_explicit_ttft = await asyncio.to_thread(
            self._generate_with_usage_detail,
            prompt,
            json_mode,
            response_schema,
            images,
        )
        stream_meta = GenerateResult(
            text="",
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            model_id=res.model_id,
            provider=res.provider,
            context_limit=res.context_limit,
            rate_limit=res.rate_limit,
            ttft_s=res.ttft_s if has_explicit_ttft else None,
        )
        self._record_stream_result(stream_meta, simulated_stream=not has_explicit_ttft)
        yield res.text

    def generate_stream_sync(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        images: list[dict] | None = None,
    ):
        """Sync streaming adapter preserving RateLimitSnapshot and simulated_stream flag."""
        res, has_explicit_ttft = self._generate_with_usage_detail(
            prompt, json_mode=json_mode, response_schema=response_schema, images=images
        )
        stream_meta = GenerateResult(
            text="",
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            model_id=res.model_id,
            provider=res.provider,
            context_limit=res.context_limit,
            rate_limit=res.rate_limit,
            ttft_s=res.ttft_s if has_explicit_ttft else None,
        )
        self._record_stream_result(stream_meta, simulated_stream=not has_explicit_ttft)
        yield res.text

    async def generate_stream_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming variant of generate_with_history preserving RateLimitSnapshot."""
        res = await asyncio.to_thread(
            self.generate_with_history,
            messages,
            system_prompt,
            json_mode,
            False,
            response_schema,
        )
        stream_meta = GenerateResult(
            text="",
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            model_id=res.model_id,
            provider=res.provider,
            context_limit=res.context_limit,
            rate_limit=res.rate_limit,
            ttft_s=None,
        )
        self._record_stream_result(stream_meta, simulated_stream=True)
        yield res.text

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Call ``provider.list_models`` on the remote server."""
        self._ensure_connected()
        result = self._unwrap_mcp_payload(self._client.call_tool(TOOL_LIST_MODELS))
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        return result if isinstance(result, list) else []

    def get_capabilities(self) -> set[str]:
        """Return the remote server's advertised tool capabilities."""
        self._ensure_connected()
        return set(self._remote_capabilities)

    def get_telemetry(self):
        """Query remote MCP server health if health_check tool is available."""
        try:
            from aurarouter.auragrid.contracts import ModelState, ModelTelemetry
        except ImportError:
            return None
        model_id = self._model_name or self._endpoint
        try:
            self._ensure_connected()
            if TOOL_HEALTH_CHECK in self._remote_capabilities:
                result = self._client.call_tool(TOOL_HEALTH_CHECK)
                if isinstance(result, dict) and result.get("healthy"):
                    return ModelTelemetry(
                        model_id=model_id,
                        provider_name=self.__class__.__name__,
                        state=ModelState.WARM,
                    )
        except Exception:
            logger.debug("providers.mcp_provider.get_telemetry_error", exc_info=True)
        return ModelTelemetry(
            model_id=model_id,
            provider_name=self.__class__.__name__,
            state=ModelState.UNKNOWN,
        )
