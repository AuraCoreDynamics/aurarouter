"""Phase 2 tests: Cross-MCP & Provider rate-limit/TTFT/error telemetry plumbing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
from aurarouter.providers.mcp_provider import McpProvider
from aurarouter.providers.openapi import OpenAPIProvider
from aurarouter.providers.protocol import REQUIRED_TOOLS, TOOL_GENERATE_WITH_HISTORY
from aurarouter.ratelimit.models import RateLimitSnapshot


class TestOpenAPITelemetry:
    def test_extracts_rate_limit_headers_and_ttft(self):
        provider = OpenAPIProvider({
            "endpoint": "http://localhost:8000/v1",
            "model_name": "gpt-4o",
        })
        resp = httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "500",
                "x-ratelimit-remaining-requests": "420",
                "x-ratelimit-reset-requests": "12s",
                "x-ratelimit-limit-tokens": "30000",
                "x-ratelimit-remaining-tokens": "25000",
                "x-ratelimit-reset-tokens": "6ms",
            },
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
            request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
        )
        with patch("httpx.Client") as mock_cls:
            client = MagicMock()
            client.post.return_value = resp
            mock_cls.return_value.__enter__.return_value = client
            res = provider.generate_with_usage("hi")

        assert res.text == "ok"
        assert res.ttft_s is not None and res.ttft_s >= 0.0
        assert isinstance(res.rate_limit, RateLimitSnapshot)
        assert res.rate_limit.rpm_limit == 500
        assert res.rate_limit.rpm_remaining == 420
        assert res.rate_limit.itpm_remaining == 25000

    def test_429_raises_rate_limit_exceeded_with_snapshot(self):
        provider = OpenAPIProvider({"endpoint": "http://localhost:8000/v1", "model_name": "gpt-4o"})
        resp = httpx.Response(
            429,
            headers={"Retry-After": "15", "x-ratelimit-remaining-requests": "0"},
            text='{"error": {"message": "Rate limit reached"}}',
            request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
        )
        with patch("httpx.Client") as mock_cls:
            client = MagicMock()
            client.post.return_value = resp
            mock_cls.return_value.__enter__.return_value = client
            with pytest.raises(RateLimitExceededError) as exc_info:
                provider.generate_with_usage("hi")

        assert exc_info.value.snapshot is not None
        assert exc_info.value.snapshot.retry_after_s == 15.0
        assert exc_info.value.snapshot.rpm_remaining == 0

    def test_400_context_overflow_raises_context_length_exceeded(self):
        provider = OpenAPIProvider({"endpoint": "http://localhost:8000/v1", "model_name": "gpt-4o"})
        resp = httpx.Response(
            400,
            text='{"error": {"code": "context_length_exceeded", "message": "This model maximum context length is 8192"}}',
            request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
        )
        with patch("httpx.Client") as mock_cls:
            client = MagicMock()
            client.post.return_value = resp
            mock_cls.return_value.__enter__.return_value = client
            with pytest.raises(ContextLengthExceededError):
                provider.generate_with_usage("huge prompt")

    def test_503_raises_provider_transient_error(self):
        provider = OpenAPIProvider({"endpoint": "http://localhost:8000/v1", "model_name": "gpt-4o"})
        resp = httpx.Response(
            503,
            text="Service Unavailable",
            request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
        )
        with patch("httpx.Client") as mock_cls:
            client = MagicMock()
            client.post.return_value = resp
            mock_cls.return_value.__enter__.return_value = client
            with pytest.raises(ProviderTransientError) as exc_info:
                provider.generate_with_usage("hi")
        assert exc_info.value.status_code == 503


class TestLlamaCppTelemetry:
    def test_slot_telemetry_and_prompt_ms_ttft(self):
        provider = LlamaCppServerProvider({
            "endpoint": "http://localhost:8080",
            "parameters": {"cache_prompt": True},
        })
        comp_resp = httpx.Response(
            200,
            json={
                "content": "hello",
                "tokens_evaluated": 50,
                "tokens_predicted": 10,
                "timings": {"prompt_ms": 125.0},
            },
            request=httpx.Request("POST", "http://localhost:8080/completion"),
        )
        with patch("httpx.Client") as mock_cls:
            client = MagicMock()
            client.post.return_value = comp_resp
            mock_cls.return_value.__enter__.return_value = client
            res = provider.generate_with_usage("hi")
            sent_payload = client.post.call_args[1]["json"]

        assert sent_payload["cache_prompt"] is True
        assert res.ttft_s == pytest.approx(0.125)

        slots_resp = httpx.Response(
            200,
            json=[
                {"id": 0, "is_processing": True},
                {"id": 1, "is_processing": True},
                {"id": 2, "is_processing": False},
                {"id": 3, "is_processing": False},
            ],
            request=httpx.Request("GET", "http://localhost:8080/slots"),
        )
        with patch("httpx.get", return_value=slots_resp):
            snap = provider.get_slot_telemetry()
        assert snap is not None
        assert snap.slot_utilization == pytest.approx(0.5)

    def test_context_exceed_400_raises_typed_error(self):
        provider = LlamaCppServerProvider({"endpoint": "http://localhost:8080"})
        resp = httpx.Response(
            400,
            text='{"error":{"type":"exceed_context_size_error","message":"request exceeds the available context size"}}',
            request=httpx.Request("POST", "http://localhost:8080/completion"),
        )
        with patch("httpx.Client") as mock_cls:
            client = MagicMock()
            client.post.return_value = resp
            mock_cls.return_value.__enter__.return_value = client
            with pytest.raises(ContextLengthExceededError):
                provider.generate_with_usage("huge")


class TestMcpProviderEnvelopeUnwrapping:
    def _make_mcp_provider(self) -> tuple[McpProvider, MagicMock]:
        with patch("aurarouter.providers.mcp_provider.GridMcpClient") as mock_cls:
            client = MagicMock()
            client.connected = True
            client.connect.return_value = True
            client.get_tools.return_value = [
                {"name": name} for name in REQUIRED_TOOLS | {TOOL_GENERATE_WITH_HISTORY}
            ]
            client.get_capabilities.return_value = set(REQUIRED_TOOLS | {TOOL_GENERATE_WITH_HISTORY})
            client.last_response_headers = {}
            mock_cls.return_value = client
            prov = McpProvider({
                "mcp_endpoint": "http://localhost:9100",
                "model_name": "claude-sonnet-4-20250514",
            })
            return prov, client

    def test_unwraps_fastmcp_json_text_and_parses_anthropic_headers(self):
        prov, client = self._make_mcp_provider()
        envelope = {
            "text": "Claude answer",
            "input_tokens": 120,
            "output_tokens": 45,
            "model_id": "claude-sonnet-4-20250514",
            "ttft_s": 0.42,
            "rate_limit": {
                "anthropic-ratelimit-requests-limit": "1000",
                "anthropic-ratelimit-requests-remaining": "850",
                "anthropic-ratelimit-input-tokens-limit": "80000",
                "anthropic-ratelimit-input-tokens-remaining": "60000",
                "anthropic-ratelimit-output-tokens-limit": "16000",
                "anthropic-ratelimit-output-tokens-remaining": "14000",
            },
        }
        # Simulate FastMCP content block wrapping a JSON string
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": json.dumps(envelope)}]
        }

        res = prov.generate_with_usage("Explain GCRA")
        assert res.text == "Claude answer"
        assert res.ttft_s == pytest.approx(0.42)
        assert isinstance(res.rate_limit, RateLimitSnapshot)
        assert res.rate_limit.rpm_remaining == 850
        assert res.rate_limit.itpm_remaining == 60000
        assert res.rate_limit.otpm_remaining == 14000

    def test_unwraps_gemini_daily_quota_error_envelope(self):
        prov, client = self._make_mcp_provider()
        err_envelope = {
            "error_type": "rate_limit_exceeded",
            "error": "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel",
            "rate_limit": {
                "retry_after_s": 3600.0,
                "is_daily_quota": True,
            },
        }
        client.call_tool.return_value = json.dumps(err_envelope)

        with pytest.raises(RateLimitExceededError) as exc_info:
            prov.generate_with_usage("hi")
        assert exc_info.value.snapshot is not None
        assert exc_info.value.snapshot.is_daily_quota is True
        assert exc_info.value.snapshot.retry_after_s == 3600.0

    def test_unwraps_context_length_error_envelope(self):
        prov, client = self._make_mcp_provider()
        client.call_tool.return_value = json.dumps({
            "error_type": "context_length_exceeded",
            "error": "prompt is too long: 210000 tokens > 200000 maximum",
            "max_tokens": 200000,
        })
        with pytest.raises(ContextLengthExceededError) as exc_info:
            prov.generate_with_history([{"role": "user", "content": "x"}])
        assert exc_info.value.max_tokens == 200000
