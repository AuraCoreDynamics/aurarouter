"""Tests for McpProvider -- MCP adapter for remote provider servers."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from aurarouter.providers.mcp_provider import McpProvider
from aurarouter.providers.protocol import (
    TOOL_GENERATE,
    TOOL_GENERATE_WITH_HISTORY,
    TOOL_LIST_MODELS,
)
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_provider(
    endpoint: str = "http://localhost:9001",
    model_name: str = "test-model",
    **overrides,
) -> McpProvider:
    cfg = {
        "mcp_endpoint": endpoint,
        "model_name": model_name,
        **overrides,
    }
    return McpProvider(cfg)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestMcpProviderInit:
    def test_requires_endpoint(self):
        with pytest.raises(ValueError, match="mcp_endpoint"):
            McpProvider({"model_name": "test"})

    def test_accepts_mcp_endpoint(self):
        p = _make_provider(endpoint="http://host:9000")
        assert p._endpoint == "http://host:9000"

    def test_accepts_endpoint_key(self):
        p = McpProvider({"endpoint": "http://host:9000", "model_name": "m"})
        assert p._endpoint == "http://host:9000"

    def test_default_timeout(self):
        p = _make_provider()
        assert p._timeout == 120.0

    def test_custom_timeout(self):
        p = _make_provider(timeout=30)
        assert p._timeout == 30.0


# ---------------------------------------------------------------------------
# _ensure_connected
# ---------------------------------------------------------------------------

class TestEnsureConnected:
    def test_connect_failure_raises(self):
        p = _make_provider()
        with patch.object(p._client, "connect", return_value=False):
            with pytest.raises(ConnectionError, match="could not connect"):
                p._ensure_connected()

    def test_protocol_violation_raises(self):
        """Server missing required tools => RuntimeError."""
        p = _make_provider()
        with patch.object(p._client, "connect", return_value=True):
            # Server only has health_check, missing required tools
            with patch.object(
                p._client, "get_tools",
                return_value=[{"name": "provider.health_check"}],
            ):
                with pytest.raises(RuntimeError, match="does not satisfy"):
                    p._ensure_connected()

    def test_valid_connection_succeeds(self):
        p = _make_provider()
        with patch.object(p._client, "connect", return_value=True):
            with patch.object(
                p._client, "get_tools",
                return_value=[
                    {"name": TOOL_GENERATE},
                    {"name": TOOL_LIST_MODELS},
                ],
            ):
                with patch.object(
                    p._client, "get_capabilities",
                    return_value={TOOL_GENERATE, TOOL_LIST_MODELS},
                ):
                    p._ensure_connected()
                    assert p._validated is True

    def test_skips_if_already_validated(self):
        p = _make_provider()
        p._validated = True
        p._client._connected = True
        # Should not call connect()
        p._ensure_connected()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def _setup_connected(self, provider: McpProvider, capabilities=None):
        """Patch provider to appear connected and validated."""
        provider._validated = True
        provider._client._connected = True
        provider._remote_capabilities = capabilities or {
            TOOL_GENERATE, TOOL_LIST_MODELS
        }

    def test_generate_returns_text(self):
        p = _make_provider()
        self._setup_connected(p)

        with patch.object(
            p._client, "call_tool",
            return_value={
                "text": "Hello world",
                "input_tokens": 5,
                "output_tokens": 2,
            },
        ) as mock_call:
            result = p.generate("Say hello")
            assert result == "Hello world"
            mock_call.assert_called_once_with(
                TOOL_GENERATE,
                prompt="Say hello",
                model="test-model",
                json_mode=False,
            )

    def test_generate_with_usage_returns_generate_result(self):
        p = _make_provider()
        self._setup_connected(p)

        with patch.object(
            p._client, "call_tool",
            return_value={
                "text": "result text",
                "input_tokens": 10,
                "output_tokens": 20,
                "model_id": "remote-model",
                "context_limit": 8192,
            },
        ):
            result = p.generate_with_usage("prompt")
            assert isinstance(result, GenerateResult)
            assert result.text == "result text"
            assert result.input_tokens == 10
            assert result.output_tokens == 20
            assert result.provider == "mcp"

    def test_generate_with_usage_handles_string_result(self):
        p = _make_provider()
        self._setup_connected(p)

        with patch.object(p._client, "call_tool", return_value="plain text"):
            result = p.generate_with_usage("prompt")
            assert result.text == "plain text"
            assert result.provider == "mcp"

    def test_generate_json_mode(self):
        p = _make_provider()
        self._setup_connected(p)

        with patch.object(
            p._client, "call_tool",
            return_value={"text": "{}", "input_tokens": 0, "output_tokens": 0},
        ) as mock_call:
            p.generate("prompt", json_mode=True)
            mock_call.assert_called_once_with(
                TOOL_GENERATE,
                prompt="prompt",
                model="test-model",
                json_mode=True,
            )


# ---------------------------------------------------------------------------
# generate_with_history
# ---------------------------------------------------------------------------

class TestGenerateWithHistory:
    def _setup_connected(self, provider, capabilities):
        provider._validated = True
        provider._client._connected = True
        provider._remote_capabilities = capabilities

    def test_delegates_to_remote_when_supported(self):
        p = _make_provider()
        self._setup_connected(p, {
            TOOL_GENERATE, TOOL_LIST_MODELS, TOOL_GENERATE_WITH_HISTORY
        })

        messages = [{"role": "user", "content": "Hi"}]
        with patch.object(
            p._client, "call_tool",
            return_value={
                "text": "Hello!",
                "input_tokens": 1,
                "output_tokens": 1,
            },
        ) as mock_call:
            result = p.generate_with_history(messages, system_prompt="Be helpful")
            assert result.text == "Hello!"
            mock_call.assert_called_once_with(
                TOOL_GENERATE_WITH_HISTORY,
                messages=messages,
                system_prompt="Be helpful",
                model="test-model",
                json_mode=False,
            )

    def test_falls_back_to_base_when_not_supported(self):
        p = _make_provider()
        self._setup_connected(p, {TOOL_GENERATE, TOOL_LIST_MODELS})

        messages = [{"role": "user", "content": "Hi"}]
        # The fallback will call generate_with_usage (via super())
        with patch.object(
            p._client, "call_tool",
            return_value={"text": "fallback", "input_tokens": 0, "output_tokens": 0},
        ):
            result = p.generate_with_history(messages)
            assert isinstance(result, GenerateResult)
            assert "fallback" in result.text


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_list_models(self):
        p = _make_provider()
        p._validated = True
        p._client._connected = True
        p._remote_capabilities = {TOOL_GENERATE, TOOL_LIST_MODELS}

        models = [{"id": "m1"}, {"id": "m2"}]
        with patch.object(p._client, "call_tool", return_value=models):
            result = p.list_models()
            assert result == models

    def test_list_models_non_list_result(self):
        p = _make_provider()
        p._validated = True
        p._client._connected = True
        p._remote_capabilities = {TOOL_GENERATE, TOOL_LIST_MODELS}

        with patch.object(p._client, "call_tool", return_value="not a list"):
            result = p.list_models()
            assert result == []
