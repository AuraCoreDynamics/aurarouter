"""Tests for analyzer-aware route_task and _call_remote_analyzer."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import _call_remote_analyzer, route_task
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
        **extra,
    }
    return cfg


def _make_fabric(config: ConfigLoader | None = None) -> ComputeFabric:
    cfg = config or _make_config()
    fabric = ComputeFabric(cfg)
    return fabric


def _mock_httpx_client(mock_response):
    """Create a mock httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# route_task — no config (legacy behaviour)
# ---------------------------------------------------------------------------


class TestRouteTaskNoConfig:
    def test_no_config_uses_builtin(self):
        """When config is None, route_task falls through to built-in logic."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 2})),
            GenerateResult(text="builtin result"),
        ]):
            result = route_task(fabric, None, task="hello", config=None)
            assert result == "builtin result"


class TestRouteTaskDefaultAnalyzer:
    def test_default_analyzer_uses_builtin(self):
        """When active analyzer is 'aurarouter-default', built-in logic runs."""
        cfg = _make_config(system={"active_analyzer": "aurarouter-default"})
        fabric = _make_fabric(cfg)
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 2})),
            GenerateResult(text="default result"),
        ]):
            result = route_task(fabric, None, task="hello", config=cfg)
            assert result == "default result"


class TestRouteTaskRemoteAnalyzerSuccess:
    def test_remote_analyzer_uses_ranked_models(self):
        """When remote analyzer succeeds and returns ranked_models, those are used."""
        cfg = _make_config(
            system={"active_analyzer": "remote-analyzer-1"},
            catalog={
                "remote-analyzer-1": {
                    "kind": "analyzer",
                    "display_name": "Remote",
                    "mcp_endpoint": "http://remote:8080/mcp",
                    "mcp_tool_name": "analyze_route",
                },
            },
        )
        fabric = _make_fabric(cfg)

        remote_result = {
            "ranked_models": ["model-a", "model-b"],
            "role": "coding",
        }

        async def _fake_call_remote(*args, **kwargs):
            return remote_result

        with patch(
            "aurarouter.mcp_tools._call_remote_analyzer",
            new=_fake_call_remote,
        ):
            with patch.object(
                fabric, "execute",
                return_value=GenerateResult(text="remote routed"),
            ):
                result = route_task(fabric, None, task="do something", config=cfg)
                assert result == "remote routed"


class TestRouteTaskRemoteAnalyzerFailure:
    def test_remote_analyzer_failure_falls_back(self):
        """When remote analyzer fails, built-in logic takes over."""
        cfg = _make_config(
            system={"active_analyzer": "remote-analyzer-1"},
            catalog={
                "remote-analyzer-1": {
                    "kind": "analyzer",
                    "display_name": "Remote",
                    "mcp_endpoint": "http://remote:8080/mcp",
                    "mcp_tool_name": "analyze_route",
                },
            },
        )
        fabric = _make_fabric(cfg)

        async def _fail_remote(*args, **kwargs):
            raise ConnectionError("connection failed")

        with patch(
            "aurarouter.mcp_tools._call_remote_analyzer",
            new=_fail_remote,
        ):
            with patch.object(fabric, "execute", side_effect=[
                GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 1})),
                GenerateResult(text="fallback result"),
            ]):
                result = route_task(fabric, None, task="hello", config=cfg)
                assert result == "fallback result"


class TestRouteTaskRemoteAnalyzerNoEndpoint:
    def test_no_endpoint_uses_builtin(self):
        """When active analyzer has no mcp_endpoint, built-in logic runs."""
        cfg = _make_config(
            system={"active_analyzer": "custom-local"},
            catalog={
                "custom-local": {
                    "kind": "analyzer",
                    "display_name": "Custom Local",
                    # No mcp_endpoint
                },
            },
        )
        fabric = _make_fabric(cfg)
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 2})),
            GenerateResult(text="local fallback"),
        ]):
            result = route_task(fabric, None, task="test", config=cfg)
            assert result == "local fallback"


class TestRouteTaskRemoteEmptyRankedModels:
    def test_empty_ranked_models_falls_back(self):
        """When remote analyzer returns empty ranked_models, falls back."""
        cfg = _make_config(
            system={"active_analyzer": "remote-analyzer-1"},
            catalog={
                "remote-analyzer-1": {
                    "kind": "analyzer",
                    "display_name": "Remote",
                    "mcp_endpoint": "http://remote:8080/mcp",
                    "mcp_tool_name": "analyze_route",
                },
            },
        )
        fabric = _make_fabric(cfg)

        async def _empty_result(*args, **kwargs):
            return {"ranked_models": [], "role": "coding"}

        with patch(
            "aurarouter.mcp_tools._call_remote_analyzer",
            new=_empty_result,
        ):
            with patch.object(fabric, "execute", side_effect=[
                GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 1})),
                GenerateResult(text="builtin fallback"),
            ]):
                result = route_task(fabric, None, task="test", config=cfg)
                assert result == "builtin fallback"


# ---------------------------------------------------------------------------
# route_task — multiple ranked_models iteration
# ---------------------------------------------------------------------------


class TestRouteTaskMultipleRankedModels:
    def test_iterates_ranked_models_uses_first_success(self):
        """When remote analyzer returns multiple ranked_models and first
        succeeds, only the first model is used.

        We patch asyncio.get_event_loop().run_until_complete to bypass
        the sync/async boundary that normally prevents coroutine execution
        in synchronous test contexts.
        """
        cfg = _make_config(
            system={"active_analyzer": "remote-multi"},
            catalog={
                "remote-multi": {
                    "kind": "analyzer",
                    "display_name": "Multi",
                    "mcp_endpoint": "http://remote:8080/mcp",
                    "mcp_tool_name": "analyze",
                },
            },
        )
        fabric = _make_fabric(cfg)

        remote_result = {
            "ranked_models": ["model-a", "model-b", "model-c"],
            "role": "coding",
        }

        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = remote_result

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            with patch.object(fabric, "execute",
                              return_value=GenerateResult(text="first model output")) as mock_exec:
                result = route_task(fabric, None, task="test", config=cfg)
                assert result == "first model output"
                # Should have been called once (first model succeeded)
                assert mock_exec.call_count == 1

    def test_first_model_returns_none_tries_next(self):
        """When first ranked model returns empty text, route_task tries next."""
        cfg = _make_config(
            system={"active_analyzer": "remote-retry"},
            catalog={
                "remote-retry": {
                    "kind": "analyzer",
                    "display_name": "Retry",
                    "mcp_endpoint": "http://remote:8080/mcp",
                    "mcp_tool_name": "analyze",
                },
            },
        )
        fabric = _make_fabric(cfg)

        remote_result = {
            "ranked_models": ["model-a", "model-b"],
            "role": "coding",
        }

        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = remote_result

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            with patch.object(fabric, "execute", side_effect=[
                GenerateResult(text=""),       # model-a returns empty
                GenerateResult(text="model-b output"),  # model-b succeeds
            ]) as mock_exec:
                result = route_task(fabric, None, task="test", config=cfg)
                assert result == "model-b output"
                assert mock_exec.call_count == 2


# ---------------------------------------------------------------------------
# _call_remote_analyzer
# ---------------------------------------------------------------------------


class TestCallRemoteAnalyzer:
    @pytest.mark.asyncio
    async def test_success_returns_parsed_dict(self):
        """Successful HTTP call returns parsed JSON result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"ranked_models": ["m1"], "role": "coding"},
        }

        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(mock_response)):
            result = await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", None
            )

        assert result is not None
        assert result["ranked_models"] == ["m1"]

    @pytest.mark.asyncio
    async def test_success_with_string_result(self):
        """When result is a JSON string, it gets parsed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": json.dumps({"ranked_models": ["m2"], "role": "reasoning"}),
        }

        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(mock_response)):
            result = await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", "some context"
            )

        assert result is not None
        assert result["ranked_models"] == ["m2"]

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self):
        """Non-200 status code returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(mock_response)):
            result = await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", None
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_includes_context_in_payload(self):
        """When context is provided, it is included in the payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"ranked_models": ["m1"]}}

        mock_client = _mock_httpx_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", "ctx"
            )

        # Verify the payload contained context
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["params"]["arguments"]["context"] == "ctx"

    @pytest.mark.asyncio
    async def test_no_context_omits_context_from_payload(self):
        """When context is None, context is not in the payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"ranked_models": ["m1"]}}

        mock_client = _mock_httpx_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", None
            )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert "context" not in payload["params"]["arguments"]

    @pytest.mark.asyncio
    async def test_timeout_exception_propagates(self):
        """httpx.TimeoutException propagates (caller catches it)."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.TimeoutException):
                await _call_remote_analyzer(
                    "http://remote:8080/mcp", "analyze", "task", None
                )

    @pytest.mark.asyncio
    async def test_malformed_json_response(self):
        """When response JSON has no 'result' key, returns empty dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected_key": "no result here"}

        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(mock_response)):
            result = await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", None
            )

        # result will be {} (from data.get("result", {}))
        assert result == {}

    @pytest.mark.asyncio
    async def test_jsonrpc_error_response(self):
        """When response has error instead of result, returns empty dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid Request"},
            "id": 1,
        }

        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(mock_response)):
            result = await _call_remote_analyzer(
                "http://remote:8080/mcp", "analyze", "task", None
            )

        # No "result" key -> data.get("result", {}) returns {}
        assert result == {}
