"""Tests for GridMcpClient (MCP JSON-RPC 2.0 transport)."""

import json

import httpx
import pytest
from unittest.mock import patch, MagicMock

from aurarouter.mcp_client.client import GridMcpClient


class TestGridMcpClientInit:
    def test_defaults(self):
        c = GridMcpClient("http://localhost:8080")
        assert c.base_url == "http://localhost:8080"
        assert c.name == "http://localhost:8080"
        assert c.connected is False
        assert c.get_tools() == []
        assert c.get_models() == []
        assert c.get_capabilities() == set()

    def test_custom_name(self):
        c = GridMcpClient("http://host:9000", name="myservice")
        assert c.name == "myservice"

    def test_strips_trailing_slash(self):
        c = GridMcpClient("http://host:9000/")
        assert c.base_url == "http://host:9000"


def _mock_httpx_client(post_side_effect):
    """Helper to create a mock httpx.Client context manager with POST."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post = post_side_effect
    return mock_client


def _jsonrpc_response(result=None, error=None, id_="abc"):
    """Build a mock JSON-RPC 2.0 response."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    body = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result or {}
    resp.json.return_value = body
    return resp


class TestGridMcpClientConnect:
    def test_connect_discovers_tools(self):
        """connect() sends JSON-RPC 2.0 tools/list and populates tools."""
        tools = [
            {"name": "auraxlm.query", "description": "RAG query"},
            {"name": "auraxlm.index", "description": "Index docs"},
        ]
        resp = _jsonrpc_response(result={"tools": tools})

        def mock_post(url, **kwargs):
            return resp

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            c = GridMcpClient("http://host:8080", name="test")
            assert c.connect() is True
            assert c.connected is True
            assert len(c.get_tools()) == 2
            assert c.get_tools()[0]["name"] == "auraxlm.query"

    def test_connect_sends_jsonrpc_envelope(self):
        """connect() POST body is valid JSON-RPC 2.0 with tools/list method."""
        calls = []

        def mock_post(url, **kwargs):
            calls.append((url, kwargs))
            return _jsonrpc_response(result={"tools": []})

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            c = GridMcpClient("http://host:8080")
            c.connect()

            assert len(calls) == 1
            url, kw = calls[0]
            assert url == "http://host:8080/mcp/message"
            body = kw["json"]
            assert body["jsonrpc"] == "2.0"
            assert body["method"] == "tools/list"
            assert "id" in body

    def test_connect_derives_capabilities_from_tool_names(self):
        """Capabilities are the set of discovered tool names."""
        tools = [
            {"name": "chain_reorder"},
            {"name": "rag_query"},
        ]
        resp = _jsonrpc_response(result={"tools": tools})

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(lambda url, **kw: resp)

            c = GridMcpClient("http://host:8080")
            c.connect()
            assert c.get_capabilities() == {"chain_reorder", "rag_query"}

    def test_connect_does_not_probe_for_models(self):
        """connect() only calls tools/list — no hardcoded model discovery."""
        tools = [{"name": "auraxlm.score_experts"}, {"name": "auraxlm.query"}]
        call_count = [0]

        def mock_post(url, **kwargs):
            call_count[0] += 1
            return _jsonrpc_response(result={"tools": tools})

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            c = GridMcpClient("http://host:8080")
            assert c.connect() is True
            assert c.get_models() == []
            # Only one POST call: tools/list (no model probe)
            assert call_count[0] == 1

    def test_connect_failure_is_graceful(self):
        """Top-level connection failure returns False, does not raise."""
        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(
                side_effect=Exception("Connection refused")
            )
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            c = GridMcpClient("http://unreachable:8080")
            assert c.connect() is False
            assert c.connected is False
            assert c.get_tools() == []
            assert c.get_models() == []

    def test_connect_jsonrpc_error_returns_false(self):
        """JSON-RPC error in tools/list response returns False."""
        resp = _jsonrpc_response(
            error={"code": -32601, "message": "Method not found"}
        )

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(lambda url, **kw: resp)

            c = GridMcpClient("http://host:8080")
            assert c.connect() is False


class TestGridMcpClientCallTool:
    def test_call_tool_not_connected_raises(self):
        c = GridMcpClient("http://host:8080")
        with pytest.raises(ConnectionError):
            c.call_tool("some_tool")

    def test_call_tool_sends_jsonrpc_envelope(self):
        """call_tool() sends JSON-RPC 2.0 tools/call with correct params."""
        calls = []
        c = GridMcpClient("http://host:8080")
        c._connected = True

        def mock_post(url, **kwargs):
            calls.append((url, kwargs))
            return _jsonrpc_response(result={"answer": "42"})

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            result = c.call_tool("auraxlm.query", prompt="test")
            assert result == {"answer": "42"}

            url, kw = calls[0]
            assert url == "http://host:8080/mcp/message"
            body = kw["json"]
            assert body["jsonrpc"] == "2.0"
            assert body["method"] == "tools/call"
            assert body["params"]["name"] == "auraxlm.query"
            assert body["params"]["arguments"] == {"prompt": "test"}

    def test_call_tool_raises_on_jsonrpc_error(self):
        """call_tool() raises RuntimeError on JSON-RPC error response."""
        c = GridMcpClient("http://host:8080")
        c._connected = True
        resp = _jsonrpc_response(
            error={"code": -32601, "message": "Method not found"}
        )

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(lambda url, **kw: resp)

            with pytest.raises(RuntimeError, match="Method not found"):
                c.call_tool("nonexistent.tool")

    def test_call_tool_http_error_propagates(self):
        c = GridMcpClient("http://host:8080")
        c._connected = True

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            mock_client.post.return_value = mock_resp
            mock_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                c.call_tool("bad_tool")


class TestDiscoverModels:
    def test_discover_models_success(self):
        """discover_models() calls specified tool and populates _models."""
        models = [{"id": "mistral-7b", "provider": "ollama"}]
        c = GridMcpClient("http://host:8080")
        c._connected = True

        def mock_post(url, **kwargs):
            return _jsonrpc_response(result=models)

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            result = c.discover_models("custom.list_models")
            assert len(result) == 1
            assert result[0]["id"] == "mistral-7b"
            assert c.get_models() == result

    def test_discover_models_non_list_result(self):
        """discover_models() returns empty list when result is not a list."""
        c = GridMcpClient("http://host:8080")
        c._connected = True

        def mock_post(url, **kwargs):
            return _jsonrpc_response(result={"error": "not a list"})

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            result = c.discover_models("bad_tool")
            assert result == []
            assert c.get_models() == []

    def test_discover_models_failure_graceful(self):
        """discover_models() handles exceptions gracefully."""
        c = GridMcpClient("http://host:8080")
        c._connected = True

        def mock_post(url, **kwargs):
            resp = _jsonrpc_response(
                error={"code": -32601, "message": "Method not found"}
            )
            return resp

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_post)

            result = c.discover_models("nonexistent.tool")
            assert result == []
            assert c.get_models() == []

    def test_get_models_empty_without_discovery(self):
        """get_models() returns empty list when no discovery has been called."""
        tools = [{"name": "some_tool"}]
        resp = _jsonrpc_response(result={"tools": tools})

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(lambda url, **kw: resp)

            c = GridMcpClient("http://host:8080")
            c.connect()
            assert c.get_models() == []
