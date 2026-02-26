"""Tests for GridMcpClient."""

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


def _mock_httpx_client(get_side_effect):
    """Helper to create a mock httpx.Client context manager."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = get_side_effect
    return mock_client


class TestGridMcpClientConnect:
    def test_connect_full_discovery(self):
        """All three endpoints respond successfully."""
        tools = [{"name": "search", "description": "Search documents"}]
        models = [{"id": "mistral-7b", "provider": "ollama"}]
        caps = ["chain_reorder", "rag_query"]

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/tools" in url:
                resp.json.return_value = tools
            elif "/models" in url:
                resp.json.return_value = models
            elif "/capabilities" in url:
                resp.json.return_value = caps
            return resp

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_get)

            c = GridMcpClient("http://host:8080", name="test")
            assert c.connect() is True
            assert c.connected is True
            assert len(c.get_tools()) == 1
            assert c.get_tools()[0]["name"] == "search"
            assert len(c.get_models()) == 1
            assert c.get_models()[0]["id"] == "mistral-7b"
            assert c.get_capabilities() == {"chain_reorder", "rag_query"}

    def test_connect_dict_wrapped_responses(self):
        """Endpoints return {tools: [...]} style dicts."""
        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/tools" in url:
                resp.json.return_value = {"tools": [{"name": "t1"}]}
            elif "/models" in url:
                resp.json.return_value = {"models": [{"id": "m1"}]}
            elif "/capabilities" in url:
                resp.json.return_value = {"capabilities": ["rag_query"]}
            return resp

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_get)

            c = GridMcpClient("http://host:8080")
            assert c.connect() is True
            assert len(c.get_tools()) == 1
            assert len(c.get_models()) == 1
            assert c.get_capabilities() == {"rag_query"}

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

    def test_connect_partial_discovery(self):
        """Tools endpoint fails, models succeeds, capabilities inferred."""
        def mock_get(url, **kwargs):
            if "/tools" in url:
                raise Exception("404 Not Found")
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/models" in url:
                resp.json.return_value = [{"id": "model1"}]
            elif "/capabilities" in url:
                raise Exception("404 Not Found")
            return resp

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_client(mock_get)

            c = GridMcpClient("http://host:8080")
            assert c.connect() is True
            assert c.get_tools() == []
            assert len(c.get_models()) == 1
            assert "models" in c.get_capabilities()
            assert "tools" not in c.get_capabilities()


class TestGridMcpClientCallTool:
    def test_call_tool_not_connected_raises(self):
        c = GridMcpClient("http://host:8080")
        with pytest.raises(ConnectionError):
            c.call_tool("some_tool")

    def test_call_tool_success(self):
        c = GridMcpClient("http://host:8080")
        c._connected = True

        with patch("aurarouter.mcp_client.client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"result": "ok"}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_cls.return_value = mock_client

            result = c.call_tool("search", query="hello")
            assert result == {"result": "ok"}
            mock_client.post.assert_called_once_with(
                "http://host:8080/api/v1/tools/search",
                json={"query": "hello"},
            )

    def test_call_tool_http_error_propagates(self):
        import httpx

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
