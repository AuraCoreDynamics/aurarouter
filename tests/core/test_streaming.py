"""Tests for streaming support across providers, fabric, and API."""

import json

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aurarouter.providers.base import BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.providers.openapi import OpenAPIProvider
from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConcreteProvider(BaseProvider):
    """Minimal concrete provider for testing base class streaming defaults."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return f"complete:{prompt}"


async def _collect(aiter):
    """Collect all tokens from an async iterator into a list."""
    tokens = []
    async for t in aiter:
        tokens.append(t)
    return tokens


# ---------------------------------------------------------------------------
# Test 1: Base provider default generate_stream yields complete response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_base_provider_generate_stream_default():
    provider = ConcreteProvider({"model_name": "test"})
    tokens = await _collect(provider.generate_stream("hello"))
    assert tokens == ["complete:hello"]


@pytest.mark.asyncio
async def test_base_provider_generate_stream_with_history_default():
    provider = ConcreteProvider({"model_name": "test"})
    messages = [{"role": "user", "content": "hi"}]
    tokens = await _collect(
        provider.generate_stream_with_history(messages, system_prompt="be helpful")
    )
    # The default concatenates messages and calls generate_with_usage -> GenerateResult
    assert len(tokens) == 1
    assert "hi" in tokens[0]


# ---------------------------------------------------------------------------
# Test 2: Ollama provider yields tokens from mocked NDJSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_generate_stream_ndjson():
    """Mock httpx.AsyncClient to return NDJSON lines for /api/generate."""
    ndjson_lines = [
        json.dumps({"response": "Hello", "done": False}),
        json.dumps({"response": " world", "done": False}),
        json.dumps({"response": "", "done": True}),
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in ndjson_lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, json=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {
        "model_name": "llama3",
        "endpoint": "http://localhost:11434/api/generate",
    }
    provider = OllamaProvider(config)

    with patch("aurarouter.providers.ollama.httpx.AsyncClient", FakeClient):
        tokens = await _collect(provider.generate_stream("test prompt"))

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_ollama_generate_stream_with_history_ndjson():
    """Mock httpx.AsyncClient to return NDJSON lines for /api/chat."""
    ndjson_lines = [
        json.dumps({"message": {"content": "Hi"}, "done": False}),
        json.dumps({"message": {"content": " there"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in ndjson_lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, json=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {
        "model_name": "llama3",
        "endpoint": "http://localhost:11434/api/generate",
    }
    provider = OllamaProvider(config)

    with patch("aurarouter.providers.ollama.httpx.AsyncClient", FakeClient):
        tokens = await _collect(
            provider.generate_stream_with_history(
                [{"role": "user", "content": "hello"}]
            )
        )

    assert tokens == ["Hi", " there"]


# ---------------------------------------------------------------------------
# Test 3: OpenAPI provider yields tokens from mocked SSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openapi_generate_stream_sse():
    """Mock httpx.AsyncClient to return SSE lines for /v1/chat/completions."""
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, json=None, headers=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {
        "model_name": "gpt-4",
        "endpoint": "http://localhost:8000/v1",
    }
    provider = OpenAPIProvider(config)

    with patch("aurarouter.providers.openapi.httpx.AsyncClient", FakeClient):
        tokens = await _collect(provider.generate_stream("test"))

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_openapi_generate_stream_with_history_sse():
    """SSE streaming with message history."""
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"A"}}]}',
        'data: {"choices":[{"delta":{"content":"B"}}]}',
        "data: [DONE]",
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, json=None, headers=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {
        "model_name": "gpt-4",
        "endpoint": "http://localhost:8000/v1",
    }
    provider = OpenAPIProvider(config)

    with patch("aurarouter.providers.openapi.httpx.AsyncClient", FakeClient):
        tokens = await _collect(
            provider.generate_stream_with_history(
                [{"role": "user", "content": "hi"}], system_prompt="sys"
            )
        )

    assert tokens == ["A", "B"]


# ---------------------------------------------------------------------------
# Test 4: LlamaCpp provider yields tokens from mocked SSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llamacpp_generate_stream_sse():
    """Mock httpx.AsyncClient to return SSE lines for /completion."""
    sse_lines = [
        'data: {"content":"foo","stop":false}',
        'data: {"content":"bar","stop":false}',
        'data: {"content":"","stop":true}',
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, json=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {"endpoint": "http://localhost:8080"}
    provider = LlamaCppServerProvider(config)

    with patch("aurarouter.providers.llamacpp_server.httpx.AsyncClient", FakeClient):
        tokens = await _collect(provider.generate_stream("prompt"))

    assert tokens == ["foo", "bar"]


@pytest.mark.asyncio
async def test_llamacpp_generate_stream_with_history_sse():
    """SSE streaming for /v1/chat/completions on llama.cpp server."""
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"X"}}]}',
        'data: {"choices":[{"delta":{"content":"Y"}}]}',
        "data: [DONE]",
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url, json=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {"endpoint": "http://localhost:8080"}
    provider = LlamaCppServerProvider(config)

    with patch("aurarouter.providers.llamacpp_server.httpx.AsyncClient", FakeClient):
        tokens = await _collect(
            provider.generate_stream_with_history(
                [{"role": "user", "content": "hello"}]
            )
        )

    assert tokens == ["X", "Y"]


# ---------------------------------------------------------------------------
# Test 5: Fabric falls back on pre-yield failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fabric_execute_stream_fallback_on_pre_yield_failure():
    """If a provider fails before yielding any tokens, fabric tries next model."""
    config = MagicMock()
    config.get_role_chain.return_value = ["model_a", "model_b"]

    config_a = {"provider": "ollama", "model_name": "a"}
    config_b = {"provider": "ollama", "model_name": "b"}
    config.get_model_config.side_effect = lambda mid: (
        config_a if mid == "model_a" else config_b
    )

    fabric = ComputeFabric(config)

    # Provider A raises immediately (no tokens yielded)
    async def failing_stream(prompt, json_mode=False):
        raise ConnectionError("provider A down")
        # Make this an async generator
        yield  # pragma: no cover

    # Provider B succeeds
    async def success_stream(prompt, json_mode=False):
        yield "token1"
        yield "token2"

    provider_a = MagicMock()
    provider_a.generate_stream = failing_stream

    provider_b = MagicMock()
    provider_b.generate_stream = success_stream

    tried = []

    def on_tried(role, model_id, success, elapsed):
        tried.append((model_id, success))

    with patch(
        "aurarouter.fabric.get_provider",
        side_effect=lambda name, cfg: provider_a if cfg is config_a else provider_b,
    ):
        tokens = await _collect(
            fabric.execute_stream("coding", "hello", on_model_tried=on_tried)
        )

    assert tokens == ["token1", "token2"]
    assert ("model_a", False) in tried
    assert ("model_b", True) in tried


# ---------------------------------------------------------------------------
# Test 6: Fabric raises on post-yield failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fabric_execute_stream_raises_on_post_yield_failure():
    """If a provider fails AFTER yielding tokens, the error propagates."""
    config = MagicMock()
    config.get_role_chain.return_value = ["model_a", "model_b"]
    config.get_model_config.return_value = {"provider": "ollama", "model_name": "a"}

    fabric = ComputeFabric(config)

    async def partial_then_fail(prompt, json_mode=False):
        yield "partial"
        raise RuntimeError("mid-stream error")

    provider_a = MagicMock()
    provider_a.generate_stream = partial_then_fail

    with patch("aurarouter.fabric.get_provider", return_value=provider_a):
        with pytest.raises(RuntimeError, match="mid-stream error"):
            await _collect(fabric.execute_stream("coding", "hello"))


# ---------------------------------------------------------------------------
# Test 7: execute_direct_stream end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_direct_stream_end_to_end():
    """AuraRouterAPI.execute_direct_stream delegates to fabric.execute_stream."""
    from aurarouter.api import AuraRouterAPI

    # We mock the entire initialization and just test the streaming delegation
    with patch.object(AuraRouterAPI, "__init__", lambda self, *a, **kw: None):
        api = AuraRouterAPI.__new__(AuraRouterAPI)

        # Create a mock fabric with execute_stream
        async def mock_execute_stream(role, prompt, json_mode=False, on_model_tried=None):
            yield "stream"
            yield "ed"

        mock_fabric = MagicMock()
        mock_fabric.execute_stream = mock_execute_stream
        api._fabric = mock_fabric

        tokens = await _collect(
            api.execute_direct_stream("coding", "write code")
        )

    assert tokens == ["stream", "ed"]


# ---------------------------------------------------------------------------
# Test: Fabric returns error when no chain defined
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fabric_execute_stream_no_chain():
    """execute_stream yields error message when role has no model chain."""
    config = MagicMock()
    config.get_role_chain.return_value = []

    fabric = ComputeFabric(config)
    tokens = await _collect(fabric.execute_stream("nonexistent", "hello"))

    assert len(tokens) == 1
    assert "ERROR" in tokens[0]
