"""Tests for LlamaCppProvider multi-turn session support."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.providers.llamacpp import LlamaCppProvider, LlamaCppServerCache


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Reset the module-level cache before each test."""
    import aurarouter.providers.llamacpp as mod
    old_cache = mod._cache
    mod._cache = LlamaCppServerCache()
    yield
    mod._cache._servers.clear()
    mod._cache = old_cache


@pytest.fixture
def fake_model(tmp_path):
    model = tmp_path / "test.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 100)
    return model


def _mock_chat_response(text="Reply", prompt_tokens=10, completion_tokens=5):
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class TestGenerateWithHistory:
    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_multi_turn_sends_full_history(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """generate_with_history sends full message list to /v1/chat/completions."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response("Turn 2 reply")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        cfg = {
            "model_path": str(fake_model),
            "_gguf_metadata": {"has_chat_template": True},
            "parameters": {},
        }
        provider = LlamaCppProvider(cfg)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = provider.generate_with_history(messages)

        assert result.text == "Turn 2 reply"
        payload = mock_client.post.call_args[1]["json"]
        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][2]["content"] == "How are you?"

    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_system_prompt_prepended(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """System prompt is prepended to the message list."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response("Ok")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        cfg = {
            "model_path": str(fake_model),
            "_gguf_metadata": {"has_chat_template": True},
            "parameters": {},
        }
        provider = LlamaCppProvider(cfg)

        messages = [{"role": "user", "content": "Hello"}]
        provider.generate_with_history(
            messages, system_prompt="You are a helpful assistant."
        )

        payload = mock_client.post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a helpful assistant."
        assert len(payload["messages"]) == 2

    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_json_mode_in_history(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """json_mode=True sets response_format in multi-turn request."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response('{"result": 42}')
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        cfg = {
            "model_path": str(fake_model),
            "_gguf_metadata": {"has_chat_template": True},
            "parameters": {},
        }
        provider = LlamaCppProvider(cfg)

        messages = [{"role": "user", "content": "Give me JSON"}]
        provider.generate_with_history(messages, json_mode=True)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["response_format"] == {"type": "json_object"}

    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_returns_usage_and_metadata(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """generate_with_history returns GenerateResult with full metadata."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response(
            "reply", prompt_tokens=25, completion_tokens=10
        )
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        cfg = {
            "model_path": str(fake_model),
            "_gguf_metadata": {"has_chat_template": True},
            "parameters": {"n_ctx": 4096},
        }
        provider = LlamaCppProvider(cfg)

        messages = [{"role": "user", "content": "test"}]
        result = provider.generate_with_history(messages)

        assert result.text == "reply"
        assert result.input_tokens == 25
        assert result.output_tokens == 10
        assert result.provider == "llamacpp"
        assert result.context_limit == 4096


class TestGetContextLimit:
    def test_from_n_ctx(self):
        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "parameters": {"n_ctx": 8192},
        })
        assert provider.get_context_limit() == 8192

    def test_from_config(self):
        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "context_limit": 16384,
            "parameters": {"n_ctx": 8192},
        })
        # config context_limit takes precedence
        assert provider.get_context_limit() == 16384
