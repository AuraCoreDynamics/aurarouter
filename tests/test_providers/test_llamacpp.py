"""Tests for aurarouter.providers.llamacpp (subprocess-backed provider)."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aurarouter.providers.llamacpp import LlamaCppProvider, LlamaCppServerCache


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Reset the module-level cache before each test."""
    import aurarouter.providers.llamacpp as mod
    old_cache = mod._cache
    mod._cache = LlamaCppServerCache()
    yield
    # Restore and don't trigger shutdown on test cache
    mod._cache._servers.clear()
    mod._cache = old_cache


@pytest.fixture
def fake_model(tmp_path):
    """Create a fake GGUF model file."""
    model = tmp_path / "test.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 100)
    return model


def _make_provider(cfg: dict) -> LlamaCppProvider:
    return LlamaCppProvider(cfg)


def _mock_chat_response(text="Hello!", prompt_tokens=10, completion_tokens=5):
    """Build a mock /v1/chat/completions response."""
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _mock_completion_response(text="Generated text", tokens_eval=10, tokens_pred=5):
    """Build a mock /completion response."""
    return {
        "content": text,
        "tokens_evaluated": tokens_eval,
        "tokens_predicted": tokens_pred,
    }


class TestGenerateWithChatTemplate:
    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_uses_chat_completion_with_template(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """POST to /v1/chat/completions when has_chat_template=True."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response("Hi there!")
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
        provider = _make_provider(cfg)
        result = provider.generate_with_usage("Say hi")

        assert result.text == "Hi there!"
        call_args = mock_client.post.call_args
        assert "/v1/chat/completions" in call_args[0][0]

    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_uses_completion_without_template(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """POST to /completion when has_chat_template=False."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_completion_response("output")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        cfg = {
            "model_path": str(fake_model),
            "_gguf_metadata": {"has_chat_template": False},
            "parameters": {},
        }
        provider = _make_provider(cfg)
        result = provider.generate_with_usage("Complete this")

        assert result.text == "output"
        call_args = mock_client.post.call_args
        assert "/completion" in call_args[0][0]


class TestJsonMode:
    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_json_mode_sets_response_format(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """json_mode=True sets response_format in chat request payload."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response('{"key": "value"}')
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
        provider = _make_provider(cfg)
        provider.generate_with_usage("Return JSON", json_mode=True)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["response_format"] == {"type": "json_object"}


class TestMissingModel:
    def test_missing_model_file_raises(self, tmp_path):
        """FileNotFoundError when model file doesn't exist."""
        cfg = {
            "model_path": str(tmp_path / "nonexistent.gguf"),
            "parameters": {},
        }
        provider = _make_provider(cfg)
        with pytest.raises(FileNotFoundError):
            provider.generate("test")


class TestCacheReuse:
    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_cache_reuses_model(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """ServerProcess.start() called only once for same model path."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response()
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
        provider = _make_provider(cfg)
        provider.generate("first")
        provider.generate("second")

        # ServerProcess should only be started once
        mock_server.start.assert_called_once()

    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_cache_uses_metadata_from_auto_tune(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """_gguf_metadata stash drives chat/completion selection."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_completion_response()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        # Auto-tune stashed metadata with no chat template
        cfg = {
            "model_path": str(fake_model),
            "_gguf_metadata": {"has_chat_template": False},
            "parameters": {},
        }
        provider = _make_provider(cfg)
        provider.generate("test")

        call_args = mock_client.post.call_args
        assert "/completion" in call_args[0][0]


class TestGenerateWithUsageTokens:
    @patch("aurarouter.providers.llamacpp.httpx.Client")
    @patch("aurarouter.providers.llamacpp.BinaryManager.resolve_server_binary")
    @patch("aurarouter.providers.llamacpp.ServerProcess")
    def test_returns_token_counts(
        self, MockSP, mock_resolve, MockClient, fake_model
    ):
        """Token counts are extracted from HTTP response."""
        mock_resolve.return_value = Path("/fake/llama-server")

        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint = "http://127.0.0.1:9999"
        MockSP.return_value = mock_server

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_chat_response(
            "response", prompt_tokens=42, completion_tokens=17
        )
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
        provider = _make_provider(cfg)
        result = provider.generate_with_usage("test")

        assert result.input_tokens == 42
        assert result.output_tokens == 17
        assert result.text == "response"
