"""Tests for OllamaProvider.generate_with_history."""
import json
from unittest.mock import patch, MagicMock
import httpx
import pytest

from aurarouter.providers.ollama import OllamaProvider


def _make_provider(config=None):
    default = {
        "model_name": "qwen2.5-coder:7b",
        "endpoint": "http://localhost:11434/api/generate",
        "parameters": {},
    }
    if config:
        default.update(config)
    return OllamaProvider(default)


def _mock_chat_response(text="hello", prompt_eval_count=10, eval_count=5):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    return resp


class TestGenerateWithHistory:
    def test_basic(self):
        provider = _make_provider()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
        ]
        mock_resp = _mock_chat_response("I'm good!")
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.generate_with_history(messages)
        assert result.text == "I'm good!"
        assert result.provider == "ollama"

    def test_system_prompt(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_chat_response("response")
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            provider.generate_with_history(messages, system_prompt="Be helpful")
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "Be helpful"

    def test_json_mode(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_chat_response('{"key": "value"}')
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            provider.generate_with_history(messages, json_mode=True)
            payload = mock_client.post.call_args[1]["json"]
            assert payload["format"] == "json"

    def test_empty_response_raises(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"message": {"content": ""}}
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=resp)))
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(RuntimeError, match="Empty response"):
                provider.generate_with_history(messages)


class TestResolveChatEndpoint:
    def test_default(self):
        provider = _make_provider()
        assert provider._resolve_chat_endpoint() == "http://localhost:11434/api/chat"

    def test_custom_port(self):
        provider = _make_provider({"endpoint": "http://myhost:9999/api/generate"})
        assert provider._resolve_chat_endpoint() == "http://myhost:9999/api/chat"


class TestGetContextLimit:
    def test_from_config(self):
        provider = _make_provider({"context_limit": 32768})
        assert provider.get_context_limit() == 32768

    def test_from_ollama_show(self):
        provider = _make_provider()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "model_info": {"general.context_length": 4096}
        }
        with patch("httpx.post", return_value=resp):
            assert provider.get_context_limit() == 4096
