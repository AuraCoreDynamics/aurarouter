"""Tests for OpenAPIProvider.generate_with_history."""
from unittest.mock import patch, MagicMock

from aurarouter.providers.openapi import OpenAPIProvider


def _make_provider(config=None):
    default = {
        "endpoint": "http://localhost:8000/v1",
        "model_name": "test-model",
        "parameters": {},
    }
    if config:
        default.update(config)
    return OpenAPIProvider(default)


def _mock_response(text="hello"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
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
        mock_resp = _mock_response("I'm good!")
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.generate_with_history(messages)
        assert result.text == "I'm good!"
        assert result.provider == "openapi"
        assert result.input_tokens == 20

    def test_system_prompt(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_response("response")
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            provider.generate_with_history(messages, system_prompt="Be helpful")
            payload = mock_client.post.call_args[1]["json"]
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "Be helpful"
            assert payload["messages"][1]["role"] == "user"

    def test_json_mode(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_response('{"key": "value"}')
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            provider.generate_with_history(messages, json_mode=True)
            payload = mock_client.post.call_args[1]["json"]
            assert payload["response_format"] == {"type": "json_object"}

    def test_with_api_key(self):
        provider = _make_provider({"api_key": "sk-test123"})
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_response("ok")
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            provider.generate_with_history(messages)
            headers = mock_client.post.call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer sk-test123"
