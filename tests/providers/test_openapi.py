"""Unit tests for the OpenAPI-compatible provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aurarouter.providers.openapi import OpenAPIProvider


@pytest.fixture
def provider():
    return OpenAPIProvider({
        "endpoint": "http://localhost:8000/v1",
        "model_name": "test-model",
    })


class TestOpenAPIProvider:
    def test_generate_calls_chat_completions(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = provider.generate("test prompt")
            assert result == "Hello world"

            call_args = mock_client.post.call_args
            assert "/chat/completions" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["model"] == "test-model"
            assert payload["messages"][0]["content"] == "test prompt"

    def test_generate_with_usage_returns_tokens(self, provider):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "result text"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = provider.generate_with_usage("test")
            assert result.text == "result text"
            assert result.input_tokens == 10
            assert result.output_tokens == 5

    def test_json_mode_sets_response_format(self, provider):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}],
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            provider.generate("test", json_mode=True)
            payload = mock_client.post.call_args[1]["json"]
            assert payload["response_format"] == {"type": "json_object"}

    def test_api_key_sent_as_bearer(self):
        provider = OpenAPIProvider({
            "endpoint": "http://localhost:8000/v1",
            "model_name": "test",
            "api_key": "sk-test-key",
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            provider.generate("test")
            headers = mock_client.post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer sk-test-key"

    def test_empty_choices_raises(self, provider):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [], "usage": {}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(ValueError, match="Empty choices"):
                provider.generate("test")

    def test_completion_format_fallback(self, provider):
        """Support legacy /completions response format (text field instead of message)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": "completion result"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = provider.generate("test")
            assert result == "completion result"

    def test_provider_in_registry(self):
        from aurarouter.providers import PROVIDER_REGISTRY

        assert "openapi" in PROVIDER_REGISTRY
