"""Tests for GoogleProvider.generate_with_history."""
from unittest.mock import patch, MagicMock

from aurarouter.providers.google import GoogleProvider


def _make_provider(config=None):
    default = {
        "model_name": "gemini-2.0-flash",
        "api_key": "test-key",
        "parameters": {},
    }
    if config:
        default.update(config)
    return GoogleProvider(default)


def _mock_response(text="hello"):
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.prompt_token_count = 50
    resp.usage_metadata.candidates_token_count = 25
    return resp


class TestGenerateWithHistory:
    @patch("google.genai.Client")
    def test_basic(self, MockGenaiClient):
        provider = _make_provider()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
        ]
        mock_resp = _mock_response("I'm good!")
        MockGenaiClient.return_value.models.generate_content.return_value = mock_resp
        result = provider.generate_with_history(messages)
        assert result.text == "I'm good!"
        assert result.provider == "google"

    @patch("google.genai.Client")
    def test_system_prompt(self, MockGenaiClient):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_response("response")
        MockGenaiClient.return_value.models.generate_content.return_value = mock_resp
        provider.generate_with_history(messages, system_prompt="Be helpful")
        call_kwargs = MockGenaiClient.return_value.models.generate_content.call_args[1]
        assert call_kwargs.get("config") is not None

    @patch("google.genai.Client")
    def test_json_mode(self, MockGenaiClient):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_resp = _mock_response('{"key": "value"}')
        MockGenaiClient.return_value.models.generate_content.return_value = mock_resp
        provider.generate_with_history(messages, json_mode=True)
        # Should pass config with response_mime_type
        call_kwargs = MockGenaiClient.return_value.models.generate_content.call_args[1]
        assert call_kwargs.get("config") is not None


class TestGetContextLimit:
    def test_known_model(self):
        provider = _make_provider({"model_name": "gemini-2.0-flash"})
        assert provider.get_context_limit() == 1048576

    def test_config_override(self):
        provider = _make_provider({"context_limit": 500000})
        assert provider.get_context_limit() == 500000

    def test_unknown_model(self):
        provider = _make_provider({"model_name": "gemini-unknown"})
        assert provider.get_context_limit() == 0
