"""Tests for ClaudeProvider.generate_with_history."""
from unittest.mock import patch, MagicMock
import pytest

from aurarouter.providers.claude import ClaudeProvider


def _make_provider(config=None):
    default = {
        "model_name": "claude-sonnet-4-5-20250929",
        "api_key": "sk-test-key",
        "parameters": {},
    }
    if config:
        default.update(config)
    return ClaudeProvider(default)


def _mock_message_response(text="hello"):
    msg = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    msg.content = [content_block]
    msg.usage = MagicMock()
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    return msg


class TestGenerateWithHistory:
    def test_basic(self):
        provider = _make_provider()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
        ]
        mock_msg = _mock_message_response("I'm good!")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            result = provider.generate_with_history(messages)
        assert result.text == "I'm good!"
        assert result.provider == "claude"
        assert result.input_tokens == 100

    def test_system_prompt(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "test"}]
        mock_msg = _mock_message_response("response")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            provider.generate_with_history(messages, system_prompt="Be helpful")
            call_kwargs = MockClient.return_value.messages.create.call_args[1]
            assert call_kwargs["system"] == "Be helpful"

    def test_filters_system_messages(self):
        provider = _make_provider()
        messages = [
            {"role": "system", "content": "You are a coder"},
            {"role": "user", "content": "hello"},
        ]
        mock_msg = _mock_message_response("hi")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            provider.generate_with_history(messages, system_prompt="Be helpful")
            call_kwargs = MockClient.return_value.messages.create.call_args[1]
            # System messages should be extracted to system parameter
            assert "You are a coder" in call_kwargs["system"]
            assert "Be helpful" in call_kwargs["system"]
            # Messages array should only contain user/assistant
            for m in call_kwargs["messages"]:
                assert m["role"] in ("user", "assistant")

    def test_no_api_key_raises(self):
        provider = ClaudeProvider({"model_name": "test", "parameters": {}})
        with pytest.raises(RuntimeError, match="API key"):
            provider.generate_with_history([{"role": "user", "content": "hi"}])


class TestGetContextLimit:
    def test_known_model(self):
        provider = _make_provider({"model_name": "claude-sonnet-4-5-20250929"})
        assert provider.get_context_limit() == 200000

    def test_config_override(self):
        provider = _make_provider({"context_limit": 100000})
        assert provider.get_context_limit() == 100000

    def test_prefix_match(self):
        provider = _make_provider({"model_name": "claude-sonnet-4-5-latest"})
        # Should match via prefix "claude-sonnet-4-5"
        assert provider.get_context_limit() == 200000

    def test_unknown_model(self):
        provider = _make_provider({"model_name": "claude-unknown-model"})
        assert provider.get_context_limit() == 0
