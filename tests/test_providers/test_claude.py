import os
from unittest.mock import patch, MagicMock

import pytest

from aurarouter.providers.claude import ClaudeProvider


def _mock_message(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def test_generate_calls_anthropic():
    cfg = {
        "model_name": "claude-sonnet-4-5-20250929",
        "api_key": "sk-test-123",
        "parameters": {"max_tokens": 1024},
    }
    provider = ClaudeProvider(cfg)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_message("def foo(): pass")

    with patch("aurarouter.providers.claude.anthropic.Anthropic", return_value=mock_client):
        result = provider.generate("write a function")

    assert result == "def foo(): pass"
    mock_client.messages.create.assert_called_once()


def test_json_mode_adds_system():
    cfg = {
        "model_name": "claude-sonnet-4-5-20250929",
        "api_key": "sk-test-123",
        "parameters": {},
    }
    provider = ClaudeProvider(cfg)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_message('{"intent":"SIMPLE_CODE"}')

    with patch("aurarouter.providers.claude.anthropic.Anthropic", return_value=mock_client):
        provider.generate("classify", json_mode=True)

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" in call_kwargs
    assert "JSON" in call_kwargs["system"]


def test_missing_api_key_raises():
    cfg = {"model_name": "claude-sonnet-4-5-20250929"}
    provider = ClaudeProvider(cfg)

    with pytest.raises(ValueError, match="No API key"):
        provider.generate("test")


def test_env_key_resolution():
    cfg = {
        "model_name": "claude-sonnet-4-5-20250929",
        "env_key": "TEST_ANTHROPIC_KEY",
        "parameters": {},
    }
    provider = ClaudeProvider(cfg)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_message("ok")

    with (
        patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "from-env"}),
        patch("aurarouter.providers.claude.anthropic.Anthropic", return_value=mock_client) as MockAnth,
    ):
        provider.generate("prompt")

    MockAnth.assert_called_once_with(api_key="from-env")
