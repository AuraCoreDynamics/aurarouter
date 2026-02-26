import os
from unittest.mock import patch, MagicMock

import pytest

from aurarouter.providers.google import GoogleProvider


def test_generate_calls_genai():
    cfg = {
        "model_name": "gemini-2.0-flash",
        "api_key": "test-key-123",
    }
    provider = GoogleProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.text = "print('hello')"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("aurarouter.providers.google.genai.Client", return_value=mock_client):
        result = provider.generate("write hello world")

    assert result == "print('hello')"
    mock_client.models.generate_content.assert_called_once()


def test_missing_api_key_raises():
    cfg = {"model_name": "gemini-2.0-flash"}
    provider = GoogleProvider(cfg)

    with pytest.raises(ValueError, match="No API key"):
        provider.generate("test")


def test_env_key_resolution():
    cfg = {"model_name": "gemini-2.0-flash", "env_key": "TEST_GOOGLE_KEY"}
    provider = GoogleProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.text = "code"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with (
        patch.dict(os.environ, {"TEST_GOOGLE_KEY": "from-env"}),
        patch("aurarouter.providers.google.genai.Client", return_value=mock_client) as MockClient,
    ):
        provider.generate("prompt")

    MockClient.assert_called_once_with(api_key="from-env")


def test_json_mode_sets_response_format():
    """Verify json_mode=True sets the correct response_mime_type for the API call."""
    cfg = {
        "model_name": "gemini-2.0-flash",
        "api_key": "test-key-123",
    }
    provider = GoogleProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.text = '{"message": "hello"}'

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("aurarouter.providers.google.genai.Client", return_value=mock_client):
        provider.generate("write json hello world", json_mode=True)

    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args
    gen_config = kwargs.get("config")
    assert gen_config is not None
    assert gen_config.response_mime_type == "application/json"
    
