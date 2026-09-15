"""Tests for LlamaCppServerProvider.generate_with_history."""
from unittest.mock import patch, MagicMock
import httpx

from aurarouter.providers.llamacpp_server import LlamaCppServerProvider


def _make_provider(config=None):
    default = {
        "endpoint": "http://localhost:8080",
        "parameters": {},
    }
    if config:
        default.update(config)
    return LlamaCppServerProvider(default)


def _mock_response(text="hello"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }
    return resp


class TestGenerateWithHistory:
    def test_basic(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "hello"}]
        mock_resp = _mock_response("world")
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.generate_with_history(messages)
        assert result.text == "world"
        assert result.provider == "llamacpp-server"

    def test_fallback_on_connect_error(self):
        provider = _make_provider()
        messages = [{"role": "user", "content": "hello"}]

        # Mock httpx.Client to raise ConnectError for /v1/chat/completions
        # but the fallback should call generate_with_usage via super()
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            # Also mock generate_with_usage for the fallback path
            with patch.object(provider, "generate_with_usage") as mock_gen:
                from aurarouter.savings.models import GenerateResult
                mock_gen.return_value = GenerateResult(text="fallback response")
                result = provider.generate_with_history(messages)
                assert result.text == "fallback response"
