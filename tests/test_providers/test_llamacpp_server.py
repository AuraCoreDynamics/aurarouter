from unittest.mock import patch, MagicMock

from aurarouter.providers.llamacpp_server import LlamaCppServerProvider


def test_generate_sends_post_to_completion():
    cfg = {
        "endpoint": "http://localhost:8080",
        "parameters": {"temperature": 0.1, "n_predict": 512},
    }
    provider = LlamaCppServerProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": "def add(a, b): return a + b"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        result = provider.generate("write a function")

    assert "def add" in result
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "http://localhost:8080/completion"
    payload = call_args[1]["json"]
    assert payload["temperature"] == 0.1
    assert payload["n_predict"] == 512


def test_json_mode_adds_json_schema():
    cfg = {
        "endpoint": "http://localhost:8080",
        "parameters": {},
    }
    provider = LlamaCppServerProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": '{"intent": "SIMPLE_CODE"}'}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        provider.generate("classify", json_mode=True)

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"]
    assert "json_schema" in payload


def test_default_endpoint():
    cfg = {"parameters": {}}
    provider = LlamaCppServerProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": "hello"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        provider.generate("test")

    call_args = mock_client.post.call_args
    assert call_args[0][0] == "http://localhost:8080/completion"


def test_provider_in_registry():
    from aurarouter.providers import PROVIDER_REGISTRY
    assert "llamacpp-server" in PROVIDER_REGISTRY
