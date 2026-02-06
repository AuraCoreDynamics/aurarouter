from unittest.mock import patch, MagicMock

from aurarouter.providers.ollama import OllamaProvider


def test_generate_sends_post():
    cfg = {
        "endpoint": "http://localhost:11434/api/generate",
        "model_name": "qwen2.5-coder:7b",
        "parameters": {"temperature": 0.1},
    }
    provider = OllamaProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "def add(a, b): return a + b"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        result = provider.generate("write a function")

    assert "def add" in result


def test_json_mode_adds_format():
    cfg = {
        "endpoint": "http://localhost:11434/api/generate",
        "model_name": "test",
        "parameters": {},
    }
    provider = OllamaProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": '{"intent": "SIMPLE_CODE"}'}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    with patch("httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        provider.generate("classify", json_mode=True)

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
    assert payload.get("format") == "json"
