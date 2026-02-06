from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from aurarouter.providers.llamacpp import LlamaCppProvider, LlamaCppModelCache


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Replace the module-level cache with a fresh one for each test."""
    import aurarouter.providers.llamacpp as mod

    monkeypatch.setattr(mod, "_cache", LlamaCppModelCache())


def test_generate_calls_create_completion(tmp_path):
    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake")

    cfg = {
        "model_path": str(model_file),
        "parameters": {"temperature": 0.1, "max_tokens": 256},
    }
    provider = LlamaCppProvider(cfg)

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {
        "choices": [{"text": "def hello(): print('hi')"}]
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        result = provider.generate("write hello")

    assert "def hello" in result
    mock_llm.create_completion.assert_called_once()


def test_json_mode_sets_response_format(tmp_path):
    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake")

    cfg = {"model_path": str(model_file), "parameters": {}}
    provider = LlamaCppProvider(cfg)

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {
        "choices": [{"text": '{"key": "val"}'}]
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        provider.generate("classify", json_mode=True)

    call_kwargs = mock_llm.create_completion.call_args[1]
    assert call_kwargs.get("response_format") == {"type": "json_object"}


def test_missing_model_file_raises():
    cfg = {"model_path": "/nonexistent/model.gguf", "parameters": {}}
    provider = LlamaCppProvider(cfg)

    with pytest.raises(FileNotFoundError, match="GGUF model not found"):
        provider.generate("test")


def test_cache_reuses_model(tmp_path):
    model_file = tmp_path / "reuse.gguf"
    model_file.write_bytes(b"fake")

    cfg = {"model_path": str(model_file), "parameters": {}}

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {"choices": [{"text": "ok"}]}

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm) as MockLlama:
        p1 = LlamaCppProvider(cfg)
        p1.generate("a")
        p2 = LlamaCppProvider(cfg)
        p2.generate("b")

    # Llama constructor should only be called once (cached)
    assert MockLlama.call_count == 1
