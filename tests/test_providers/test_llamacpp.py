from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from aurarouter.providers.llamacpp import LlamaCppProvider, LlamaCppModelCache


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Replace the module-level cache with a fresh one for each test."""
    import aurarouter.providers.llamacpp as mod

    monkeypatch.setattr(mod, "_cache", LlamaCppModelCache())


def _make_mock_llm(*, has_chat_template: bool = True):
    """Create a mock Llama instance with configurable metadata."""
    mock_llm = MagicMock()
    metadata = {"general.architecture": "llama"}
    if has_chat_template:
        metadata["tokenizer.chat_template"] = "{% ... %}"
    mock_llm.metadata = metadata
    return mock_llm


def test_generate_uses_chat_completion_with_template(tmp_path):
    """Models with a chat template should use create_chat_completion."""
    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake")

    cfg = {
        "model_path": str(model_file),
        "parameters": {"temperature": 0.1, "max_tokens": 256},
    }
    provider = LlamaCppProvider(cfg)

    mock_llm = _make_mock_llm(has_chat_template=True)
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "def hello(): print('hi')"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        result = provider.generate("write hello")

    assert "def hello" in result
    mock_llm.create_chat_completion.assert_called_once()
    mock_llm.create_completion.assert_not_called()


def test_generate_uses_completion_without_template(tmp_path):
    """Models without a chat template should fall back to create_completion."""
    model_file = tmp_path / "base.gguf"
    model_file.write_bytes(b"fake")

    cfg = {
        "model_path": str(model_file),
        "parameters": {"temperature": 0.1, "max_tokens": 256},
    }
    provider = LlamaCppProvider(cfg)

    mock_llm = _make_mock_llm(has_chat_template=False)
    mock_llm.create_completion.return_value = {
        "choices": [{"text": "plain text output"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        result = provider.generate("continue this")

    assert "plain text output" in result
    mock_llm.create_completion.assert_called_once()
    mock_llm.create_chat_completion.assert_not_called()


def test_json_mode_sets_response_format(tmp_path):
    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake")

    cfg = {"model_path": str(model_file), "parameters": {}}
    provider = LlamaCppProvider(cfg)

    mock_llm = _make_mock_llm(has_chat_template=True)
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"key": "val"}'}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        provider.generate("classify", json_mode=True)

    call_kwargs = mock_llm.create_chat_completion.call_args[1]
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

    mock_llm = _make_mock_llm(has_chat_template=True)
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm) as MockLlama:
        p1 = LlamaCppProvider(cfg)
        p1.generate("a")
        p2 = LlamaCppProvider(cfg)
        p2.generate("b")

    # Llama constructor should only be called once (cached)
    assert MockLlama.call_count == 1


def test_cache_uses_metadata_from_auto_tune(tmp_path):
    """When _gguf_metadata is stashed by auto_tune_model, use it instead of re-reading."""
    model_file = tmp_path / "tuned.gguf"
    model_file.write_bytes(b"fake")

    cfg = {
        "model_path": str(model_file),
        "parameters": {},
        "_gguf_metadata": {"has_chat_template": False},
    }
    provider = LlamaCppProvider(cfg)

    mock_llm = MagicMock()
    # Even though we set metadata on the mock, the _gguf_metadata override
    # should take precedence and result in create_completion being used.
    mock_llm.metadata = {"tokenizer.chat_template": "{% ... %}"}
    mock_llm.create_completion.return_value = {
        "choices": [{"text": "base mode"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        result = provider.generate("test")

    assert "base mode" in result
    mock_llm.create_completion.assert_called_once()


def test_generate_with_usage_returns_tokens(tmp_path):
    model_file = tmp_path / "tokens.gguf"
    model_file.write_bytes(b"fake")

    cfg = {"model_path": str(model_file), "parameters": {}}
    provider = LlamaCppProvider(cfg)

    mock_llm = _make_mock_llm(has_chat_template=True)
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "result"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 17},
    }

    with patch("aurarouter.providers.llamacpp.Llama", return_value=mock_llm):
        gen_result = provider.generate_with_usage("test")

    assert gen_result.text == "result"
    assert gen_result.input_tokens == 42
    assert gen_result.output_tokens == 17
