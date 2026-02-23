"""Tests for LlamaCppProvider.generate_with_history."""
from unittest.mock import patch, MagicMock
import pytest


def _mock_llama():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "test response"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 15},
    }
    return llm


class TestGenerateWithHistory:
    @patch("aurarouter.providers.llamacpp._cache")
    def test_basic(self, mock_cache):
        from aurarouter.providers.llamacpp import LlamaCppProvider

        llm = _mock_llama()
        mock_cache.get_or_load.return_value = llm

        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "parameters": {},
        })
        messages = [{"role": "user", "content": "hello"}]
        result = provider.generate_with_history(messages)
        assert result.text == "test response"
        assert result.provider == "llamacpp"
        llm.create_chat_completion.assert_called_once()

    @patch("aurarouter.providers.llamacpp._cache")
    def test_system_prompt(self, mock_cache):
        from aurarouter.providers.llamacpp import LlamaCppProvider

        llm = _mock_llama()
        mock_cache.get_or_load.return_value = llm

        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "parameters": {},
        })
        messages = [{"role": "user", "content": "test"}]
        provider.generate_with_history(messages, system_prompt="Be helpful")
        call_kwargs = llm.create_chat_completion.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "Be helpful"

    @patch("aurarouter.providers.llamacpp._cache")
    def test_json_mode(self, mock_cache):
        from aurarouter.providers.llamacpp import LlamaCppProvider

        llm = _mock_llama()
        mock_cache.get_or_load.return_value = llm

        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "parameters": {},
        })
        messages = [{"role": "user", "content": "test"}]
        provider.generate_with_history(messages, json_mode=True)
        call_kwargs = llm.create_chat_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("aurarouter.providers.llamacpp._cache")
    def test_kv_cache_reuse(self, mock_cache):
        from aurarouter.providers.llamacpp import LlamaCppProvider

        llm = _mock_llama()
        mock_cache.get_or_load.return_value = llm

        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "parameters": {},
        })
        # Call twice - same Llama instance should be used
        provider.generate_with_history([{"role": "user", "content": "first"}])
        provider.generate_with_history([{"role": "user", "content": "second"}])
        assert llm.create_chat_completion.call_count == 2
        assert mock_cache.get_or_load.call_count == 2  # Both calls go through cache


class TestGetContextLimit:
    def test_from_n_ctx(self):
        from aurarouter.providers.llamacpp import LlamaCppProvider
        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "parameters": {"n_ctx": 8192},
        })
        assert provider.get_context_limit() == 8192

    def test_from_config(self):
        from aurarouter.providers.llamacpp import LlamaCppProvider
        provider = LlamaCppProvider({
            "model_path": "/fake/model.gguf",
            "context_limit": 16384,
            "parameters": {"n_ctx": 8192},
        })
        # config context_limit takes precedence
        assert provider.get_context_limit() == 16384
