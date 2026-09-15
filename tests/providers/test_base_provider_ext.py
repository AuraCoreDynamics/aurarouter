"""Tests for BaseProvider extensions (generate_with_history, get_context_limit)."""

from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult


class _MockProvider(BaseProvider):
    def __init__(self, config=None):
        super().__init__(config or {})
        self._last_prompt = ""

    def generate(self, prompt, json_mode=False):
        self._last_prompt = prompt
        return "mock response"

    def generate_with_usage(self, prompt, json_mode=False):
        self._last_prompt = prompt
        return GenerateResult(text="mock response", input_tokens=10, output_tokens=5)


def test_generate_with_history_default():
    p = _MockProvider()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you"},
    ]
    result = p.generate_with_history(messages)
    assert result.text == "mock response"
    assert "[User]" in p._last_prompt
    assert "hello" in p._last_prompt
    assert "[Assistant]" in p._last_prompt
    assert "hi there" in p._last_prompt
    assert "how are you" in p._last_prompt


def test_generate_with_history_system_prompt():
    p = _MockProvider()
    messages = [{"role": "user", "content": "test"}]
    p.generate_with_history(messages, system_prompt="Be helpful.")
    assert "[System]" in p._last_prompt
    assert "Be helpful." in p._last_prompt


def test_get_context_limit_from_config():
    p = _MockProvider(config={"context_limit": 8192})
    assert p.get_context_limit() == 8192


def test_get_context_limit_default():
    p = _MockProvider(config={})
    assert p.get_context_limit() == 0
