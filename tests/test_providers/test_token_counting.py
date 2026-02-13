"""Tests for generate_with_usage() token counting across providers."""

from unittest.mock import patch, MagicMock

from aurarouter.providers.base import BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.providers.google import GoogleProvider
from aurarouter.providers.claude import ClaudeProvider
from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
from aurarouter.savings.models import GenerateResult


# ── Ollama ──────────────────────────────────────────────────────────

def test_ollama_generate_with_usage():
    cfg = {
        "endpoint": "http://localhost:11434/api/generate",
        "model_name": "qwen2.5-coder:7b",
        "parameters": {},
    }
    provider = OllamaProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "response": "def add(a, b): return a + b",
        "prompt_eval_count": 25,
        "eval_count": 12,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        result = provider.generate_with_usage("write a function")

    assert isinstance(result, GenerateResult)
    assert result.text == "def add(a, b): return a + b"
    assert result.input_tokens == 25
    assert result.output_tokens == 12


def test_ollama_generate_with_usage_missing_counts():
    """Ollama response without token fields should default to 0."""
    cfg = {
        "endpoint": "http://localhost:11434/api/generate",
        "model_name": "test",
        "parameters": {},
    }
    provider = OllamaProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "hello"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        result = provider.generate_with_usage("hello")

    assert result.input_tokens == 0
    assert result.output_tokens == 0


# ── Google ──────────────────────────────────────────────────────────

def test_google_generate_with_usage():
    cfg = {
        "model_name": "gemini-2.0-flash",
        "api_key": "MOCK_KEY",
    }
    provider = GoogleProvider(cfg)

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 30
    mock_usage.candidates_token_count = 15

    mock_resp = MagicMock()
    mock_resp.text = "gemini output"
    mock_resp.usage_metadata = mock_usage

    with patch("aurarouter.providers.google.genai") as mock_genai:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        result = provider.generate_with_usage("test prompt")

    assert isinstance(result, GenerateResult)
    assert result.text == "gemini output"
    assert result.input_tokens == 30
    assert result.output_tokens == 15


def test_google_generate_with_usage_no_metadata():
    """Google response with None usage_metadata should default to 0."""
    cfg = {
        "model_name": "gemini-2.0-flash",
        "api_key": "MOCK_KEY",
    }
    provider = GoogleProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.text = "ok"
    mock_resp.usage_metadata = None

    with patch("aurarouter.providers.google.genai") as mock_genai:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        result = provider.generate_with_usage("test")

    assert result.input_tokens == 0
    assert result.output_tokens == 0


# ── Claude ──────────────────────────────────────────────────────────

def test_claude_generate_with_usage():
    cfg = {
        "model_name": "claude-sonnet-4-5-20250929",
        "api_key": "sk-test-123",
        "parameters": {"max_tokens": 1024},
    }
    provider = ClaudeProvider(cfg)

    mock_usage = MagicMock()
    mock_usage.input_tokens = 45
    mock_usage.output_tokens = 22

    block = MagicMock()
    block.text = "claude output"
    mock_msg = MagicMock()
    mock_msg.content = [block]
    mock_msg.usage = mock_usage

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg

    with patch("aurarouter.providers.claude.anthropic.Anthropic", return_value=mock_client):
        result = provider.generate_with_usage("test prompt")

    assert isinstance(result, GenerateResult)
    assert result.text == "claude output"
    assert result.input_tokens == 45
    assert result.output_tokens == 22


# ── LlamaCpp Server ─────────────────────────────────────────────────

def test_llamacpp_server_generate_with_usage():
    cfg = {
        "endpoint": "http://localhost:8080",
        "parameters": {},
    }
    provider = LlamaCppServerProvider(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "content": "server output",
        "tokens_evaluated": 18,
        "tokens_predicted": 9,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        result = provider.generate_with_usage("test prompt")

    assert isinstance(result, GenerateResult)
    assert result.text == "server output"
    assert result.input_tokens == 18
    assert result.output_tokens == 9


# ── Base fallback ───────────────────────────────────────────────────

def test_base_generate_with_usage_fallback():
    """Subclass that only implements generate() should still get
    generate_with_usage() returning GenerateResult with zero tokens."""

    class StubProvider(BaseProvider):
        def generate(self, prompt: str, json_mode: bool = False) -> str:
            return "stub output"

    provider = StubProvider({})
    result = provider.generate_with_usage("test")

    assert isinstance(result, GenerateResult)
    assert result.text == "stub output"
    assert result.input_tokens == 0
    assert result.output_tokens == 0
