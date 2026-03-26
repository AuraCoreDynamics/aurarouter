import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Optional

from aurarouter.savings.models import GenerateResult


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model_config: dict):
        self.config = model_config

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False,
                 response_schema: dict | None = None) -> str:
        """Send a prompt and return the text response."""
        ...

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> GenerateResult:
        """Generate a response with token-usage metadata.

        The default implementation delegates to ``generate()`` and returns
        zero token counts.  Providers that can report usage should override
        this method.
        """
        try:
            text = self.generate(prompt, json_mode=json_mode, response_schema=response_schema)
        except TypeError:
            # Backward compatibility: subclasses that haven't adopted response_schema yet
            text = self.generate(prompt, json_mode=json_mode)
        return GenerateResult(text=text)

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Session-aware generation with message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            system_prompt: Optional system instruction prepended to context
            json_mode: Request JSON-formatted output

        Returns:
            GenerateResult with text, token counts, and optional gist.

        Default implementation concatenates messages into a single prompt
        and calls generate_with_usage(). Providers should override for
        native multi-turn support (e.g., Ollama /api/chat).
        """
        parts = []
        if system_prompt:
            parts.append(f"[System]\n{system_prompt}\n")
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}\n")
        combined_prompt = "\n".join(parts)
        return self.generate_with_usage(combined_prompt, json_mode=json_mode)

    async def generate_stream(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens. Default: yield complete response from generate()."""
        try:
            result = self.generate(prompt, json_mode=json_mode, response_schema=response_schema)
        except TypeError:
            result = self.generate(prompt, json_mode=json_mode)
        yield result

    async def generate_stream_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        """Streaming variant of generate_with_history()."""
        result = self.generate_with_history(
            messages, system_prompt, json_mode=json_mode
        )
        yield result.text

    def get_context_limit(self) -> int:
        """Return the model's context window size in tokens.

        Default reads from config["context_limit"]. Providers may override
        to report dynamically (e.g., from model metadata).
        Returns 0 if unknown.
        """
        return self.config.get("context_limit", 0)

    def resolve_api_key(self) -> Optional[str]:
        """Resolve an API key from config value or environment variable."""
        key = self.config.get("api_key")
        if key and "YOUR_PASTED_KEY" not in str(key) and "YOUR_API_KEY" not in str(key):
            return key
        env_key = self.config.get("env_key")
        if env_key:
            return os.environ.get(env_key)
        return None
