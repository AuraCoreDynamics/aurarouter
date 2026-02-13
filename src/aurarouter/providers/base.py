import os
from abc import ABC, abstractmethod
from typing import Optional

from aurarouter.savings.models import GenerateResult


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model_config: dict):
        self.config = model_config

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Send a prompt and return the text response."""
        ...

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
        """Generate a response with token-usage metadata.

        The default implementation delegates to ``generate()`` and returns
        zero token counts.  Providers that can report usage should override
        this method.
        """
        text = self.generate(prompt, json_mode=json_mode)
        return GenerateResult(text=text)

    def resolve_api_key(self) -> Optional[str]:
        """Resolve an API key from config value or environment variable."""
        key = self.config.get("api_key")
        if key and "YOUR_PASTED_KEY" not in str(key) and "YOUR_API_KEY" not in str(key):
            return key
        env_key = self.config.get("env_key")
        if env_key:
            return os.environ.get(env_key)
        return None
