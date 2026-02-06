import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model_config: dict):
        self.config = model_config

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Send a prompt and return the text response."""
        ...

    def resolve_api_key(self) -> Optional[str]:
        """Resolve an API key from config value or environment variable."""
        key = self.config.get("api_key")
        if key and "YOUR_PASTED_KEY" not in str(key) and "YOUR_API_KEY" not in str(key):
            return key
        env_key = self.config.get("env_key")
        if env_key:
            return os.environ.get(env_key)
        return None
