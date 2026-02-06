from __future__ import annotations

from typing import TYPE_CHECKING

from aurarouter.providers.base import BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.providers.google import GoogleProvider
from aurarouter.providers.claude import ClaudeProvider
from aurarouter.providers.llamacpp import LlamaCppProvider

if TYPE_CHECKING:
    pass

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "google": GoogleProvider,
    "claude": ClaudeProvider,
    "llamacpp": LlamaCppProvider,
}


def get_provider(name: str, model_config: dict) -> BaseProvider:
    """Look up a provider class by name and return an instance."""
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available: {', '.join(PROVIDER_REGISTRY)}"
        )
    return cls(model_config)
