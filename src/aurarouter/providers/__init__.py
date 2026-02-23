from __future__ import annotations

from typing import TYPE_CHECKING

from aurarouter.providers.base import BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.providers.google import GoogleProvider
from aurarouter.providers.claude import ClaudeProvider
from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
from aurarouter.providers.openapi import OpenAPIProvider

# Conditionally import LlamaCppProvider if llama-cpp-python is available
try:
    from aurarouter.providers.llamacpp import LlamaCppProvider
    _llamacpp_available = True
except ImportError:
    _llamacpp_available = False
    LlamaCppProvider = None  # type: ignore

if TYPE_CHECKING:
    pass

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "google": GoogleProvider,
    "claude": ClaudeProvider,
    "llamacpp-server": LlamaCppServerProvider,
    "openapi": OpenAPIProvider,
}

# Only add llamacpp if available
if _llamacpp_available:
    PROVIDER_REGISTRY["llamacpp"] = LlamaCppProvider  # type: ignore


def get_provider(name: str, model_config: dict) -> BaseProvider:
    """Look up a provider class by name and return an instance."""
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available: {', '.join(PROVIDER_REGISTRY)}"
        )
    return cls(model_config)
