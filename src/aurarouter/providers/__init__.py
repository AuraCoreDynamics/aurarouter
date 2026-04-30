from __future__ import annotations

from typing import TYPE_CHECKING

from aurarouter.providers.base import BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.providers.llamacpp import LlamaCppProvider
from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
from aurarouter.providers.openapi import OpenAPIProvider
from aurarouter.providers.mcp_provider import McpProvider
from aurarouter.providers.onnx import ONNXProvider

if TYPE_CHECKING:
    from aurarouter.config import ModelConfig


PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "llamacpp": LlamaCppProvider,
    "llamacpp-server": LlamaCppServerProvider,
    "openapi": OpenAPIProvider,
    "mcp": McpProvider,
    "onnx": ONNXProvider,
}


def get_provider(name: str, model_config: dict) -> BaseProvider:
    """Return a provider instance."""
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available: {', '.join(PROVIDER_REGISTRY)}"
        )
    return cls(model_config)
