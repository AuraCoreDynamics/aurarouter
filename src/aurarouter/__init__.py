"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.5.8"

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from aurarouter.media import (
    ImageInput,
    VISION_OCR_SCHEMA,
    VISION_EXTRACTION_SCHEMA,
    VISION_SCHEMA_PRESETS,
)
from aurarouter.registry import RuntimeModelRegistry

__all__ = [
    "APIConfig",
    "AuraRouterAPI",
    "ConfigLoader",
    "ComputeFabric",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "ImageInput",
    "VISION_OCR_SCHEMA",
    "VISION_EXTRACTION_SCHEMA",
    "VISION_SCHEMA_PRESETS",
    "RuntimeModelRegistry",
    "__version__",
]
