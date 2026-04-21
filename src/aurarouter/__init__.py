"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.5.5"

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from aurarouter.registry import RuntimeModelRegistry

__all__ = [
    "APIConfig",
    "AuraRouterAPI",
    "ConfigLoader",
    "ComputeFabric",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "RuntimeModelRegistry",
    "__version__",
]
