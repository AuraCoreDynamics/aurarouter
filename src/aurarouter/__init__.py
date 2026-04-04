"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.5.4"

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

__all__ = ["APIConfig", "AuraRouterAPI", "ConfigLoader", "ComputeFabric", "__version__"]
