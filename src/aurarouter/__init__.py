"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.4.0"

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

__all__ = ["ConfigLoader", "ComputeFabric", "__version__"]
