"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.3.0"

import importlib.util

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

__all__ = ["ConfigLoader", "ComputeFabric", "__version__"]

# Conditionally import AuraGrid integration if SDK is available
# This allows aurarouter to work standalone without AuraGrid installed
if importlib.util.find_spec("aurarouter.auragrid"):
    __all__.append("auragrid")
