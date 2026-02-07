"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.2.0"

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

# Conditionally import AuraGrid integration if SDK is available
# This allows aurarouter to work standalone without AuraGrid installed
try:
    from aurarouter import auragrid
    _auragrid_available = True
except ImportError:
    _auragrid_available = False

__all__ = ["ConfigLoader", "ComputeFabric", "__version__"]

if _auragrid_available:
    __all__.append("auragrid")
