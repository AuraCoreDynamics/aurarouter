"""AuraRouter: Multi-model MCP routing fabric for local and cloud LLMs."""

__version__ = "0.3.0"

import importlib.util
import sys

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

__all__ = ["ConfigLoader", "ComputeFabric", "__version__"]

# Make auragrid available as aurarouter.auragrid submodule
# This allows both "from auragrid import X" and "from aurarouter.auragrid import X"
if importlib.util.find_spec("auragrid"):
    import auragrid
    sys.modules["aurarouter.auragrid"] = auragrid
    __all__.append("auragrid")
