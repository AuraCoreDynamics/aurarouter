"""AuraRouter BitNet Backend Package — 1-bit LLM inference on CPU."""

from .metadata import METADATA
from .diagnostics import run_diagnostic
from .runtime import setup_runtime_environment

__all__ = ["METADATA", "run_diagnostic", "setup_runtime_environment"]
