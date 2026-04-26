"""AuraRouter BitNet Backend Package — 1-bit LLM inference on CPU."""

from .metadata import METADATA
from .diagnostics import run_diagnostic
from .runtime import setup_runtime_environment

__all__ = ["METADATA", "run_diagnostic", "setup_runtime_environment", "get_catalog_artifact"]


def get_catalog_artifact() -> dict:
    """Return a catalog artifact dict for AuraRouter catalog registration."""
    return {
        "artifact_id": "aurarouter-bitnet",
        "kind": "model",
        "display_name": "BitNet 1.58-bit (CPU Ternary)",
        "capabilities": ["ternary-inference", "cpu-only", "edge-deployment"],
        "supported_intents": ["LOCAL_INFERENCE"],
        "spec": {
            "compute_type": "CPU",
            "flavor": "BitNet",
            "weight_bits": "1.58",
        },
    }
