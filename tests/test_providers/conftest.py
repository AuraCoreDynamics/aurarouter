"""Shared fixtures for provider tests."""

import sys
from types import ModuleType
from unittest.mock import MagicMock


def pytest_configure(config):
    """Inject a mock llama_cpp module if the real one isn't installed."""
    if "llama_cpp" not in sys.modules:
        mod = ModuleType("llama_cpp")
        mod.Llama = MagicMock  # type: ignore[attr-defined]
        sys.modules["llama_cpp"] = mod
