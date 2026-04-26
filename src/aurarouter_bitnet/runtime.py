"""Runtime environment setup for the BitNet backend."""

import os
import platform
import sys
from pathlib import Path


def _resolve_bin_dir() -> Path:
    """Return the platform-specific bin/ directory."""
    machine = platform.machine().lower()

    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        plat = "win-x64"
    elif sys.platform == "linux" and machine == "x86_64":
        plat = "linux-x64"
    elif sys.platform == "darwin" and machine == "x86_64":
        plat = "macos-x64"
    elif sys.platform == "darwin" and machine == "arm64":
        plat = "macos-arm64"
    else:
        raise FileNotFoundError(
            f"No BitNet binaries available for {sys.platform}/{machine}"
        )

    return Path(__file__).parent / "bin" / plat


def setup_runtime_environment() -> Path:
    """Resolve the platform binary path and prepare the runtime environment.

    Returns the absolute Path to the llama-server binary.
    Raises FileNotFoundError if the binary is missing.
    """
    bin_dir = _resolve_bin_dir()
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    binary_path = bin_dir / binary_name

    if not binary_path.is_file():
        raise FileNotFoundError(
            f"BitNet binary not found: {binary_path}. "
            f"Place the llama-server binary in the bin/ directory."
        )

    # Add the bin dir to PATH so sibling shared libs are discoverable.
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    # On Windows, register DLL search path for any companion libraries.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(bin_dir))

    return binary_path
