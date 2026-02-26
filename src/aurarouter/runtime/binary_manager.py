"""Platform detection and llama-server binary resolution."""

import os
import platform
import sys
from pathlib import Path

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Runtime")


class BinaryManager:
    """Resolves platform-specific llama.cpp binaries.

    Search order for llama-server binary:
    1. ``AURAROUTER_LLAMACPP_BIN`` environment variable (user override)
    2. Bundled binary in ``aurarouter/bin/{platform}/`` (packaged with the wheel)
    """

    @staticmethod
    def detect_platform() -> str:
        """Return platform key: ``'win-x64'``, ``'linux-x64'``, or ``'macos-x64'``.

        Raises
        ------
        RuntimeError
            If the current platform or architecture is unsupported.
        """
        machine = platform.machine().lower()

        if sys.platform == "win32" and machine in ("amd64", "x86_64"):
            return "win-x64"
        elif sys.platform == "linux" and machine == "x86_64":
            return "linux-x64"
        elif sys.platform == "darwin" and machine == "x86_64":
            return "macos-x64"
        else:
            raise RuntimeError(
                f"Unsupported platform: sys.platform={sys.platform!r}, "
                f"machine={platform.machine()!r}. "
                "Supported: win-x64, linux-x64, macos-x64."
            )

    @staticmethod
    def get_bundled_bin_dir() -> Path:
        """Return the path to the bundled binary directory for this platform."""
        plat = BinaryManager.detect_platform()
        return Path(__file__).resolve().parent.parent / "bin" / plat

    @staticmethod
    def _binary_name() -> str:
        """Return the llama-server binary filename for this OS."""
        return "llama-server.exe" if sys.platform == "win32" else "llama-server"

    @classmethod
    def resolve_server_binary(cls, config: dict | None = None) -> Path:
        """Find the llama-server binary using the search order.

        Parameters
        ----------
        config:
            Optional model/system configuration dict (unused, kept for API
            compatibility).

        Returns
        -------
        Path
            Absolute path to the binary.

        Raises
        ------
        FileNotFoundError
            If no binary can be found in any search location.
        """
        binary_name = cls._binary_name()

        # 1. Environment variable override
        env_path = os.environ.get("AURAROUTER_LLAMACPP_BIN")
        if env_path:
            p = Path(env_path)
            if cls.validate_binary(p):
                logger.info("Using llama-server from AURAROUTER_LLAMACPP_BIN: %s", p)
                return p.resolve()
            logger.warning(
                "AURAROUTER_LLAMACPP_BIN=%s is not valid; falling back to bundled.",
                env_path,
            )

        # 2. Bundled binary (packaged in aurarouter/bin/{platform}/)
        try:
            bundled_dir = cls.get_bundled_bin_dir()
            bundled = bundled_dir / binary_name
            if cls.validate_binary(bundled):
                logger.info("Using bundled llama-server: %s", bundled)
                return bundled.resolve()
        except RuntimeError:
            pass  # Unsupported platform — skip bundled

        raise FileNotFoundError(
            f"llama-server binary not found. Searched:\n"
            f"  1. AURAROUTER_LLAMACPP_BIN environment variable\n"
            f"  2. Bundled directory: aurarouter/bin/<platform>/{binary_name}\n"
            f"Install binaries with: python scripts/fetch_llamacpp_binaries.py\n"
            f"Or set AURAROUTER_LLAMACPP_BIN=/path/to/{binary_name}"
        )

    @staticmethod
    def validate_binary(path: Path) -> bool:
        """Check that the binary exists and is executable."""
        if not path.is_file():
            return False
        if sys.platform != "win32":
            return os.access(path, os.X_OK)
        # On Windows, .exe files are always "executable"
        return True
