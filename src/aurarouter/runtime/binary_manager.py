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
    def get_discovered_backends(cls) -> list[dict]:
        """Scan and return all discovered backend packages with their metadata and diagnostics."""
        discovered = []
        binary_name = cls._binary_name()
        plat = cls.detect_platform()

        import importlib.metadata
        import importlib.util

        for dist in importlib.metadata.distributions():
            pkg_name = dist.metadata["Name"].replace("-", "_")
            if pkg_name.startswith("aurarouter_") and pkg_name != "aurarouter":
                try:
                    spec = importlib.util.find_spec(pkg_name)
                    if not spec or not spec.origin: continue
                    
                    diag_mod = importlib.import_module(f"{pkg_name}.diagnostics")
                    meta_mod = importlib.import_module(f"{pkg_name}.metadata")
                    
                    diag = diag_mod.run_diagnostic()
                    meta = meta_mod.METADATA
                    bin_path = Path(spec.origin).parent / "bin" / plat / binary_name
                    
                    discovered.append({
                        "name": pkg_name,
                        "flavor": meta.get("flavor"),
                        "score": 0, # Score calculation omitted here for brevity or can be duplicated
                        "is_valid": cls.validate_binary(bin_path),
                        "diagnostics": diag,
                        "metadata": meta
                    })
                except: continue
        return discovered

    @staticmethod
    def _get_cache_path() -> Path:
        """Return the path to the backend selection cache file."""
        base = Path.home() / ".auracore" / "aurarouter"
        base.mkdir(parents=True, exist_ok=True)
        return base / "backend_cache.json"

    @classmethod
    def clear_backend_cache(cls) -> None:
        """Force a re-scan of hardware by deleting the backend cache."""
        cache_path = cls._get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Backend cache cleared. Hardware will be re-scanned on next start.")

    @classmethod
    def resolve_server_binary(cls, config: dict | None = None) -> Path:
        """Find the llama-server binary using dynamic discovery and caching.

        Search Order:
        1. Environment variable override
        2. Cached Backend selection (if still valid)
        3. Dynamically discovered Backend Packages (Ranked by capability)
        4. Bundled legacy directory
        """
        binary_name = cls._binary_name()

        # 1. Environment variable override
        env_path = os.environ.get("AURAROUTER_LLAMACPP_BIN")
        if env_path:
            p = Path(env_path)
            if cls.validate_binary(p):
                logger.info("Using llama-server from AURAROUTER_LLAMACPP_BIN: %s", p)
                return p.resolve()

        # 2. Check Cache
        import importlib.metadata
        import importlib.util
        import json

        cache_path = cls._get_cache_path()
        
        # Get current fingerprints (installed package names + versions)
        current_packages = {}
        for dist in importlib.metadata.distributions():
            pkg_name = dist.metadata["Name"].replace("-", "_")
            if pkg_name.startswith("aurarouter_") and pkg_name != "aurarouter":
                current_packages[pkg_name] = dist.version

        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                
                # Cache is valid if the set of installed packages is identical
                if cache_data.get("installed_packages") == current_packages:
                    cached_pkg = cache_data.get("selected_backend")
                    if cached_pkg in current_packages:
                        spec = importlib.util.find_spec(cached_pkg)
                        if spec and spec.origin:
                            pkg_path = Path(spec.origin).parent
                            plat = cls.detect_platform()
                            bin_path = pkg_path / "bin" / plat / binary_name
                            if cls.validate_binary(bin_path):
                                # Load the backend environment
                                module = importlib.import_module(cached_pkg)
                                if hasattr(module, "setup_runtime_environment"):
                                    module.setup_runtime_environment()
                                
                                logger.info(f"Using cached backend: {cached_pkg}")
                                return bin_path.resolve()
            except Exception as e:
                logger.debug(f"Failed to load backend cache: {e}")

        # 3. Dynamic Discovery of Backend Plugins
        plat = cls.detect_platform()
        discovered_backends = []

        for pkg_name in current_packages.keys():
            try:
                # Verify it has our required backend interface
                spec = importlib.util.find_spec(pkg_name)
                if not spec or not spec.origin:
                    continue

                # Import the backend to evaluate it
                module = importlib.import_module(pkg_name)
                
                # Check for diagnostics and metadata
                diag_mod = importlib.import_module(f"{pkg_name}.diagnostics")
                meta_mod = importlib.import_module(f"{pkg_name}.metadata")
                
                diag_results = diag_mod.run_diagnostic()
                meta = meta_mod.METADATA
                
                # Scoring logic
                score = 0
                comp_type = meta.get("compute_type", "")
                flavor = meta.get("flavor", "").lower()
                
                if comp_type == "GPU (NVIDIA)" and diag_results.get("gpu_present"):
                    # Prefer CUDA 13 over 12
                    base_score = 110 if "13" in flavor else 100
                    # Verify DLL health
                    if diag_results.get("dll_load_status") == "Success":
                        score = base_score
                    else:
                        score = 10 # Low priority if DLLs won't load
                elif "GPU" in comp_type: # Vulkan/Metal
                    score = 80
                elif comp_type == "CPU":
                    score = 50
                
                # If it passed basic binary check
                bin_path = Path(spec.origin).parent / "bin" / plat / binary_name
                if cls.validate_binary(bin_path):
                    discovered_backends.append({
                        "name": pkg_name,
                        "flavor": meta.get("flavor", "Unknown"),
                        "path": bin_path,
                        "score": score,
                        "module": module,
                        "metadata": meta,
                        "diagnostics": diag_results
                    })
                    logger.debug(f"Discovered backend {pkg_name} (Score: {score})")
            except (ImportError, KeyError, Exception) as e:
                logger.debug(f"Skipping potential backend {pkg_name}: {e}")
                continue

        # Sort by score descending and use the best one
        if discovered_backends:
            best = sorted(discovered_backends, key=lambda x: x["score"], reverse=True)[0]
            
            # Initialize the chosen backend's environment
            if hasattr(best["module"], "setup_runtime_environment"):
                best["module"].setup_runtime_environment()
                
            logger.info(f"Selected best backend: {best['name']} (Score: {best['score']})")
            
            # Save to cache
            try:
                with open(cache_path, "w") as f:
                    json.dump({
                        "installed_packages": current_packages,
                        "selected_backend": best["name"]
                    }, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save backend cache: {e}")

            return best["path"].resolve()

        # 4. Last Resort: Bundled directory (legacy/fallback)
        try:
            bundled_dir = cls.get_bundled_bin_dir()
            bundled = bundled_dir / binary_name
            if cls.validate_binary(bundled):
                logger.info("Using bundled fallback llama-server: %s", bundled)
                return bundled.resolve()
        except RuntimeError:
            pass

        raise FileNotFoundError(
            "No local llama.cpp backend found. AuraRouter requires a backend package.\n\n"
            "Please install one of the following:\n"
            "  - pip install aurarouter-cuda13  (NVIDIA RTX 50/40/30/20 series)\n"
            "  - pip install aurarouter-cuda12  (Older NVIDIA cards)\n"
            "  - pip install aurarouter-win-cpu (Windows CPU only)\n"
            "  - pip install aurarouter-macos   (MacOS)\n"
            "  - pip install aurarouter-vulkan  (Linux/Generic GPU)"
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
