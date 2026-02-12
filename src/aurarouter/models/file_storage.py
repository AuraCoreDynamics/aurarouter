"""Local file-based model storage with JSON registry.

Manages GGUF model files in a local directory (default: ~/.auracore/models/)
and maintains a registry (models.json) tracking downloaded models.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.FileModelStorage")

DEFAULT_MODEL_DIR = Path.home() / ".auracore" / "models"


class FileModelStorage:
    """Synchronous local file storage manager for GGUF models."""

    def __init__(self, models_dir: Optional[Path | str] = None):
        self._models_dir = Path(models_dir) if models_dir else DEFAULT_MODEL_DIR
        self._registry_path = self._models_dir / "models.json"
        self._registry: list[dict] = []
        self._load_registry()

    # ------------------------------------------------------------------
    # Registry I/O
    # ------------------------------------------------------------------

    def _load_registry(self) -> None:
        """Load registry from disk, or start empty."""
        if self._registry_path.is_file():
            try:
                with open(self._registry_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._registry = data
                else:
                    logger.warning("Registry file has unexpected format, starting fresh.")
                    self._registry = []
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Could not load registry: {exc}")
                self._registry = []
        else:
            self._registry = []

    def _save_registry(self) -> None:
        """Atomically write registry to disk."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._models_dir), suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._registry, f, indent=2)
            os.replace(tmp_path, str(self._registry_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return all registered models.

        Each entry contains:
        - repo: HuggingFace repo ID (or "unknown")
        - filename: GGUF file name
        - path: Full path to the file
        - size_bytes: File size in bytes
        - downloaded_at: ISO timestamp
        """
        return list(self._registry)

    def has_model(self, filename: str) -> bool:
        """Check if a model exists (in registry AND on filesystem)."""
        for entry in self._registry:
            if entry["filename"] == filename:
                return Path(entry["path"]).is_file()
        # Also check the directory directly
        return (self._models_dir / filename).is_file()

    def get_model_path(self, filename: str) -> Optional[Path]:
        """Resolve a filename to its full path, or None if not found."""
        # Check registry first
        for entry in self._registry:
            if entry["filename"] == filename:
                p = Path(entry["path"])
                if p.is_file():
                    return p
        # Fall back to directory check
        p = self._models_dir / filename
        if p.is_file():
            return p
        return None

    def register(self, repo: str, filename: str, path: Path) -> None:
        """Register a downloaded model in the registry.

        If an entry with the same filename already exists, it is updated.
        """
        path = Path(path)
        size_bytes = path.stat().st_size if path.is_file() else 0

        # Update existing entry if present
        for entry in self._registry:
            if entry["filename"] == filename:
                entry["repo"] = repo
                entry["path"] = str(path)
                entry["size_bytes"] = size_bytes
                entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
                self._save_registry()
                logger.info(f"Updated registry entry for {filename}")
                return

        # New entry
        self._registry.append({
            "repo": repo,
            "filename": filename,
            "path": str(path),
            "size_bytes": size_bytes,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save_registry()
        logger.info(f"Registered model: {filename}")

    def remove(self, filename: str, delete_file: bool = False) -> bool:
        """Remove model from registry. Optionally delete the file too.

        Returns True if the model was found and removed.
        """
        for i, entry in enumerate(self._registry):
            if entry["filename"] == filename:
                if delete_file:
                    p = Path(entry["path"])
                    if p.is_file():
                        p.unlink()
                        logger.info(f"Deleted model file: {p}")
                del self._registry[i]
                self._save_registry()
                logger.info(f"Removed {filename} from registry")
                return True
        return False

    def scan(self) -> int:
        """Scan models_dir for .gguf files not in registry and add them.

        Returns the number of newly registered files.
        """
        if not self._models_dir.is_dir():
            return 0

        known = {entry["filename"] for entry in self._registry}
        added = 0

        for p in self._models_dir.iterdir():
            if p.is_file() and p.suffix == ".gguf" and p.name not in known:
                self._registry.append({
                    "repo": "unknown",
                    "filename": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                })
                added += 1
                logger.info(f"Discovered unregistered model: {p.name}")

        if added:
            self._save_registry()

        return added

    @property
    def models_dir(self) -> Path:
        """The managed model directory."""
        return self._models_dir
