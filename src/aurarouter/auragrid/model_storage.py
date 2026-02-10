"""
Grid storage for GGUF model distribution via IStorageProvider.

Provides chunked upload/download, manifest management, and checksum verification
for large model files distributed across AuraGrid.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.ModelStorage")

# 64MB chunks for large model uploads/downloads
CHUNK_SIZE = 64 * 1024 * 1024


class GridModelStorage:
    """Manages model storage on AuraGrid with chunked upload/download."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize grid model storage.

        Args:
            cache_dir: Local directory for model caching (default: ~/.aurarouter/models)
        """
        self._storage = None
        self._grid_storage_available = False
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".aurarouter" / "models"

    async def start(self) -> None:
        """Initialize storage provider connection."""
        if not self._storage:
            await self._init_storage()

    async def _init_storage(self) -> None:
        """Initialize connection to AuraGrid storage provider."""
        try:
            from auragrid.sdk.storage import get_storage_provider
            
            self._storage = await get_storage_provider()
            self._grid_storage_available = True
            logger.info("Connected to AuraGrid storage provider")

        except ImportError:
            self._grid_storage_available = False
            logger.debug("AuraGrid SDK not available, grid storage disabled")

        except Exception as e:
            self._grid_storage_available = False
            logger.error(f"Error initializing storage provider: {e}", exc_info=True)

    async def upload_model(self, local_path: str, model_id: str) -> bool:
        """
        Upload a model file to grid storage with chunked upload and manifest.

        Creates a manifest at models/{model_id}/_manifest.json containing:
        - file_name
        - total_size
        - chunk_count
        - checksum (SHA256)

        Args:
            local_path: Path to local model file
            model_id: Unique identifier for the model

        Returns:
            True if upload succeeded, False otherwise
        """
        if not self._grid_storage_available or not self._storage:
            logger.warning("Grid storage not available, upload skipped")
            return False

        try:
            local_file = Path(local_path)
            if not local_file.exists():
                logger.error(f"Local file not found: {local_path}")
                return False

            file_name = local_file.name
            file_size = local_file.stat().st_size
            
            logger.info(f"Uploading model {model_id} ({file_size} bytes) in {CHUNK_SIZE} byte chunks")

            # Calculate checksum and upload in chunks
            hasher = hashlib.sha256()
            chunk_index = 0

            with open(local_path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    hasher.update(chunk)
                    
                    # Upload chunk
                    chunk_path = f"models/{model_id}/chunks/{chunk_index:04d}.bin"
                    await self._storage.write_async(chunk_path, chunk)
                    
                    logger.debug(f"Uploaded chunk {chunk_index} for {model_id}")
                    chunk_index += 1

            checksum = hasher.hexdigest()

            # Create and upload manifest
            manifest = {
                "file_name": file_name,
                "total_size": file_size,
                "chunk_count": chunk_index,
                "chunk_size": CHUNK_SIZE,
                "checksum": checksum,
            }

            manifest_path = f"models/{model_id}/_manifest.json"
            manifest_data = json.dumps(manifest).encode("utf-8")
            await self._storage.write_async(manifest_path, manifest_data)

            logger.info(f"Model {model_id} uploaded successfully (checksum: {checksum[:8]}...)")
            return True

        except Exception as e:
            logger.error(f"Error uploading model {model_id}: {e}", exc_info=True)
            return False

    async def download_model(self, model_id: str, local_path: str) -> bool:
        """
        Download a model from grid storage with chunked download and checksum verification.

        Args:
            model_id: Unique identifier for the model
            local_path: Path where model should be saved locally

        Returns:
            True if download and verification succeeded, False otherwise
        """
        if not self._grid_storage_available or not self._storage:
            logger.warning("Grid storage not available, download skipped")
            return False

        try:
            # Download and parse manifest
            manifest_path = f"models/{model_id}/_manifest.json"
            manifest_data = await self._storage.read_async(manifest_path)
            manifest = json.loads(manifest_data.decode("utf-8"))

            chunk_count = manifest["chunk_count"]
            expected_checksum = manifest["checksum"]
            total_size = manifest["total_size"]

            logger.info(f"Downloading model {model_id} ({total_size} bytes, {chunk_count} chunks)")

            # Ensure cache directory exists
            self._ensure_cache_dir(local_path)

            # Download chunks and verify checksum
            hasher = hashlib.sha256()

            with open(local_path, "wb") as f:
                for chunk_index in range(chunk_count):
                    chunk_path = f"models/{model_id}/chunks/{chunk_index:04d}.bin"
                    chunk = await self._storage.read_async(chunk_path)
                    
                    hasher.update(chunk)
                    f.write(chunk)
                    
                    logger.debug(f"Downloaded chunk {chunk_index}/{chunk_count} for {model_id}")

            # Verify checksum
            actual_checksum = hasher.hexdigest()
            if actual_checksum != expected_checksum:
                logger.error(
                    f"Checksum mismatch for {model_id}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )
                # Clean up corrupted file
                Path(local_path).unlink(missing_ok=True)
                return False

            logger.info(f"Model {model_id} downloaded and verified successfully")
            return True

        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}", exc_info=True)
            # Clean up partial file
            Path(local_path).unlink(missing_ok=True)
            return False

    async def has_model(self, model_id: str) -> bool:
        """
        Check if a model exists in grid storage.

        Args:
            model_id: Unique identifier for the model

        Returns:
            True if model exists with valid manifest, False otherwise
        """
        if not self._grid_storage_available or not self._storage:
            return False

        try:
            manifest_path = f"models/{model_id}/_manifest.json"
            await self._storage.read_async(manifest_path)
            return True

        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """
        List all available model IDs in grid storage.

        Returns:
            List of model IDs
        """
        if not self._grid_storage_available or not self._storage:
            logger.debug("Grid storage not available, returning empty model list")
            return []

        try:
            files = await self._storage.list_async("models/")
            
            # Extract unique model IDs from paths like "models/{model_id}/..."
            model_ids = set()
            for file_path in files:
                parts = file_path.split('/')
                if len(parts) >= 2 and parts[0] == "models":
                    model_ids.add(parts[1])
            
            result = sorted(model_ids)
            logger.info(f"Found {len(result)} models in grid storage")
            return result

        except Exception as e:
            logger.error(f"Error listing models: {e}", exc_info=True)
            return []

    def _ensure_cache_dir(self, file_path: str) -> None:
        """
        Ensure parent directory exists for the given file path.

        Args:
            file_path: Path to file whose parent directory should be created
        """
        parent_dir = Path(file_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured cache directory exists: {parent_dir}")
