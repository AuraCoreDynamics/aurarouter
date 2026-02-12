import shutil
import asyncio
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Downloader")

DEFAULT_MODEL_DIR = Path.home() / ".auracore" / "models"

try:
    from aurarouter.auragrid.model_storage import GridModelStorage
except ImportError:
    GridModelStorage = None


def download_model(
    repo: str,
    filename: str,
    dest: Optional[str] = None,
    grid_storage: Optional[GridModelStorage] = None,
) -> Path:
    """Download a GGUF model from HuggingFace Hub.

    Parameters
    ----------
    repo:
        HuggingFace repository ID (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF").
    filename:
        Name of the GGUF file inside the repo.
    dest:
        Optional destination directory.  Defaults to ``~/.auracore/models/``.
    grid_storage:
        Optional GridModelStorage instance for grid-aware storage.

    Returns
    -------
    Path to the downloaded (or copied) GGUF file.
    """
    dest_dir = Path(dest) if dest else DEFAULT_MODEL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / filename

    if final_path.is_file():
        logger.info(f"Model already exists at: {final_path}")
        return final_path

    if grid_storage is not None and GridModelStorage is not None:
        logger.info(f"Checking grid storage for {filename}...")
        try:
            asyncio.run(grid_storage.download_model(repo, str(final_path)))
            if final_path.is_file():
                logger.info(f"Model downloaded from grid storage: {final_path}")
                return final_path
        except Exception as e:
            logger.warning(f"Failed to download from grid storage: {e}")

    if hf_hub_download is None:
        raise ImportError(
            "huggingface-hub is required for model downloading.\n"
            "Install with:  pip install aurarouter[local]"
        )

    logger.info(f"Downloading {repo}/{filename} ...")
    cached = hf_hub_download(repo_id=repo, filename=filename)

    # Copy from HF cache to our models directory
    shutil.copy2(cached, final_path)
    logger.info(f"Saved to: {final_path}")

    if grid_storage is not None and GridModelStorage is not None:
        logger.info(f"Uploading {filename} to grid storage...")
        try:
            asyncio.run(grid_storage.upload_model(str(final_path), repo))
            logger.info("Upload to grid storage complete.")
        except Exception as e:
            logger.warning(f"Failed to upload to grid storage: {e}")

    logger.info("\nAdd this to your auraconfig.yaml:")
    logger.info(f'model_path: "{final_path}"')
    return final_path
