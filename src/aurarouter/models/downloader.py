import shutil
import asyncio
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Downloader")

DEFAULT_MODEL_DIR = Path.home() / ".auracore" / "models"


from auragrid.model_storage import GridModelStorage

logger = get_logger("AuraRouter.Downloader")

DEFAULT_MODEL_DIR = Path.home() / ".auracore" / "models"


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
        print(f"   Model already exists at: {final_path}")
        return final_path

    if grid_storage:
        print(f"   Checking grid storage for {filename}...")
        try:
            asyncio.run(grid_storage.download_model(repo, str(final_path)))
            if final_path.is_file():
                print(f"   Model downloaded from grid storage: {final_path}")
                return final_path
        except Exception as e:
            print(f"   Failed to download from grid storage: {e}")

    print(f"   Downloading {repo}/{filename} ...")
    cached = hf_hub_download(repo_id=repo, filename=filename)

    # Copy from HF cache to our models directory
    shutil.copy2(cached, final_path)
    print(f"   Saved to: {final_path}")

    if grid_storage:
        print(f"   Uploading {filename} to grid storage...")
        try:
            asyncio.run(grid_storage.upload_model(str(final_path), repo))
            print("   Upload to grid storage complete.")
        except Exception as e:
            print(f"   Failed to upload to grid storage: {e}")

    print("\n   Add this to your auraconfig.yaml:")
    print(f'   model_path: "{final_path}"')
    return final_path
