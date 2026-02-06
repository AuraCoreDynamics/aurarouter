import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Downloader")

DEFAULT_MODEL_DIR = Path.home() / ".auracore" / "models"


def download_model(
    repo: str,
    filename: str,
    dest: Optional[str] = None,
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

    print(f"   Downloading {repo}/{filename} ...")
    cached = hf_hub_download(repo_id=repo, filename=filename)

    # Copy from HF cache to our models directory
    shutil.copy2(cached, final_path)
    print(f"   Saved to: {final_path}")
    print(f"\n   Add this to your auraconfig.yaml:")
    print(f'   model_path: "{final_path}"')
    return final_path
