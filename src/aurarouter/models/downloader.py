import shutil
import asyncio
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]

from aurarouter._logging import get_logger
from aurarouter.models.file_storage import FileModelStorage


def _make_progress_tqdm(callback):
    """Create a tqdm-compatible class that forwards progress to *callback*.

    The callback signature is ``callback(downloaded_bytes, total_bytes)``.
    """

    class _CallbackTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.n = 0

        def update(self, n=1):
            self.n += n
            callback(self.n, self.total)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return _CallbackTqdm

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
    progress_callback: Optional[callable] = None,
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
    storage = FileModelStorage(dest_dir)

    if final_path.is_file():
        logger.info(f"Model already exists at: {final_path}")
        # Ensure it's registered even if it was placed manually
        storage.register(repo=repo, filename=filename, path=final_path)
        return final_path

    if grid_storage is not None and GridModelStorage is not None:
        logger.info(f"Checking grid storage for {filename}...")
        try:
            asyncio.run(grid_storage.download_model(repo, str(final_path)))
            if final_path.is_file():
                logger.info(f"Model downloaded from grid storage: {final_path}")
                storage.register(repo=repo, filename=filename, path=final_path)
                return final_path
        except Exception as e:
            logger.warning(f"Failed to download from grid storage: {e}")

    if hf_hub_download is None:
        raise ImportError(
            "huggingface-hub is required for model downloading.\n"
            "Install with:  pip install aurarouter[local]"
        )

    logger.info(f"Downloading {repo}/{filename} ...")
    dl_kwargs = {"repo_id": repo, "filename": filename}
    if progress_callback is not None:
        dl_kwargs["tqdm_class"] = _make_progress_tqdm(progress_callback)
    cached = hf_hub_download(**dl_kwargs)

    # Copy from HF cache to our models directory
    shutil.copy2(cached, final_path)
    logger.info(f"Saved to: {final_path}")

    # Attempt auto-tune to extract metadata and recommend parameters
    gguf_metadata = None
    recommended_params = None
    try:
        from aurarouter.tuning import extract_gguf_metadata, recommend_llamacpp_params

        gguf_metadata = extract_gguf_metadata(final_path)
        recommended_params = recommend_llamacpp_params(final_path, gguf_metadata)
    except Exception as exc:
        logger.warning(f"Auto-tune on download failed: {exc}")
        gguf_metadata = None

    storage.register(
        repo=repo, filename=filename, path=final_path, metadata=gguf_metadata,
    )

    if grid_storage is not None and GridModelStorage is not None:
        logger.info(f"Uploading {filename} to grid storage...")
        try:
            asyncio.run(grid_storage.upload_model(str(final_path), repo))
            logger.info("Upload to grid storage complete.")
        except Exception as e:
            logger.warning(f"Failed to upload to grid storage: {e}")

    # Emit a ready-to-paste config snippet
    model_id_suggestion = final_path.stem.lower().replace(" ", "-")
    if recommended_params:
        params_lines = "\n".join(
            f"      {k}: {v}" for k, v in recommended_params.items()
        )
        logger.info(
            f"\nAdd this to your auraconfig.yaml under 'models':\n"
            f"  {model_id_suggestion}:\n"
            f"    provider: llamacpp\n"
            f'    model_path: "{final_path}"\n'
            f"    parameters:\n"
            f"{params_lines}"
        )
    else:
        logger.info(
            f"\nAdd this to your auraconfig.yaml under 'models':\n"
            f"  {model_id_suggestion}:\n"
            f"    provider: llamacpp\n"
            f'    model_path: "{final_path}"'
        )
    return final_path
