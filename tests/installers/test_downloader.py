from pathlib import Path
from unittest.mock import patch, MagicMock

from aurarouter.models.downloader import download_model


def test_download_to_custom_dest(tmp_path):
    """Verify download_model calls hf_hub_download and copies to dest."""
    cached_file = tmp_path / "hf_cache" / "model.gguf"
    cached_file.parent.mkdir()
    cached_file.write_bytes(b"fake-gguf-data")

    dest_dir = tmp_path / "my_models"

    with patch(
        "aurarouter.models.downloader.hf_hub_download",
        return_value=str(cached_file),
    ) as mock_hf:
        result = download_model(
            repo="Qwen/Test-GGUF",
            filename="model.gguf",
            dest=str(dest_dir),
        )

    mock_hf.assert_called_once_with(repo_id="Qwen/Test-GGUF", filename="model.gguf")
    assert result == dest_dir / "model.gguf"
    assert result.is_file()
    assert result.read_bytes() == b"fake-gguf-data"


def test_skips_if_already_exists(tmp_path):
    """If the file is already at dest, skip the download."""
    dest_dir = tmp_path / "models"
    dest_dir.mkdir()
    existing = dest_dir / "already.gguf"
    existing.write_bytes(b"existing-data")

    with patch("aurarouter.models.downloader.hf_hub_download") as mock_hf:
        result = download_model(
            repo="Qwen/Test-GGUF",
            filename="already.gguf",
            dest=str(dest_dir),
        )

    mock_hf.assert_not_called()
    assert result == existing


def test_default_dest_uses_auracore_dir(tmp_path):
    """When no dest is given, uses DEFAULT_MODEL_DIR."""
    cached_file = tmp_path / "cached.gguf"
    cached_file.write_bytes(b"data")

    default_dir = tmp_path / ".auracore" / "models"

    with (
        patch("aurarouter.models.downloader.hf_hub_download", return_value=str(cached_file)),
        patch("aurarouter.models.downloader.DEFAULT_MODEL_DIR", default_dir),
    ):
        result = download_model(repo="org/repo", filename="cached.gguf")

    assert result == default_dir / "cached.gguf"
    assert result.is_file()
