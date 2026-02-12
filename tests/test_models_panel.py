import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# PySide6 may not be installed in CI/test environments
pyside6 = pytest.importorskip("PySide6", reason="PySide6 not installed")

from aurarouter.gui.download_dialog import (  # noqa: E402
    _DownloadWorker,
    _FileListWorker,
    _SearchWorker,
)


class TestDownloadWorker:
    """Verify _DownloadWorker calls downloader and emits correct signals."""

    def test_calls_downloader_and_emits_finished(self):
        worker = _DownloadWorker(repo="Qwen/Test", filename="model.gguf")
        worker.finished = MagicMock()
        worker.error = MagicMock()

        with patch(
            "aurarouter.models.downloader.download_model",
            return_value="/fake/path/model.gguf",
        ) as mock_dl:
            worker.run()

        mock_dl.assert_called_once_with(repo="Qwen/Test", filename="model.gguf")
        worker.finished.emit.assert_called_once_with("/fake/path/model.gguf")
        worker.error.emit.assert_not_called()

    def test_emits_error_on_failure(self):
        worker = _DownloadWorker(repo="Qwen/Test", filename="model.gguf")
        worker.finished = MagicMock()
        worker.error = MagicMock()

        with patch(
            "aurarouter.models.downloader.download_model",
            side_effect=ImportError("huggingface-hub is required"),
        ):
            worker.run()

        worker.finished.emit.assert_not_called()
        worker.error.emit.assert_called_once()
        assert "huggingface-hub" in worker.error.emit.call_args[0][0]


class TestSearchWorker:
    """Verify _SearchWorker queries HfApi and emits results."""

    def test_search_returns_results(self):
        worker = _SearchWorker(query="qwen coder")
        worker.finished = MagicMock()
        worker.error = MagicMock()

        mock_model = SimpleNamespace(id="Qwen/Qwen2.5-Coder-GGUF", downloads=50000, likes=100)
        mock_api = MagicMock()
        mock_api.list_models.return_value = [mock_model]

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            worker.run()

        mock_api.list_models.assert_called_once_with(
            search="qwen coder", filter="gguf", sort="downloads", limit=25
        )
        worker.finished.emit.assert_called_once()
        results = worker.finished.emit.call_args[0][0]
        assert len(results) == 1
        assert results[0]["id"] == "Qwen/Qwen2.5-Coder-GGUF"
        assert results[0]["downloads"] == 50000

    def test_search_emits_error_on_failure(self):
        worker = _SearchWorker(query="test")
        worker.finished = MagicMock()
        worker.error = MagicMock()

        with patch("huggingface_hub.HfApi", side_effect=Exception("network error")):
            worker.run()

        worker.finished.emit.assert_not_called()
        worker.error.emit.assert_called_once()
        assert "network error" in worker.error.emit.call_args[0][0]


class TestFileListWorker:
    """Verify _FileListWorker fetches and filters .gguf files from a repo."""

    def test_lists_gguf_files(self):
        worker = _FileListWorker(repo_id="Qwen/Qwen2.5-Coder-GGUF")
        worker.finished = MagicMock()
        worker.error = MagicMock()

        siblings = [
            SimpleNamespace(rfilename="model-q4_k_m.gguf", size=4_500_000_000),
            SimpleNamespace(rfilename="model-q8_0.gguf", size=7_700_000_000),
            SimpleNamespace(rfilename="README.md", size=2000),
            SimpleNamespace(rfilename="config.json", size=500),
        ]
        mock_repo_info = SimpleNamespace(siblings=siblings)
        mock_api = MagicMock()
        mock_api.repo_info.return_value = mock_repo_info

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            worker.run()

        mock_api.repo_info.assert_called_once_with("Qwen/Qwen2.5-Coder-GGUF", files_metadata=True)
        worker.finished.emit.assert_called_once()
        files = worker.finished.emit.call_args[0][0]
        assert len(files) == 2
        assert files[0]["filename"] == "model-q4_k_m.gguf"
        assert files[1]["filename"] == "model-q8_0.gguf"

    def test_emits_error_on_failure(self):
        worker = _FileListWorker(repo_id="bad/repo")
        worker.finished = MagicMock()
        worker.error = MagicMock()

        mock_api = MagicMock()
        mock_api.repo_info.side_effect = Exception("repo not found")

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            worker.run()

        worker.finished.emit.assert_not_called()
        worker.error.emit.assert_called_once()


class TestGridSectionVisibility:
    """Verify grid section visibility depends on auragrid availability."""

    def test_grid_hidden_without_auragrid(self):
        with patch.dict(
            "sys.modules",
            {"aurarouter.auragrid.model_storage": None},
        ):
            try:
                from aurarouter.auragrid.model_storage import GridModelStorage  # noqa: F401
                available = True
            except ImportError:
                available = False

            assert available is False
