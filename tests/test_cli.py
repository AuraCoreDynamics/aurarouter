import sys
from unittest.mock import patch, MagicMock

import pytest

from aurarouter.cli import main


def _run_cli(*args: str):
    """Invoke main() with the given CLI arguments."""
    with patch.object(sys, "argv", ["aurarouter", *args]):
        main()


class TestInstallDispatch:
    """Verify --install flags dispatch to the correct installers."""

    def test_install_all(self):
        with (
            patch("aurarouter.installers.template.create_config_template") as mock_tmpl,
            patch("aurarouter.installers.registry.install_all") as mock_all,
        ):
            _run_cli("--install")
        mock_tmpl.assert_called_once()
        mock_all.assert_called_once()

    def test_install_gemini(self):
        mock_inst = MagicMock()
        with (
            patch("aurarouter.installers.template.create_config_template"),
            patch("aurarouter.installers.gemini.GeminiInstaller", return_value=mock_inst),
        ):
            _run_cli("--install-gemini")
        mock_inst.install.assert_called_once()

    def test_install_claude(self):
        mock_inst = MagicMock()
        with (
            patch("aurarouter.installers.template.create_config_template"),
            patch("aurarouter.installers.claude_inst.ClaudeInstaller", return_value=mock_inst),
        ):
            _run_cli("--install-claude")
        mock_inst.install.assert_called_once()


class TestDownloadModel:
    """Verify the download-model subcommand dispatches correctly."""

    def test_dispatches_to_downloader(self):
        with patch("aurarouter.models.downloader.download_model") as mock_dl:
            _run_cli("download-model", "--repo", "Qwen/test", "--file", "model.gguf")
        mock_dl.assert_called_once_with(repo="Qwen/test", filename="model.gguf", dest=None)

    def test_with_custom_dest(self):
        with patch("aurarouter.models.downloader.download_model") as mock_dl:
            _run_cli("download-model", "--repo", "Qwen/test", "--file", "m.gguf", "--dest", "/tmp/models")
        mock_dl.assert_called_once_with(repo="Qwen/test", filename="m.gguf", dest="/tmp/models")


class TestMcpServer:
    """Verify default mode creates and runs the MCP server."""

    def test_default_runs_mcp(self):
        mock_mcp = MagicMock()
        with (
            patch("aurarouter.config.ConfigLoader") as MockCfg,
            patch("aurarouter.server.create_mcp_server", return_value=mock_mcp),
        ):
            _run_cli("--config", "auraconfig.yaml")
        MockCfg.assert_called_once_with(config_path="auraconfig.yaml")
        mock_mcp.run.assert_called_once()

    def test_mcp_server_friendly_error_without_config(self, capsys):
        """MCP server mode prints a helpful message and exits when no config found."""
        with (
            patch(
                "aurarouter.config.ConfigLoader",
                side_effect=FileNotFoundError("no config"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_cli()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "aurarouter --install" in captured.out
        assert "aurarouter gui" in captured.out


class TestGuiSubcommand:
    """Verify the gui subcommand tries to import and launch the GUI."""

    def test_gui_dispatches_to_launch_gui(self):
        mock_launch = MagicMock()
        # gui.app does a top-level check_pyside6() which fails if PySide6 is
        # absent, so we mock the entire import chain that cli.py uses.
        mock_gui_app = MagicMock()
        mock_gui_app.launch_gui = mock_launch

        with (
            patch("aurarouter.config.ConfigLoader"),
            patch.dict("sys.modules", {"aurarouter.gui.app": mock_gui_app}),
        ):
            # --config is a top-level flag, must come before the subcommand
            _run_cli("--config", "auraconfig.yaml", "gui")
        mock_launch.assert_called_once()

    def test_gui_opens_without_config(self):
        """GUI should launch with empty config when no config file exists."""
        mock_launch = MagicMock()
        mock_gui_app = MagicMock()
        mock_gui_app.launch_gui = mock_launch

        with (
            patch(
                "aurarouter.config.ConfigLoader",
                side_effect=[FileNotFoundError("no config"), MagicMock()],
            ) as MockCfg,
            patch.dict("sys.modules", {"aurarouter.gui.app": mock_gui_app}),
        ):
            _run_cli("gui")

        # First call raises FileNotFoundError, second call uses allow_missing=True
        assert MockCfg.call_count == 2
        assert MockCfg.call_args_list[1] == ((), {"allow_missing": True})
        mock_launch.assert_called_once()

    def test_gui_import_error_propagates(self):
        """If the gui module fails to import, the error should propagate."""
        with (
            patch.dict("sys.modules", {"aurarouter.gui.app": None}),
            pytest.raises(ImportError),
        ):
            _run_cli("gui")


class TestListModels:
    """Verify the list-models subcommand dispatches correctly."""

    def test_list_models_empty(self, capsys):
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = []
        mock_storage.models_dir = "/fake/models"

        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            _run_cli("list-models")

        captured = capsys.readouterr()
        assert "No models found" in captured.out

    def test_list_models_with_entries(self, capsys):
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = [
            {"filename": "model.gguf", "size_bytes": 1024 * 1024 * 100, "repo": "org/repo"},
        ]
        mock_storage.models_dir = "/fake/models"

        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            _run_cli("list-models")

        captured = capsys.readouterr()
        assert "model.gguf" in captured.out
        assert "100 MB" in captured.out
        assert "org/repo" in captured.out


class TestRemoveModel:
    """Verify the remove-model subcommand dispatches correctly."""

    def test_remove_model_success(self, capsys):
        mock_storage = MagicMock()
        mock_storage.remove.return_value = True

        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            _run_cli("remove-model", "--file", "model.gguf")

        mock_storage.remove.assert_called_once_with("model.gguf", delete_file=True)
        captured = capsys.readouterr()
        assert "Removed and deleted" in captured.out

    def test_remove_model_keep_file(self, capsys):
        mock_storage = MagicMock()
        mock_storage.remove.return_value = True

        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            _run_cli("remove-model", "--file", "model.gguf", "--keep-file")

        mock_storage.remove.assert_called_once_with("model.gguf", delete_file=False)
        captured = capsys.readouterr()
        assert "Unregistered" in captured.out

    def test_remove_model_not_found(self, capsys):
        mock_storage = MagicMock()
        mock_storage.remove.return_value = False

        with (
            patch(
                "aurarouter.models.file_storage.FileModelStorage",
                return_value=mock_storage,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_cli("remove-model", "--file", "missing.gguf")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out
