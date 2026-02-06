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

    def test_gui_missing_pyside6_exits(self):
        """When PySide6 is not installed, the gui subcommand should exit."""
        with (
            patch.dict("sys.modules", {"aurarouter.gui.app": None}),
            pytest.raises(SystemExit),
        ):
            _run_cli("gui")
