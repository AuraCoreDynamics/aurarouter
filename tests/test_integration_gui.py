"""Integration tests for the redesigned GUI panels (v0.5.1).

These tests verify that all panels can be instantiated with a mock API
and that panel registration / keyboard shortcuts work correctly.
No display server is needed -- tests run headlessly via QApplication.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_qapp = None


def _get_qapp():
    """Return a shared QApplication (creating one if needed)."""
    global _qapp
    if _qapp is None:
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            pytest.skip("PySide6 not available")
        _qapp = QApplication.instance() or QApplication(sys.argv)
    return _qapp


@pytest.fixture(autouse=True)
def _ensure_qapp():
    """Ensure a QApplication exists for the test session."""
    _get_qapp()


@pytest.fixture()
def mock_api(config):
    """Return a lightweight AuraRouterAPI built from the test config fixture."""
    from aurarouter.api import APIConfig, AuraRouterAPI

    api = AuraRouterAPI(APIConfig(config_path=str(config.config_path)))
    return api


def _make_window(mock_api):
    """Create a main window with mocked environment context."""
    from aurarouter.gui.environment import EnvironmentContext
    from aurarouter.gui.main_window import AuraRouterWindow

    ctx = MagicMock(spec=EnvironmentContext)
    ctx.name = "Local"
    ctx.get_state.return_value = MagicMock(value="stopped")
    ctx.dispose = MagicMock()

    window = AuraRouterWindow(api=mock_api, env_context=ctx)
    return window


# ---------------------------------------------------------------------------
# Panel instantiation (lightweight panels only -- avoid panels that start
# background threads/timers in their constructor, which can hang in CI).
# ---------------------------------------------------------------------------


class TestPanelInstantiation:
    """Each panel can be constructed without raising."""

    def test_help_panel(self):
        from aurarouter.gui.help.help_panel import HelpPanel

        panel = HelpPanel()
        assert panel is not None

    def test_routing_panel(self, mock_api):
        from aurarouter.gui.help import HELP
        from aurarouter.gui.routing_panel import RoutingPanel

        panel = RoutingPanel(mock_api, help_registry=HELP)
        assert panel is not None

    def test_settings_panel(self, mock_api):
        from aurarouter.gui.help import HELP
        from aurarouter.gui.settings_panel import SettingsPanel

        panel = SettingsPanel(mock_api, help_registry=HELP)
        assert panel is not None

    def test_workspace_panel(self, mock_api):
        from aurarouter.gui.help import HELP
        from aurarouter.gui.workspace_panel import WorkspacePanel

        panel = WorkspacePanel(mock_api, help_registry=HELP)
        assert panel is not None

    def test_models_panel_import(self):
        """ModelsPanel class is importable (constructor starts bg threads)."""
        from aurarouter.gui.models_panel import ModelsPanel

        assert ModelsPanel is not None

    def test_monitor_panel_import(self):
        """MonitorPanel class is importable (constructor starts bg timers)."""
        from aurarouter.gui.monitor_panel import MonitorPanel

        assert MonitorPanel is not None


# ---------------------------------------------------------------------------
# Panel registration
# ---------------------------------------------------------------------------


class TestPanelRegistration:
    """Panel registration into the application shell works."""

    def test_factories_registered(self, mock_api):
        window = _make_window(mock_api)
        try:
            for section in ("workspace", "routing", "models", "monitor", "settings", "help"):
                assert section in window._panel_factories, f"Missing factory for {section}"
        finally:
            window.close()

    def test_lazy_panel_creation(self, mock_api):
        window = _make_window(mock_api)
        try:
            # Before navigation, no panels are cached.
            assert len(window._panel_cache) == 0
            # Navigate to help (lightweight) -- should create the panel.
            panel = window._get_or_create_panel("help")
            assert panel is not None
            assert "help" in window._panel_cache
        finally:
            window.close()


# ---------------------------------------------------------------------------
# Keyboard shortcuts
# ---------------------------------------------------------------------------


class TestKeyboardShortcuts:
    """Keyboard shortcuts are registered on the shell."""

    def test_shortcut_signals_exist(self, mock_api):
        window = _make_window(mock_api)
        try:
            assert hasattr(window, "workspace_execute_requested")
            assert hasattr(window, "workspace_new_requested")
            assert hasattr(window, "workspace_cancel_requested")
        finally:
            window.close()

    def test_shortcut_methods_callable(self, mock_api):
        window = _make_window(mock_api)
        try:
            # Calling the shortcut methods should not raise.
            window._shortcut_execute()
            window._shortcut_new_task()
            window._shortcut_cancel()
        finally:
            window.close()
