"""Redesigned main application window with sidebar navigation.

Sidebar-driven layout replacing the legacy tab-based window:  top bar, collapsible sidebar, stacked content
area, and a slim status bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from aurarouter.api import AuraRouterAPI


class InferenceWorker(QObject):
    """Background worker for running inference tasks with progress signals."""

    # Existing signals
    intent_detected = Signal(str)
    plan_generated = Signal(list)
    step_started = Signal(int, str)
    step_completed = Signal(int, str)
    model_tried = Signal(str, str, bool, float)
    finished = Signal(str)
    error = Signal(str)
    trace_node_added = Signal(dict)
    trace_node_updated = Signal(str, dict)

    # Review loop signals (TG-B3)
    review_started = Signal(int)  # iteration number
    review_completed = Signal(int, str)  # iteration, verdict
    correction_started = Signal(int, int)  # iteration, step_count
from aurarouter.gui.environment import EnvironmentContext, HealthStatus, ServiceState
from aurarouter.gui.help import HELP
from aurarouter.gui.help.help_panel import HelpPanel
from aurarouter.gui.models_panel import ModelsPanel
from aurarouter.gui.monitor_panel import MonitorPanel
from aurarouter.gui.routing_panel import RoutingPanel
from aurarouter.gui.service_controller import ServiceController
from aurarouter.gui.settings_panel import SettingsPanel
from aurarouter.gui.theme import DARK_PALETTE, SPACING, TYPOGRAPHY, get_palette
from aurarouter.gui.widgets.sidebar_nav import SidebarNav
from aurarouter.gui.widgets.status_badge import StatusBadge
from aurarouter.gui.workspace_panel import WorkspacePanel

# ServiceState → StatusBadge mode mapping
_STATE_TO_BADGE: dict[str, tuple[str, str]] = {
    ServiceState.STOPPED.value: ("stopped", "Stopped"),
    ServiceState.STARTING.value: ("loading", "Starting"),
    ServiceState.LOADING_MODEL.value: ("loading", "Loading Model"),
    ServiceState.RUNNING.value: ("running", "Running"),
    ServiceState.PAUSING.value: ("loading", "Pausing"),
    ServiceState.PAUSED.value: ("paused", "Paused"),
    ServiceState.STOPPING.value: ("loading", "Stopping"),
    ServiceState.ERROR.value: ("error", "Error"),
}

_ONBOARDING_FLAG = Path.home() / ".auracore" / "aurarouter" / "onboarding_complete"

# Section definitions — key, icon, label
_SECTIONS: list[tuple[str, str, str]] = [
    ("workspace", "\u25b6", "Workspace"),
    ("routing", "\u25c6", "Routing"),
    ("models", "\u25fc", "Models"),
    ("monitor", "\u25c9", "Monitor"),
    ("settings", "\u2699", "Settings"),
    ("help", "?", "Help"),
]

_GRID_SECTION: tuple[str, str, str] = ("grid", "\u229e", "Grid")


class AuraRouterWindow(QMainWindow):
    """Redesigned main application window with sidebar navigation."""

    # Emitted when the user triggers the workspace execute action (Ctrl+Return).
    workspace_execute_requested = Signal()
    # Emitted when the user triggers workspace new-task (Ctrl+N).
    workspace_new_requested = Signal()
    # Emitted when the user triggers workspace cancel (Escape).
    workspace_cancel_requested = Signal()

    def __init__(
        self,
        api: AuraRouterAPI,
        env_context: EnvironmentContext,
    ) -> None:
        super().__init__()
        self._api = api
        self._context = env_context

        self._panel_factories: dict[str, Callable[[], QWidget]] = {}
        self._panel_cache: dict[str, QWidget] = {}
        self._panel_placeholders: dict[str, QWidget] = {}

        # Check AuraGrid availability.
        try:
            import aurarouter.auragrid  # noqa: F401
            self._auragrid_available = True
        except ImportError:
            self._auragrid_available = False

        self._service_controller = ServiceController(env_context)

        self.setWindowTitle("AuraRouter")
        self.setMinimumSize(960, 700)

        self._build_ui()
        self._wire_signals()
        self._setup_shortcuts()
        self._register_default_panels()

        # Update initial state.
        self._on_state_changed(env_context.get_state().value)

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Top bar (48px) ----
        self._top_bar = self._build_top_bar()
        root.addWidget(self._top_bar)

        # ---- Body: sidebar + content ----
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Sidebar
        self._sidebar = SidebarNav(palette=get_palette("dark"))
        for key, icon, label in _SECTIONS:
            self._sidebar.add_item(key, icon, label)
        if self._auragrid_available or self._context.name == "AuraGrid":
            self._sidebar.add_item(*_GRID_SECTION)
        body.addWidget(self._sidebar)

        # Content stack
        self._content_stack = QStackedWidget()
        self._content_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        body.addWidget(self._content_stack)

        root.addLayout(body, 1)

        # ---- Status bar (24px) ----
        self._status_bar = QStatusBar()
        self._status_bar.setFixedHeight(24)
        self.setStatusBar(self._status_bar)

        self._status_message = QLabel("Ready")
        self._status_bar.addWidget(self._status_message, 1)

        self._model_count_label = QLabel("")
        self._status_bar.addPermanentWidget(self._model_count_label)

        self._help_hint = QLabel("F1 for help")
        self._help_hint.setStyleSheet(f"color: {DARK_PALETTE.text_disabled};")
        self._status_bar.addPermanentWidget(self._help_hint)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet(
            f"background-color: {DARK_PALETTE.bg_secondary};"
            f"border-bottom: 1px solid {DARK_PALETTE.separator};"
        )

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACING.md, 0, SPACING.md, 0)
        layout.setSpacing(SPACING.md)

        # Title
        title = QLabel("AuraRouter")
        title.setStyleSheet(
            f"font-size: {TYPOGRAPHY.size_h2}px; font-weight: bold;"
            f"color: {DARK_PALETTE.accent};"
        )
        layout.addWidget(title)

        layout.addStretch()

        # Loading progress bar (small, hidden by default)
        self._loading_bar = QProgressBar()
        self._loading_bar.setMaximumWidth(120)
        self._loading_bar.setMaximumHeight(14)
        self._loading_bar.setRange(0, 0)  # indeterminate
        self._loading_bar.setVisible(False)
        layout.addWidget(self._loading_bar)

        # Environment selector
        layout.addWidget(QLabel("Env:"))
        self._env_combo = QComboBox()
        self._env_combo.addItem("Local")
        if self._auragrid_available:
            self._env_combo.addItem("AuraGrid")
        self._env_combo.setMinimumWidth(100)
        # Set to current context.
        idx = self._env_combo.findText(self._context.name)
        if idx >= 0:
            self._env_combo.setCurrentIndex(idx)
        layout.addWidget(self._env_combo)

        # StatusBadge
        self._status_badge = StatusBadge(mode="stopped", palette=get_palette("dark"))
        layout.addWidget(self._status_badge)

        # Play / Pause / Stop icon buttons
        self._play_btn = QPushButton("\u25b6")
        self._play_btn.setToolTip("Start service")
        self._play_btn.setFixedSize(32, 32)
        layout.addWidget(self._play_btn)

        self._pause_btn = QPushButton("\u23f8")
        self._pause_btn.setToolTip("Pause / Resume service")
        self._pause_btn.setFixedSize(32, 32)
        layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("\u25a0")
        self._stop_btn.setToolTip("Stop service")
        self._stop_btn.setFixedSize(32, 32)
        layout.addWidget(self._stop_btn)

        return bar

    # ==================================================================
    # Panel registration and lazy loading
    # ==================================================================

    def register_panel(self, section: str, factory: Callable[[], QWidget]) -> None:
        """Register a panel factory for a sidebar section.

        The factory is called lazily the first time the user navigates to
        the section.  If a placeholder already exists it is replaced.

        Parameters
        ----------
        section:
            One of the section keys (``"workspace"``, ``"routing"``, etc.).
        factory:
            Callable that returns a ``QWidget`` to display in the content area.
        """
        self._panel_factories[section] = factory

        # If the panel was already materialised, discard the cached version
        # so the factory runs again on next navigation.
        if section in self._panel_cache:
            old = self._panel_cache.pop(section)
            idx = self._content_stack.indexOf(old)
            if idx >= 0:
                self._content_stack.removeWidget(old)
                old.deleteLater()

    def _register_default_panels(self) -> None:
        """Register real panel factories for every section."""
        # Core panels — lazily instantiated on first navigation.
        self.register_panel(
            "workspace",
            lambda: WorkspacePanel(self._api, help_registry=HELP),
        )
        self.register_panel(
            "routing",
            lambda: RoutingPanel(self._api, help_registry=HELP),
        )
        self.register_panel(
            "models",
            lambda: ModelsPanel(self._api, help_registry=HELP),
        )
        self.register_panel(
            "monitor",
            lambda: MonitorPanel(self._api, help_registry=HELP),
        )
        self.register_panel(
            "settings",
            lambda: SettingsPanel(self._api, help_registry=HELP),
        )
        self.register_panel(
            "help",
            lambda: HelpPanel(),
        )

        # AuraGrid-specific panels (only when SDK is available).
        if self._auragrid_available or self._context.name == "AuraGrid":
            from aurarouter.gui.grid_deployment_panel import GridDeploymentPanel
            from aurarouter.gui.grid_status_panel import GridStatusPanel

            self.register_panel(
                "grid",
                lambda: self._build_grid_panel(GridStatusPanel, GridDeploymentPanel),
            )

        # Create initial placeholders in the stack so that the first
        # sidebar click triggers lazy creation via _get_or_create_panel.
        all_sections = list(_SECTIONS)
        if self._auragrid_available or self._context.name == "AuraGrid":
            all_sections.append(_GRID_SECTION)

        for key, _icon, label in all_sections:
            placeholder = QLabel(f"{label}\n\nLoading\u2026")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                f"font-size: {TYPOGRAPHY.size_h2}px;"
                f"color: {DARK_PALETTE.text_disabled};"
            )
            self._panel_placeholders[key] = placeholder
            self._content_stack.addWidget(placeholder)

    def _build_grid_panel(self, StatusCls, DeploymentCls) -> QWidget:
        """Build a combined Grid panel with status and deployment sub-panels."""
        from PySide6.QtWidgets import QSplitter, QVBoxLayout

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(StatusCls(context=self._context))
        splitter.addWidget(DeploymentCls(context=self._context))
        layout.addWidget(splitter)
        return container

    def _get_or_create_panel(self, section: str) -> QWidget:
        """Return the panel for *section*, creating it lazily if needed."""
        if section in self._panel_cache:
            return self._panel_cache[section]

        if section in self._panel_factories:
            panel = self._panel_factories[section]()
            self._panel_cache[section] = panel

            # Replace the placeholder in the stack.
            placeholder = self._panel_placeholders.get(section)
            if placeholder is not None:
                idx = self._content_stack.indexOf(placeholder)
                self._content_stack.removeWidget(placeholder)
                placeholder.deleteLater()
                del self._panel_placeholders[section]
            self._content_stack.addWidget(panel)
            return panel

        # Fall back to placeholder.
        return self._panel_placeholders.get(
            section,
            self._content_stack.currentWidget(),
        )

    # ==================================================================
    # Signal wiring
    # ==================================================================

    def _wire_signals(self) -> None:
        # Sidebar → content switching.
        self._sidebar.current_changed.connect(self._on_section_changed)

        # Top-bar buttons → service controller.
        self._play_btn.clicked.connect(self._on_play_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._stop_btn.clicked.connect(self._service_controller.stop_service)

        # Service controller → UI updates.
        self._service_controller.state_changed.connect(self._on_state_changed)
        self._service_controller.health_result.connect(self._on_health_result)
        self._service_controller.error.connect(self._on_service_error)

        # Environment selector.
        self._env_combo.currentTextChanged.connect(self._on_environment_changed)

    # ==================================================================
    # Section switching
    # ==================================================================

    def _on_section_changed(self, key: str) -> None:
        panel = self._get_or_create_panel(key)
        self._content_stack.setCurrentWidget(panel)

    # ==================================================================
    # Service control (Play / Pause / Stop)
    # ==================================================================

    def _on_play_clicked(self) -> None:
        self._service_controller.start_service()

    def _on_pause_clicked(self) -> None:
        state = self._service_controller.current_state()
        if state == ServiceState.PAUSED:
            self._service_controller.resume_service()
        else:
            self._service_controller.pause_service()

    def _on_state_changed(self, state_value: str) -> None:
        badge_mode, badge_text = _STATE_TO_BADGE.get(
            state_value, ("stopped", state_value),
        )
        self._status_badge.set_mode(badge_mode, badge_text)
        self._status_message.setText(f"Service: {badge_text}")

        # Show loading bar during transitional states.
        try:
            state = ServiceState(state_value)
        except ValueError:
            state = ServiceState.STOPPED

        self._loading_bar.setVisible(
            state in (ServiceState.STARTING, ServiceState.LOADING_MODEL),
        )

        # Button enable/disable matrix.
        is_stopped = state in (ServiceState.STOPPED, ServiceState.ERROR)
        is_running = state == ServiceState.RUNNING
        is_paused = state == ServiceState.PAUSED

        self._play_btn.setEnabled(is_stopped)
        self._pause_btn.setEnabled(is_running or is_paused)
        self._stop_btn.setEnabled(is_running or is_paused)

        # Toggle pause icon/tooltip.
        if is_paused:
            self._pause_btn.setText("\u25b6")
            self._pause_btn.setToolTip("Resume service")
        else:
            self._pause_btn.setText("\u23f8")
            self._pause_btn.setToolTip("Pause service")

        # Disable env selector while service is active.
        self._env_combo.setEnabled(is_stopped)

    def _on_health_result(self, status: HealthStatus) -> None:
        if status.healthy:
            self._status_badge.set_mode("healthy", "Healthy")
        else:
            self._status_badge.set_mode("unhealthy", status.message[:30])
        self._status_message.setText(
            f"Health: {'OK' if status.healthy else status.message}"
        )

    def _on_service_error(self, message: str) -> None:
        self._status_message.setText(f"Service error: {message}")

    # ==================================================================
    # Environment switching
    # ==================================================================

    def _on_environment_changed(self, env_name: str) -> None:
        """Switch between Local and AuraGrid contexts at runtime."""
        if env_name == self._context.name:
            return

        # Confirm if service is active.
        current_state = self._context.get_state()
        if current_state not in (ServiceState.STOPPED, ServiceState.ERROR):
            reply = QMessageBox.question(
                self,
                "Switch Environment",
                "Switching environments will stop the running service. Continue?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                # Revert selector without re-triggering.
                self._env_combo.blockSignals(True)
                idx = self._env_combo.findText(self._context.name)
                if idx >= 0:
                    self._env_combo.setCurrentIndex(idx)
                self._env_combo.blockSignals(False)
                return

        # Dispose old context.
        self._context.dispose()

        # Determine config path.
        config_path: Optional[str] = None
        try:
            cp = self._context.get_config_loader().config_path
            if cp:
                config_path = str(cp)
        except Exception:
            pass

        # Create new context.
        if env_name == "AuraGrid":
            from aurarouter.gui.env_grid import AuraGridEnvironmentContext
            self._context = AuraGridEnvironmentContext(config_path=config_path)
        else:
            from aurarouter.gui.env_local import LocalEnvironmentContext
            self._context = LocalEnvironmentContext(config_path=config_path)

        # Rewire controller.
        self._service_controller.set_context(self._context)

        self._status_message.setText(f"Switched to {env_name} environment.")

    # ==================================================================
    # Keyboard shortcuts
    # ==================================================================

    def _setup_shortcuts(self) -> None:
        # Workspace actions (forwarded as signals).
        QShortcut(QKeySequence("Ctrl+Return"), self, self._shortcut_execute)
        QShortcut(QKeySequence("Ctrl+N"), self, self._shortcut_new_task)
        QShortcut(QKeySequence("Escape"), self, self._shortcut_cancel)

        # Navigation shortcuts.
        QShortcut(QKeySequence("Ctrl+,"), self, lambda: self._go_to_section("settings"))
        QShortcut(QKeySequence("F1"), self, lambda: self._go_to_section("help"))

        # Ctrl+1 through Ctrl+6 for sections.
        all_sections = list(_SECTIONS)
        for i, (key, _icon, _label) in enumerate(all_sections[:6]):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            # Capture key by value in default arg.
            shortcut.activated.connect(
                lambda k=key: self._go_to_section(k)
            )

    def _go_to_section(self, key: str) -> None:
        self._sidebar.set_current(key)
        self._on_section_changed(key)

    def _shortcut_execute(self) -> None:
        self.workspace_execute_requested.emit()
        # Forward to workspace panel if it has been instantiated.
        wp = self._panel_cache.get("workspace")
        if wp is not None and hasattr(wp, "execute_requested"):
            wp.execute_requested.emit()

    def _shortcut_new_task(self) -> None:
        self.workspace_new_requested.emit()
        wp = self._panel_cache.get("workspace")
        if wp is not None and hasattr(wp, "new_requested"):
            wp.new_requested.emit()

    def _shortcut_cancel(self) -> None:
        self.workspace_cancel_requested.emit()
        wp = self._panel_cache.get("workspace")
        if wp is not None and hasattr(wp, "cancel_requested"):
            wp.cancel_requested.emit()

    # ==================================================================
    # Onboarding
    # ==================================================================

    def trigger_onboarding_if_needed(self) -> None:
        """Show the onboarding wizard if not yet completed.

        Should be called after the window is shown (e.g. via QTimer.singleShot).
        """
        from aurarouter.gui.help.onboarding import (
            OnboardingWizard,
            needs_onboarding,
        )

        if not needs_onboarding():
            return

        wizard = OnboardingWizard(parent=self)
        wizard.exec()

    def restart_onboarding(self) -> None:
        """Delete the onboarding flag and re-show the wizard."""
        if _ONBOARDING_FLAG.exists():
            _ONBOARDING_FLAG.unlink()

        from aurarouter.gui.help.onboarding import OnboardingWizard

        wizard = OnboardingWizard(parent=self)
        wizard.exec()

    # ==================================================================
    # Model count convenience
    # ==================================================================

    def update_model_count(self, count: int) -> None:
        """Update the model count display in the status bar."""
        self._model_count_label.setText(f"Models: {count}")

    # ==================================================================
    # Accessors
    # ==================================================================

    @property
    def api(self) -> AuraRouterAPI:
        return self._api

    @property
    def env_context(self) -> EnvironmentContext:
        return self._context

    @property
    def service_controller(self) -> ServiceController:
        return self._service_controller

    # ==================================================================
    # Close
    # ==================================================================

    def closeEvent(self, event) -> None:
        self._context.dispose()
        event.accept()
