"""Toolbar widget with environment selector, service controls, and health indicator."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.environment import HealthStatus, ServiceState

# Mapping ServiceState → (colour, display text)
_STATE_DISPLAY: dict[str, tuple[str, str]] = {
    ServiceState.STOPPED.value: ("#d32f2f", "Stopped"),
    ServiceState.STARTING.value: ("#9e9e9e", "Starting..."),
    ServiceState.LOADING_MODEL.value: ("#1565c0", "Loading model..."),
    ServiceState.RUNNING.value: ("#388e3c", "Running"),
    ServiceState.PAUSING.value: ("#9e9e9e", "Pausing..."),
    ServiceState.PAUSED.value: ("#f9a825", "Paused"),
    ServiceState.STOPPING.value: ("#9e9e9e", "Stopping..."),
    ServiceState.ERROR.value: ("#d32f2f", "Error"),
}


class ServiceToolbar(QWidget):
    """Horizontal toolbar: environment selector + Start/Pause/Stop + health."""

    environment_changed = Signal(str)   # "Local" or "AuraGrid"
    start_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    resume_clicked = Signal()
    health_clicked = Signal()

    def __init__(self, auragrid_available: bool = False, parent=None):
        super().__init__(parent)
        self._current_state = ServiceState.STOPPED

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)

        # ---- Environment selector ----
        layout.addWidget(QLabel("Environment:"))
        self._env_combo = QComboBox()
        self._env_combo.addItem("Local")
        if auragrid_available:
            self._env_combo.addItem("AuraGrid")
        self._env_combo.currentTextChanged.connect(self._on_env_changed)
        self._env_combo.setMinimumWidth(110)
        layout.addWidget(self._env_combo)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep1)

        # ---- State indicator ----
        self._state_label = QLabel()
        self._state_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._state_label)

        # ---- Loading progress (visible only during model loading) ----
        self._loading_progress = QProgressBar()
        self._loading_progress.setMaximumWidth(120)
        self._loading_progress.setMaximumHeight(16)
        self._loading_progress.setRange(0, 0)  # indeterminate
        self._loading_progress.setVisible(False)
        layout.addWidget(self._loading_progress)

        # ---- Control buttons ----
        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedWidth(70)
        self._start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedWidth(70)
        self._pause_btn.clicked.connect(self._on_pause_or_resume)
        layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.clicked.connect(self.stop_clicked)
        layout.addWidget(self._stop_btn)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep2)

        # ---- Health indicator (clickable for dashboard popup) ----
        self._last_health: HealthStatus | None = None
        self._health_label = QLabel("Health: --")
        self._health_label.setStyleSheet("color: gray;")
        self._health_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._health_label.mousePressEvent = lambda _: self._show_health_dashboard()
        layout.addWidget(self._health_label)

        health_btn = QPushButton("Check")
        health_btn.setFixedWidth(56)
        health_btn.clicked.connect(self.health_clicked)
        layout.addWidget(health_btn)

        layout.addStretch()

        # Apply initial button states.
        self._apply_state(ServiceState.STOPPED)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_state(self, state_value: str) -> None:
        """Update the toolbar to reflect the given ``ServiceState`` value."""
        try:
            state = ServiceState(state_value)
        except ValueError:
            return
        self._apply_state(state)

    def set_health(self, status: HealthStatus) -> None:
        """Update the health indicator."""
        self._last_health = status
        if status.healthy:
            self._health_label.setText("Health: OK")
            self._health_label.setStyleSheet(
                "color: #388e3c; font-weight: bold; text-decoration: underline;"
            )
        else:
            self._health_label.setText(f"Health: {status.message[:40]}")
            self._health_label.setStyleSheet(
                "color: #d32f2f; font-weight: bold; text-decoration: underline;"
            )

    def set_environment(self, env_name: str) -> None:
        """Programmatically set the environment selector."""
        idx = self._env_combo.findText(env_name)
        if idx >= 0:
            self._env_combo.setCurrentIndex(idx)

    def current_environment(self) -> str:
        return self._env_combo.currentText()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_state(self, state: ServiceState) -> None:
        self._current_state = state
        colour, label = _STATE_DISPLAY.get(
            state.value, ("#9e9e9e", state.value)
        )
        self._state_label.setText(f"  {label}")
        self._state_label.setStyleSheet(
            f"color: {colour}; font-weight: bold;"
        )

        # Show indeterminate progress bar during model loading.
        self._loading_progress.setVisible(state == ServiceState.LOADING_MODEL)

        # Button enabled/disabled matrix.
        is_stopped = state in (ServiceState.STOPPED, ServiceState.ERROR)
        is_running = state == ServiceState.RUNNING
        is_paused = state == ServiceState.PAUSED

        self._start_btn.setEnabled(is_stopped)
        self._pause_btn.setEnabled(is_running or is_paused)
        self._stop_btn.setEnabled(is_running or is_paused)

        # Toggle Pause ↔ Resume label.
        if is_paused:
            self._pause_btn.setText("Resume")
        else:
            self._pause_btn.setText("Pause")

        # Disable env selector while service is active.
        self._env_combo.setEnabled(is_stopped)

    def _on_pause_or_resume(self) -> None:
        if self._current_state == ServiceState.PAUSED:
            self.resume_clicked.emit()
        else:
            self.pause_clicked.emit()

    def _on_env_changed(self, text: str) -> None:
        self.environment_changed.emit(text)

    def _show_health_dashboard(self) -> None:
        """Open a popup showing per-model health details."""
        dlg = _HealthDashboardDialog(self._last_health, parent=self)
        dlg.check_all_clicked.connect(self.health_clicked)
        dlg.exec()


class _HealthDashboardDialog(QDialog):
    """Popup dialog showing per-model health status."""

    check_all_clicked = Signal()

    def __init__(self, status: HealthStatus | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Health Dashboard")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        if status is None:
            layout.addWidget(QLabel("No health check has been run yet.\nClick 'Check All' below."))
        else:
            # Overall status.
            overall = QLabel("Overall: OK" if status.healthy else f"Overall: {status.message}")
            overall.setStyleSheet(
                f"font-weight: bold; color: {'#388e3c' if status.healthy else '#d32f2f'};"
            )
            layout.addWidget(overall)

            # Per-model details.
            if status.details:
                for model_id, ok in status.details.items():
                    dot = "\u2705" if ok else "\u274c"
                    row = QLabel(f"  {dot}  {model_id}")
                    row.setStyleSheet(f"color: {'#388e3c' if ok else '#d32f2f'};")
                    layout.addWidget(row)
            else:
                layout.addWidget(QLabel("  (no model details available)"))

        # Check All button.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        check_btn = QPushButton("Check All")
        check_btn.clicked.connect(self.check_all_clicked)
        check_btn.clicked.connect(self.accept)
        btn_row.addWidget(check_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
