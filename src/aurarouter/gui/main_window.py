from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aurarouter.fabric import ComputeFabric
from aurarouter.gui.config_panel import ConfigPanel
from aurarouter.gui.document_input import DocumentInputWidget
from aurarouter.gui.environment import EnvironmentContext, HealthStatus, ServiceState
from aurarouter.gui.models_panel import ModelsPanel
from aurarouter.gui.routing_visualizer import RoutingVisualizer
from aurarouter.gui.service_controller import ServiceController
from aurarouter.gui.service_toolbar import ServiceToolbar
from aurarouter.routing import analyze_intent, generate_plan

_HISTORY_MAX = 20
_HISTORY_PATH = Path.home() / ".auracore" / "aurarouter" / "history.json"

# User-facing intent labels (internal values stay SIMPLE_CODE / COMPLEX_REASONING).
_INTENT_DISPLAY = {
    "SIMPLE_CODE": "Direct",
    "COMPLEX_REASONING": "Multi-Step",
}


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------

class InferenceWorker(QObject):
    """Runs the intent -> plan -> execute pipeline off the main thread."""

    intent_detected = Signal(str)
    plan_generated = Signal(list)
    step_started = Signal(int, str)
    step_completed = Signal(int, str)
    model_tried = Signal(str, str, bool, float)  # role, model_id, success, elapsed_s
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        fabric: ComputeFabric,
        task: str,
        file_context: str,
        language: str,
    ):
        super().__init__()
        self.fabric = fabric
        self.task = task
        self.file_context = file_context
        self.language = language

    def _emit_model_tried(
        self, role: str, model_id: str, success: bool, elapsed: float
    ) -> None:
        self.model_tried.emit(role, model_id, success, elapsed)

    def run(self) -> None:
        try:
            intent = analyze_intent(self.fabric, self.task)
            self.intent_detected.emit(intent)

            if intent == "SIMPLE_CODE":
                prompt = (
                    f"TASK: {self.task}\n"
                    f"LANG: {self.language}\n"
                    f"CONTEXT: {self.file_context}\n"
                    "RESPOND WITH OUTPUT ONLY."
                )
                result = self.fabric.execute(
                    "coding", prompt, on_model_tried=self._emit_model_tried,
                )
                self.finished.emit(result or "Error: Generation failed.")
                return

            plan = generate_plan(self.fabric, self.task, self.file_context)
            self.plan_generated.emit(plan)

            output: list[str] = []
            for i, step in enumerate(plan):
                self.step_started.emit(i, step)
                prompt = (
                    f"GOAL: {step}\n"
                    f"LANG: {self.language}\n"
                    f"CONTEXT: {self.file_context}\n"
                    f"PREVIOUS_OUTPUT: {output}\n"
                    "Return ONLY the requested output."
                )
                result = self.fabric.execute(
                    "coding", prompt, on_model_tried=self._emit_model_tried,
                )
                chunk = (
                    f"\n# --- Step {i + 1}: {step} ---\n{result}"
                    if result
                    else f"\n# Step {i + 1} Failed."
                )
                output.append(chunk)
                self.step_completed.emit(i, chunk)

            self.finished.emit("\n".join(output))

        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------

class AuraRouterWindow(QMainWindow):
    def __init__(self, context: EnvironmentContext):
        super().__init__()
        self._context = context
        self._fabric = ComputeFabric(context.get_config_loader())

        self._thread: Optional[QThread] = None
        self._worker: Optional[InferenceWorker] = None

        # Track dynamic (environment-specific) tab indices for cleanup.
        self._extra_tab_labels: list[str] = []

        self.setWindowTitle("AuraRouter")
        self.setMinimumSize(900, 700)

        # Check if AuraGrid SDK is importable for the toolbar.
        try:
            import aurarouter.auragrid  # noqa: F401
            auragrid_available = True
        except ImportError:
            auragrid_available = False

        self._history: list[dict] = self._load_history()

        self._service_controller = ServiceController(context)
        self._build_ui(auragrid_available)
        self._wire_signals()
        self._setup_shortcuts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, auragrid_available: bool) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ---- Service toolbar ----
        self._toolbar = ServiceToolbar(
            auragrid_available=auragrid_available, parent=self
        )
        root_layout.addWidget(self._toolbar)

        # ---- Tab widget ----
        self._tabs = QTabWidget()

        # Tab 1: Execute
        execute_tab = QWidget()
        self._build_execute_tab(execute_tab)
        self._tabs.addTab(execute_tab, "Execute")

        # Tab 2: Models
        self._models_panel = ModelsPanel(context=self._context)
        self._tabs.addTab(self._models_panel, "Models")

        # Tab 3: Configuration
        self._config_panel = ConfigPanel(context=self._context)
        self._config_panel.config_saved.connect(self._on_config_saved)
        self._tabs.addTab(self._config_panel, "Configuration")

        # Environment-specific extra tabs.
        self._add_extra_tabs()

        root_layout.addWidget(self._tabs)

        # ---- Status bar ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")

    def _build_execute_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        # ---- Input section ----
        input_group = QGroupBox("Task Input")
        inp = QVBoxLayout(input_group)

        # Recent tasks history.
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("Recent Tasks:"))
        self._history_combo = QComboBox()
        self._history_combo.addItem("(new)")
        for entry in self._history:
            self._history_combo.addItem(entry.get("task", "")[:80])
        self._history_combo.currentIndexChanged.connect(self._on_history_selected)
        history_row.addWidget(self._history_combo, 1)
        inp.addLayout(history_row)

        inp.addWidget(QLabel("Task Description:"))
        self.task_input = QTextEdit()
        self.task_input.setPlaceholderText(
            "Describe what you need...\n"
            "e.g. 'Summarize this document', 'Generate a REST API', "
            "'Analyze the attached data'"
        )
        self.task_input.setMaximumHeight(120)
        inp.addWidget(self.task_input)

        row = QHBoxLayout()

        ctx_col = QVBoxLayout()
        ctx_col.addWidget(QLabel("Context (optional):"))
        self.context_input = DocumentInputWidget()
        ctx_col.addWidget(self.context_input)
        row.addLayout(ctx_col)

        lang_col = QVBoxLayout()
        lang_col.addWidget(QLabel("Output Format:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(
            [
                "text",
                "markdown",
                "python",
                "csharp",
                "javascript",
                "typescript",
                "rust",
                "go",
                "java",
                "cpp",
                "bash",
            ]
        )
        lang_col.addWidget(self.language_combo)
        row.addLayout(lang_col)

        inp.addLayout(row)

        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setFixedHeight(36)
        self.execute_btn.clicked.connect(self._on_execute)
        inp.addWidget(self.execute_btn)

        layout.addWidget(input_group)

        # ---- Routing pipeline section ----
        route_group = QGroupBox("Routing Pipeline")
        route = QVBoxLayout(route_group)

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("Intent:"))
        self.intent_label = QLabel("--")
        self.intent_label.setStyleSheet("font-weight: bold;")
        info_row.addWidget(self.intent_label)
        info_row.addStretch()
        info_row.addWidget(QLabel("Plan Steps:"))
        self.plan_label = QLabel("--")
        self.plan_label.setStyleSheet("font-weight: bold;")
        info_row.addWidget(self.plan_label)
        info_row.addStretch()
        route.addLayout(info_row)

        self.plan_display = QTextEdit()
        self.plan_display.setReadOnly(True)
        self.plan_display.setMaximumHeight(80)
        self.plan_display.setPlaceholderText("Plan steps will appear here...")
        route.addWidget(self.plan_display)

        # Routing visualizer (pipeline stage boxes).
        self._routing_visualizer = RoutingVisualizer()
        route.addWidget(self._routing_visualizer)

        layout.addWidget(route_group)

        # ---- Output section ----
        out_group = QGroupBox("Output")
        out = QVBoxLayout(out_group)

        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setFont(QFont("Consolas", 10))
        out.addWidget(self.output_display)

        layout.addWidget(out_group)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        # Toolbar → controller.
        self._toolbar.start_clicked.connect(self._service_controller.start_service)
        self._toolbar.stop_clicked.connect(self._service_controller.stop_service)
        self._toolbar.pause_clicked.connect(self._service_controller.pause_service)
        self._toolbar.resume_clicked.connect(self._service_controller.resume_service)
        self._toolbar.health_clicked.connect(self._service_controller.run_health_check)

        # Controller → toolbar.
        self._service_controller.state_changed.connect(self._toolbar.set_state)
        self._service_controller.state_changed.connect(self._on_state_changed)
        self._service_controller.health_result.connect(self._on_health_result)
        self._service_controller.error.connect(self._on_service_error)

        # Environment switching.
        self._toolbar.environment_changed.connect(self._on_environment_changed)

    # ------------------------------------------------------------------
    # Environment switching
    # ------------------------------------------------------------------

    def _on_environment_changed(self, env_name: str) -> None:
        """Switch between Local and AuraGrid contexts at runtime."""
        if env_name == self._context.name:
            return

        # Confirm if service is active.
        current = self._context.get_state()
        if current not in (ServiceState.STOPPED, ServiceState.ERROR):
            reply = QMessageBox.question(
                self,
                "Switch Environment",
                "Switching environments will stop the running service. Continue?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._toolbar.set_environment(self._context.name)
                return

        # Dispose old context.
        self._context.dispose()

        # Remove extra tabs from previous environment.
        self._remove_extra_tabs()

        # Create new context.
        config_path = None
        if self._context.get_config_loader().config_path:
            config_path = str(self._context.get_config_loader().config_path)

        if env_name == "AuraGrid":
            from aurarouter.gui.env_grid import AuraGridEnvironmentContext

            self._context = AuraGridEnvironmentContext(config_path=config_path)
        else:
            from aurarouter.gui.env_local import LocalEnvironmentContext

            self._context = LocalEnvironmentContext(config_path=config_path)

        # Rewire.
        self._service_controller.set_context(self._context)
        self._fabric = ComputeFabric(self._context.get_config_loader())

        # Rebuild environment-aware panels.
        self._models_panel.set_context(self._context)
        self._config_panel.set_context(self._context)

        # Add new extra tabs.
        self._add_extra_tabs()

        self.status_bar.showMessage(f"Switched to {env_name} environment.")

    def _add_extra_tabs(self) -> None:
        for label, widget in self._context.get_extra_tabs():
            self._tabs.addTab(widget, label)
            self._extra_tab_labels.append(label)

    def _remove_extra_tabs(self) -> None:
        for label in self._extra_tab_labels:
            for i in range(self._tabs.count() - 1, -1, -1):
                if self._tabs.tabText(i) == label:
                    widget = self._tabs.widget(i)
                    self._tabs.removeTab(i)
                    widget.deleteLater()
        self._extra_tab_labels.clear()

    # ------------------------------------------------------------------
    # Config saved callback
    # ------------------------------------------------------------------

    def _on_config_saved(self) -> None:
        """Refresh the ComputeFabric when config is saved from the config panel."""
        self._fabric.update_config(self._context.get_config_loader())
        self.status_bar.showMessage("Configuration saved and applied.")

    # ------------------------------------------------------------------
    # Service state / health
    # ------------------------------------------------------------------

    def _on_state_changed(self, state_value: str) -> None:
        self.status_bar.showMessage(f"Service: {state_value}")

    def _on_health_result(self, status: HealthStatus) -> None:
        self._toolbar.set_health(status)
        self.status_bar.showMessage(
            f"Health: {'OK' if status.healthy else status.message}"
        )

    def _on_service_error(self, message: str) -> None:
        self.status_bar.showMessage(f"Service error: {message}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _on_execute(self) -> None:
        task = self.task_input.toPlainText().strip()
        if not task:
            self.status_bar.showMessage("Error: Task description is empty.")
            return

        self.output_display.clear()
        self.plan_display.clear()
        self.intent_label.setText("Analyzing...")
        self.plan_label.setText("--")
        self._routing_visualizer.reset()
        self.execute_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage("Running inference...")

        self._worker = InferenceWorker(
            fabric=self._fabric,
            task=task,
            file_context=self.context_input.get_context(),
            language=self.language_combo.currentText(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.intent_detected.connect(self._on_intent)
        self._worker.plan_generated.connect(self._on_plan)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_completed.connect(self._on_step_completed)
        self._worker.model_tried.connect(self._routing_visualizer.on_model_tried)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_intent(self, intent: str) -> None:
        display = _INTENT_DISPLAY.get(intent, intent)
        self.intent_label.setText(display)
        self._routing_visualizer.on_intent_detected(intent)
        if intent == "SIMPLE_CODE":
            self.status_bar.showMessage("Direct task — generating response...")
        else:
            self.status_bar.showMessage("Multi-step task — generating plan...")

    def _on_plan(self, plan: list) -> None:
        self.plan_label.setText(str(len(plan)))
        self.plan_display.setPlainText(
            "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(plan))
        )
        self.progress_bar.setRange(0, len(plan))
        self.progress_bar.setValue(0)

    def _on_step_started(self, index: int, description: str) -> None:
        self.status_bar.showMessage(f"Step {index + 1}: {description}")

    def _on_step_completed(self, index: int, result: str) -> None:
        self.output_display.append(result)
        self.progress_bar.setValue(index + 1)

    def _on_finished(self, result: str) -> None:
        if not self.output_display.toPlainText():
            self.output_display.setPlainText(result)
        self.execute_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Done.")
        # Save to history.
        task = self.task_input.toPlainText().strip()
        if task:
            self._add_to_history(task, self.output_display.toPlainText())

    def _on_error(self, message: str) -> None:
        self.output_display.setPlainText(f"ERROR: {message}")
        self.execute_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Error: {message}")

    def _cleanup_thread(self) -> None:
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+Return"), self, self._on_execute)
        QShortcut(QKeySequence("Ctrl+N"), self, self._on_new_prompt)
        QShortcut(QKeySequence("Escape"), self, self._on_cancel)

    def _on_new_prompt(self) -> None:
        """Clear all inputs for a fresh prompt (Ctrl+N)."""
        self.task_input.clear()
        self.context_input.clear()
        self.output_display.clear()
        self.plan_display.clear()
        self.intent_label.setText("--")
        self.plan_label.setText("--")
        self._routing_visualizer.reset()
        self._history_combo.setCurrentIndex(0)
        self.status_bar.showMessage("Ready")

    def _on_cancel(self) -> None:
        """Cancel a running execution (Escape)."""
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
            self._cleanup_thread()
            self.execute_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Execution cancelled.")

    # ------------------------------------------------------------------
    # Prompt history
    # ------------------------------------------------------------------

    @staticmethod
    def _load_history() -> list[dict]:
        try:
            if _HISTORY_PATH.is_file():
                return json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []

    def _save_history(self) -> None:
        try:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            _HISTORY_PATH.write_text(
                json.dumps(self._history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _add_to_history(self, task: str, result: str) -> None:
        entry = {"task": task, "result": result[:4000]}
        # Remove duplicate if same task already exists.
        self._history = [h for h in self._history if h.get("task") != task]
        self._history.insert(0, entry)
        self._history = self._history[:_HISTORY_MAX]
        self._save_history()

        # Update combo box.
        self._history_combo.blockSignals(True)
        self._history_combo.clear()
        self._history_combo.addItem("(new)")
        for h in self._history:
            self._history_combo.addItem(h.get("task", "")[:80])
        self._history_combo.setCurrentIndex(0)
        self._history_combo.blockSignals(False)

    def _on_history_selected(self, index: int) -> None:
        if index <= 0:
            return
        entry = self._history[index - 1]
        self.task_input.setPlainText(entry.get("task", ""))
        result = entry.get("result", "")
        if result:
            self.output_display.setPlainText(result)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)
        self._context.dispose()
        event.accept()
