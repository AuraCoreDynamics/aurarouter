from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.gui.config_panel import ConfigPanel
from aurarouter.routing import analyze_intent, generate_plan


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------

class InferenceWorker(QObject):
    """Runs the intent -> plan -> execute pipeline off the main thread."""

    intent_detected = Signal(str)
    plan_generated = Signal(list)
    step_started = Signal(int, str)
    step_completed = Signal(int, str)
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

    def run(self) -> None:
        try:
            intent = analyze_intent(self.fabric, self.task)
            self.intent_detected.emit(intent)

            if intent == "SIMPLE_CODE":
                prompt = (
                    f"TASK: {self.task}\n"
                    f"LANG: {self.language}\n"
                    f"CONTEXT: {self.file_context}\n"
                    "CODE ONLY."
                )
                result = self.fabric.execute("coding", prompt)
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
                    f"PREVIOUS_CODE: {output}\n"
                    "Return ONLY valid code."
                )
                code = self.fabric.execute("coding", prompt)
                chunk = (
                    f"\n# --- Step {i + 1}: {step} ---\n{code}"
                    if code
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
    def __init__(self, config: ConfigLoader):
        super().__init__()
        self._config = config
        self._fabric = ComputeFabric(config)

        self._thread: Optional[QThread] = None
        self._worker: Optional[InferenceWorker] = None

        self.setWindowTitle("AuraRouter - Compute Fabric GUI")
        self.setMinimumSize(900, 700)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ---- Tab widget ----
        self._tabs = QTabWidget()

        # Tab 1: Execute (existing task execution UI)
        execute_tab = QWidget()
        self._build_execute_tab(execute_tab)
        self._tabs.addTab(execute_tab, "Execute")

        # Tab 2: Configuration
        self._config_panel = ConfigPanel(self._config)
        self._config_panel.config_saved.connect(self._on_config_saved)
        self._tabs.addTab(self._config_panel, "Configuration")

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

        inp.addWidget(QLabel("Task Description:"))
        self.task_input = QTextEdit()
        self.task_input.setPlaceholderText(
            "Describe what you want to generate...\n"
            "e.g. 'Write a Python function to calculate Fibonacci numbers'"
        )
        self.task_input.setMaximumHeight(120)
        inp.addWidget(self.task_input)

        row = QHBoxLayout()

        ctx_col = QVBoxLayout()
        ctx_col.addWidget(QLabel("File Context (optional):"))
        self.context_input = QLineEdit()
        self.context_input.setPlaceholderText("Paste relevant code context...")
        ctx_col.addWidget(self.context_input)
        row.addLayout(ctx_col)

        lang_col = QVBoxLayout()
        lang_col.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(
            ["python", "csharp", "javascript", "typescript", "rust", "go", "java", "cpp", "bash"]
        )
        lang_col.addWidget(self.language_combo)
        row.addLayout(lang_col)

        inp.addLayout(row)

        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setFixedHeight(36)
        self.execute_btn.clicked.connect(self._on_execute)
        inp.addWidget(self.execute_btn)

        layout.addWidget(input_group)

        # ---- Routing info section ----
        route_group = QGroupBox("Routing Information")
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
    # Config saved callback
    # ------------------------------------------------------------------

    def _on_config_saved(self) -> None:
        """Refresh the ComputeFabric when config is saved from the config panel."""
        self._fabric.update_config(self._config)
        self.status_bar.showMessage("Configuration saved and applied.")

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
        self.execute_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage("Running inference...")

        self._worker = InferenceWorker(
            fabric=self._fabric,
            task=task,
            file_context=self.context_input.text(),
            language=self.language_combo.currentText(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.intent_detected.connect(self._on_intent)
        self._worker.plan_generated.connect(self._on_plan)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_completed.connect(self._on_step_completed)
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
        self.intent_label.setText(intent)
        if intent == "SIMPLE_CODE":
            self.status_bar.showMessage("Simple task - generating code...")
        else:
            self.status_bar.showMessage("Complex task - generating plan...")

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

    def closeEvent(self, event) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)
        event.accept()
