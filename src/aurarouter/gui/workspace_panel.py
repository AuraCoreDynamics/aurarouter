"""Workspace (Execute) panel — the hero screen for AuraRouter.

Three-column layout:
  Left   – Task History sidebar (search, list, clear)
  Center – Task input, DAG visualizer (always visible), output with syntax
           highlighting
  Right  – Context panel (file attachments, output format, execution settings,
           help tooltips)

Execution runs on a background QThread via :class:`WorkspaceWorker`, which
calls :meth:`AuraRouterAPI.execute_task` with progress callbacks wired
through Qt signals.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QRegularExpression, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aurarouter.api import AuraRouterAPI
from aurarouter.gui.dag_visualizer import DAGVisualizer
from aurarouter.gui.document_input import DocumentInputWidget
from aurarouter.gui.execution_trace import ExecutionTrace, NodeStatus, TraceNode
from aurarouter.gui.theme import DARK_PALETTE, SPACING, TYPOGRAPHY
from aurarouter.gui.widgets.collapsible_section import CollapsibleSection
from aurarouter.gui.widgets.help_tooltip import HelpTooltip
from aurarouter.intent_registry import IntentRegistry, build_intent_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_MAX = 50
_HISTORY_PATH = Path.home() / ".auracore" / "aurarouter" / "history.json"

_OUTPUT_FORMATS = [
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


# ---------------------------------------------------------------------------
# Python syntax highlighter
# ---------------------------------------------------------------------------

class _PythonHighlighter(QSyntaxHighlighter):
    """Minimal Python syntax highlighter for the output area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []
        p = DARK_PALETTE

        # Keywords
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor(p.accent))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "False", "None", "True", "and", "as", "assert", "async",
            "await", "break", "class", "continue", "def", "del", "elif",
            "else", "except", "finally", "for", "from", "global", "if",
            "import", "in", "is", "lambda", "nonlocal", "not", "or",
            "pass", "raise", "return", "try", "while", "with", "yield",
        ]
        pattern = r"\b(?:" + "|".join(keywords) + r")\b"
        self._rules.append((QRegularExpression(pattern), kw_fmt))

        # Strings (double-quoted)
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor(p.success))
        self._rules.append(
            (QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), str_fmt)
        )
        # Strings (single-quoted)
        self._rules.append(
            (QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt)
        )

        # Comments
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor(p.text_disabled))
        comment_fmt.setFontItalic(True)
        self._rules.append((QRegularExpression(r"#[^\n]*"), comment_fmt))

        # Numbers
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor(p.warning))
        self._rules.append(
            (QRegularExpression(r"\b\d+(\.\d+)?\b"), num_fmt)
        )

        # Decorators
        dec_fmt = QTextCharFormat()
        dec_fmt.setForeground(QColor(p.info))
        self._rules.append((QRegularExpression(r"@\w+"), dec_fmt))

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for regex, fmt in self._rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


# ---------------------------------------------------------------------------
# Idle template DAG
# ---------------------------------------------------------------------------

def _build_idle_trace() -> ExecutionTrace:
    """Return a grayed-out template DAG for the idle state."""
    trace = ExecutionTrace()
    trace.add_node(TraceNode(
        id="tpl-classify", label="Classify", role="router",
        status=NodeStatus.PENDING, parent_ids=[],
    ))
    trace.add_node(TraceNode(
        id="tpl-plan", label="Plan", role="reasoning",
        status=NodeStatus.PENDING, parent_ids=["tpl-classify"],
    ))
    trace.add_node(TraceNode(
        id="tpl-execute", label="Execute", role="coding",
        status=NodeStatus.PENDING, parent_ids=["tpl-plan"],
    ))
    trace.add_node(TraceNode(
        id="tpl-review", label="Review", role="reviewer",
        status=NodeStatus.PENDING, parent_ids=["tpl-execute"],
    ))
    return trace


# ---------------------------------------------------------------------------
# Relative timestamp helper
# ---------------------------------------------------------------------------

def _relative_time(iso_ts: str) -> str:
    """Convert an ISO-8601 timestamp to a human-friendly relative string."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        secs = delta.total_seconds()
        if secs < 60:
            return "just now"
        if secs < 3600:
            m = int(secs // 60)
            return f"{m}m ago"
        if secs < 86400:
            h = int(secs // 3600)
            return f"{h}h ago"
        days = int(secs // 86400)
        if days == 1:
            return "Yesterday"
        if days < 7:
            return f"{days}d ago"
        return dt.strftime("%b %d")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# WorkspaceWorker — background execution via QThread
# ---------------------------------------------------------------------------

class WorkspaceWorker(QObject):
    """Runs :meth:`AuraRouterAPI.execute_task` off the main thread.

    Emits fine-grained signals so the panel can update the DAG, output
    area, and status bar in real time.
    """

    # Fine-grained progress signals
    intent_detected = Signal(str, int)          # intent, complexity
    plan_generated = Signal(list)               # list[str]
    step_started = Signal(int, str)             # index, description
    step_completed = Signal(int, str)           # index, output_chunk
    token_received = Signal(str)                # single token
    model_tried = Signal(str, str, bool, float) # role, model_id, success, elapsed
    review_result = Signal(str, str)            # verdict, feedback

    # Terminal signals
    finished = Signal(str)                      # final output
    error = Signal(str)                         # error message

    # DAG trace signals (compatible with DAGVisualizer)
    trace_node_added = Signal(dict)
    trace_node_updated = Signal(str, dict)

    def __init__(
        self,
        api: AuraRouterAPI,
        task: str,
        context: str,
        output_format: str,
        local_only: bool = False,
        skip_review: bool = False,
    ) -> None:
        super().__init__()
        self._api = api
        self._task = task
        self._context = context
        self._output_format = output_format
        self._local_only = local_only
        self._skip_review = skip_review
        self._cancelled = False

        # Tracking state for DAG node emission
        self._step_count = 0
        self._total_tokens = 0
        self._model_count: set[str] = set()
        self._t0 = 0.0

    def cancel(self) -> None:
        """Request cancellation (best-effort; checked between steps)."""
        self._cancelled = True

    def run(self) -> None:
        """Entry point — called from QThread.started."""
        try:
            self._t0 = time.monotonic()
            self._run_pipeline()
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------ #
    # Internal pipeline
    # ------------------------------------------------------------------ #

    def _run_pipeline(self) -> None:
        from aurarouter.routing import (
            analyze_intent,
            generate_correction_plan,
            generate_plan,
            review_output,
        )

        fabric = self._api._fabric  # noqa: SLF001 — internal access

        # --- Classify -------------------------------------------------------
        self.trace_node_added.emit({
            "id": "classify-0", "label": "Classify", "role": "router",
            "status": "running", "parent_ids": [],
        })

        triage = analyze_intent(fabric, self._task)
        intent = triage.intent if hasattr(triage, "intent") else str(triage)
        complexity = triage.complexity if hasattr(triage, "complexity") else 5

        self.trace_node_updated.emit("classify-0", {
            "status": "success", "result_preview": intent,
        })
        self.intent_detected.emit(intent, complexity)

        if self._cancelled:
            return

        if intent in ("SIMPLE_CODE", "DIRECT"):
            # --- Direct execution -------------------------------------------
            self.trace_node_added.emit({
                "id": "execute-0", "label": "Execute", "role": "coding",
                "status": "running", "parent_ids": ["classify-0"],
            })

            prompt = (
                f"TASK: {self._task}\n"
                f"LANG: {self._output_format}\n"
                f"CONTEXT: {self._context}\n"
                "RESPOND WITH OUTPUT ONLY."
            )
            gen_result = fabric.execute(
                "coding", prompt, on_model_tried=self._on_model_tried,
                on_token=self.token_received.emit,
            )
            result_text = gen_result.text if gen_result else ""

            self.trace_node_updated.emit("execute-0", {
                "status": "success" if result_text else "failed",
                "result_preview": result_text[:200],
            })
            output = result_text or "Error: Generation failed."
        else:
            # --- Multi-step: plan -------------------------------------------
            self.trace_node_added.emit({
                "id": "plan-0", "label": "Plan", "role": "reasoning",
                "status": "running", "parent_ids": ["classify-0"],
            })

            plan = generate_plan(fabric, self._task, self._context)

            self.trace_node_updated.emit("plan-0", {
                "status": "success", "result_preview": str(plan)[:200],
            })
            self.plan_generated.emit(plan)

            # --- Execute steps ----------------------------------------------
            parts: list[str] = []
            for i, step in enumerate(plan):
                if self._cancelled:
                    return

                node_id = f"step-{i}"
                self.trace_node_added.emit({
                    "id": node_id, "label": f"Step {i + 1}", "role": "coding",
                    "status": "running", "parent_ids": ["plan-0"],
                })
                self.step_started.emit(i, step)

                prompt = (
                    f"GOAL: {step}\n"
                    f"LANG: {self._output_format}\n"
                    f"CONTEXT: {self._context}\n"
                    f"PREVIOUS_OUTPUT: {parts}\n"
                    "Return ONLY the requested output."
                )
                gen_result = fabric.execute(
                    "coding", prompt, on_model_tried=self._on_model_tried,
                    on_token=self.token_received.emit,
                )
                result_text = gen_result.text if gen_result else ""
                chunk = (
                    f"\n# --- Step {i + 1}: {step} ---\n{result_text}"
                    if result_text
                    else f"\n# Step {i + 1} Failed."
                )
                parts.append(chunk)

                self.trace_node_updated.emit(node_id, {
                    "status": "success" if result_text else "failed",
                    "result_preview": result_text[:200],
                })
                self.step_completed.emit(i, chunk)

            output = "\n".join(parts)

        # --- Review loop (closed-loop execution) ----------------------------
        if not self._skip_review:
            max_iterations = fabric.get_max_review_iterations()
            reviewer_chain = fabric.config.get_role_chain("reviewer")

            if max_iterations > 0 and reviewer_chain:
                for iteration in range(1, max_iterations + 1):
                    if self._cancelled:
                        return

                    self.trace_node_added.emit({
                        "id": f"review-{iteration}",
                        "label": f"Review #{iteration}",
                        "role": "reviewer",
                        "status": "running",
                        "parent_ids": [],
                    })

                    review = review_output(
                        fabric, self._task, output, iteration=iteration,
                    )

                    verdict_status = (
                        "success" if review.verdict.upper() == "PASS" else "failed"
                    )
                    self.trace_node_updated.emit(f"review-{iteration}", {
                        "status": verdict_status,
                        "result_preview": review.verdict,
                    })
                    self.review_result.emit(review.verdict, review.feedback)

                    if review.verdict.upper() == "PASS":
                        break
                    if iteration == max_iterations:
                        break

                    # Generate and execute correction plan
                    correction_steps = generate_correction_plan(
                        fabric, self._task, output, review,
                    )

                    corrected: list[str] = []
                    for ci, cstep in enumerate(correction_steps):
                        if self._cancelled:
                            return

                        cnode_id = f"correction-{iteration}-step-{ci}"
                        self.trace_node_added.emit({
                            "id": cnode_id,
                            "label": f"Correct {iteration}.{ci + 1}",
                            "role": "coding",
                            "status": "running",
                            "parent_ids": [f"review-{iteration}"],
                        })
                        step_prompt = (
                            f"GOAL: {cstep}\nCONTEXT: {self._context}\n"
                            f"PREVIOUS_OUTPUT:\n{output}\n"
                            f"REVIEWER_FEEDBACK: {review.feedback}"
                        )
                        chunk_result = fabric.execute("coding", step_prompt)
                        chunk = chunk_result.text if chunk_result else ""
                        status = "success" if chunk else "failed"
                        self.trace_node_updated.emit(cnode_id, {"status": status})
                        corrected.append(
                            chunk or f"\n# Correction Step {ci + 1} Failed."
                        )

                    output = "\n".join(corrected)

        self.finished.emit(output)

    def _on_model_tried(
        self, role: str, model_id: str, success: bool, elapsed: float,
    ) -> None:
        self._model_count.add(model_id)
        self.model_tried.emit(role, model_id, success, elapsed)


# ---------------------------------------------------------------------------
# WorkspacePanel — the hero screen
# ---------------------------------------------------------------------------

class WorkspacePanel(QWidget):
    """Primary task execution panel with DAG visualization."""

    # Signals for shell to connect keyboard shortcuts
    execute_requested = Signal()
    new_requested = Signal()
    cancel_requested = Signal()

    def __init__(
        self,
        api: AuraRouterAPI,
        help_registry=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._api = api
        self._help_registry = help_registry

        self._thread: Optional[QThread] = None
        self._worker: Optional[WorkspaceWorker] = None
        self._history: list[dict] = self._load_history()
        self._executing = False

        self._build_ui()
        self._wire_signals()

    # ================================================================== #
    # UI Construction
    # ================================================================== #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Main three-column splitter
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- LEFT: History Sidebar ----
        self._main_splitter.addWidget(self._build_history_sidebar())

        # ---- CENTER: Task Input + DAG + Output ----
        self._main_splitter.addWidget(self._build_center_column())

        # ---- RIGHT: Context Panel ----
        self._main_splitter.addWidget(self._build_context_panel())

        # Set initial column widths
        self._main_splitter.setSizes([180, 600, 200])
        self._main_splitter.setStretchFactor(0, 0)  # left fixed
        self._main_splitter.setStretchFactor(1, 1)  # center stretches
        self._main_splitter.setStretchFactor(2, 0)  # right fixed

        root.addWidget(self._main_splitter)

        # ---- Status bar at bottom ----
        self._status_bar = QLabel("Ready")
        self._status_bar.setStyleSheet(
            f"color: {DARK_PALETTE.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px; "
            f"padding: {SPACING.xs}px {SPACING.sm}px; "
            f"background-color: {DARK_PALETTE.bg_secondary}; "
            f"border-top: 1px solid {DARK_PALETTE.separator};"
        )
        root.addWidget(self._status_bar)

    # ---- Left column: History sidebar ----

    def _build_history_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(160)
        sidebar.setMaximumWidth(260)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        layout.setSpacing(SPACING.sm)

        # Header
        header = QLabel("History")
        header.setStyleSheet(
            f"font-size: {TYPOGRAPHY.size_h2}px; "
            f"font-weight: bold; "
            f"color: {DARK_PALETTE.text_primary};"
        )
        layout.addWidget(header)

        # Search box
        self._history_search = QLineEdit()
        self._history_search.setPlaceholderText("Search tasks...")
        self._history_search.setClearButtonEnabled(True)
        self._history_search.textChanged.connect(self._on_history_filter)
        layout.addWidget(self._history_search)

        # Task list
        self._history_list = QListWidget()
        self._history_list.setAlternatingRowColors(True)
        self._history_list.setStyleSheet(
            f"QListWidget {{"
            f"  background-color: {DARK_PALETTE.bg_secondary};"
            f"  border: 1px solid {DARK_PALETTE.border};"
            f"  border-radius: 4px;"
            f"  font-size: {TYPOGRAPHY.size_small}px;"
            f"}}"
            f"QListWidget::item {{"
            f"  padding: {SPACING.sm}px;"
            f"  border-bottom: 1px solid {DARK_PALETTE.separator};"
            f"}}"
            f"QListWidget::item:selected {{"
            f"  background-color: {DARK_PALETTE.bg_selected};"
            f"}}"
            f"QListWidget::item:hover {{"
            f"  background-color: {DARK_PALETTE.bg_hover};"
            f"}}"
        )
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        layout.addWidget(self._history_list, 1)

        self._populate_history_list()

        # Clear button
        clear_btn = QPushButton("Clear History")
        clear_btn.setObjectName("danger")
        clear_btn.clicked.connect(self._on_clear_history)
        layout.addWidget(clear_btn)

        return sidebar

    # ---- Center column: Input + DAG + Output ----

    def _build_center_column(self) -> QWidget:
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        layout.setSpacing(0)

        # Vertical splitter for input / DAG / output
        self._center_splitter = QSplitter(Qt.Orientation.Vertical)

        # -- TOP: Task input area --
        self._center_splitter.addWidget(self._build_input_area())

        # -- MIDDLE: DAG Visualizer (always visible) --
        self._center_splitter.addWidget(self._build_dag_area())

        # -- BOTTOM: Output area --
        self._center_splitter.addWidget(self._build_output_area())

        # Set proportions: input small, DAG medium, output large
        self._center_splitter.setSizes([120, 180, 300])
        self._center_splitter.setStretchFactor(0, 0)
        self._center_splitter.setStretchFactor(1, 1)
        self._center_splitter.setStretchFactor(2, 2)

        layout.addWidget(self._center_splitter)
        return center

    def _build_input_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, SPACING.sm)
        layout.setSpacing(SPACING.sm)

        # Label row with help tooltip
        label_row = QHBoxLayout()
        label_row.setSpacing(SPACING.sm)
        label = QLabel("Task Description")
        label.setStyleSheet(
            f"font-weight: bold; color: {DARK_PALETTE.text_primary};"
        )
        label_row.addWidget(label)
        label_row.addStretch()

        help_text = ""
        if self._help_registry:
            topic = self._help_registry.get("concept.pipeline")
            if topic:
                help_text = topic.body
        if not help_text:
            help_text = (
                "<b>Task Input</b><br>"
                "Describe what you need in natural language. "
                "AuraRouter will classify, plan, execute, and review."
            )
        label_row.addWidget(HelpTooltip(help_text))
        layout.addLayout(label_row)

        # Task text input
        self._task_input = QTextEdit()
        self._task_input.setPlaceholderText(
            "Describe what you need...\n"
            "e.g. 'Summarize this document', 'Generate a REST API', "
            "'Analyze the attached data'"
        )
        self._task_input.setMinimumHeight(60)
        self._task_input.setMaximumHeight(140)
        layout.addWidget(self._task_input)

        # Button row with intent selector
        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACING.sm)

        # Intent selector combobox
        self._intent_combo = QComboBox()
        self._intent_combo.setFixedHeight(36)
        self._intent_combo.setMinimumWidth(160)
        self._intent_combo.setToolTip(
            "Select an intent to override automatic classification, "
            "or leave on 'Auto (classify)' for automatic routing."
        )
        self._populate_intent_combo()
        btn_row.addWidget(self._intent_combo)

        self._execute_btn = QPushButton("Execute")
        self._execute_btn.setObjectName("primary")
        self._execute_btn.setFixedHeight(36)
        self._execute_btn.setMinimumWidth(100)
        self._execute_btn.clicked.connect(self._on_execute)
        btn_row.addWidget(self._execute_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        hint = QLabel("Ctrl+Enter to execute")
        hint.setStyleSheet(
            f"color: {DARK_PALETTE.text_disabled}; "
            f"font-size: {TYPOGRAPHY.size_small}px;"
        )
        btn_row.addWidget(hint)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        return area

    def _build_dag_area(self) -> QWidget:
        """Build the always-visible DAG visualizer area."""
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.xs)

        # DAG label row with help
        dag_header = QHBoxLayout()
        dag_header.setSpacing(SPACING.sm)
        dag_label = QLabel("Execution Trace")
        dag_label.setStyleSheet(
            f"font-weight: bold; color: {DARK_PALETTE.text_primary}; "
            f"font-size: {TYPOGRAPHY.size_body}px;"
        )
        dag_header.addWidget(dag_label)
        dag_header.addStretch()

        dag_help_text = ""
        if self._help_registry:
            topic = self._help_registry.get("concept.moe")
            if topic:
                dag_help_text = topic.body
        if not dag_help_text:
            dag_help_text = (
                "<b>DAG Visualizer</b><br>"
                "Shows each pipeline stage as a node. "
                "Green = success, Red = failed. Click a node for details."
            )
        dag_header.addWidget(HelpTooltip(dag_help_text))
        layout.addLayout(dag_header)

        # The DAG visualizer widget — always visible, never collapsed
        self._dag_visualizer = DAGVisualizer()
        # Override the collapsed default: expand and remove collapse controls
        self._dag_visualizer._expanded = True  # noqa: SLF001
        self._dag_visualizer._canvas.setVisible(True)  # noqa: SLF001
        self._dag_visualizer._header.setVisible(False)  # noqa: SLF001 — hide toggle header
        self._dag_visualizer.setMaximumHeight(16777215)  # remove height cap
        self._dag_visualizer.setMinimumHeight(120)
        layout.addWidget(self._dag_visualizer, 1)

        # Set idle template DAG
        idle_trace = _build_idle_trace()
        self._dag_visualizer._trace = idle_trace  # noqa: SLF001
        self._dag_visualizer._canvas.set_trace(idle_trace)  # noqa: SLF001

        # Idle hint label (shown when no execution has run)
        self._dag_idle_label = QLabel("Run a task to see the trace")
        self._dag_idle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dag_idle_label.setStyleSheet(
            f"color: {DARK_PALETTE.text_disabled}; "
            f"font-size: {TYPOGRAPHY.size_small}px; "
            f"font-style: italic;"
        )
        layout.addWidget(self._dag_idle_label)

        # Summary bar below DAG
        self._dag_summary = QLabel("")
        self._dag_summary.setStyleSheet(
            f"color: {DARK_PALETTE.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px; "
            f"padding: {SPACING.xs}px 0;"
        )
        layout.addWidget(self._dag_summary)

        return area

    def _build_output_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, SPACING.sm, 0, 0)
        layout.setSpacing(SPACING.sm)

        # Output header row with help
        out_header = QHBoxLayout()
        out_header.setSpacing(SPACING.sm)
        out_label = QLabel("Output")
        out_label.setStyleSheet(
            f"font-weight: bold; color: {DARK_PALETTE.text_primary};"
        )
        out_header.addWidget(out_label)
        out_header.addStretch()

        out_help_text = ""
        if self._help_registry:
            topic = self._help_registry.get("concept.roles")
            if topic:
                out_help_text = topic.body
        if not out_help_text:
            out_help_text = (
                "<b>Output</b><br>"
                "The generated result appears here. "
                "Syntax highlighting is applied for Python output."
            )
        out_header.addWidget(HelpTooltip(out_help_text))
        layout.addLayout(out_header)

        # Output display
        self._output_display = QTextEdit()
        self._output_display.setReadOnly(True)
        mono_font = QFont(TYPOGRAPHY.family_mono, TYPOGRAPHY.size_mono)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self._output_display.setFont(mono_font)
        layout.addWidget(self._output_display, 1)

        # Syntax highlighter (applied to the output document)
        self._highlighter = _PythonHighlighter(self._output_display.document())

        # Copy button
        copy_row = QHBoxLayout()
        copy_row.addStretch()
        self._copy_btn = QPushButton("Copy to Clipboard")
        self._copy_btn.setFixedHeight(28)
        self._copy_btn.clicked.connect(self._on_copy_output)
        copy_row.addWidget(self._copy_btn)
        layout.addLayout(copy_row)

        return area

    # ---- Right column: Context Panel ----

    def _build_context_panel(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setMinimumWidth(180)
        wrapper.setMaximumWidth(320)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        layout.setSpacing(SPACING.md)

        # Context header with help
        ctx_header = QHBoxLayout()
        ctx_label = QLabel("Context")
        ctx_label.setStyleSheet(
            f"font-size: {TYPOGRAPHY.size_h2}px; "
            f"font-weight: bold; "
            f"color: {DARK_PALETTE.text_primary};"
        )
        ctx_header.addWidget(ctx_label)
        ctx_header.addStretch()

        panel_help_text = ""
        if self._help_registry:
            topic = self._help_registry.get("panel.workspace")
            if topic:
                panel_help_text = topic.body
        if not panel_help_text:
            panel_help_text = (
                "<b>Context Panel</b><br>"
                "Attach files, set output format, and configure execution settings."
            )
        ctx_header.addWidget(HelpTooltip(panel_help_text))
        layout.addLayout(ctx_header)

        # File attachments
        attach_label = QLabel("File Attachments")
        attach_label.setStyleSheet(
            f"font-weight: bold; font-size: {TYPOGRAPHY.size_body}px; "
            f"color: {DARK_PALETTE.text_secondary};"
        )
        layout.addWidget(attach_label)

        self._context_input = DocumentInputWidget()
        layout.addWidget(self._context_input)

        # Output format dropdown
        fmt_label = QLabel("Output Format")
        fmt_label.setStyleSheet(
            f"font-weight: bold; font-size: {TYPOGRAPHY.size_body}px; "
            f"color: {DARK_PALETTE.text_secondary};"
        )
        layout.addWidget(fmt_label)

        self._format_combo = QComboBox()
        self._format_combo.addItems(_OUTPUT_FORMATS)
        layout.addWidget(self._format_combo)

        # Execution settings (collapsible)
        self._settings_section = CollapsibleSection(
            "Execution Settings", initially_expanded=False,
        )

        settings_inner = QWidget()
        settings_layout = QVBoxLayout(settings_inner)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(SPACING.sm)

        self._local_only_cb = QCheckBox("Local only")
        self._local_only_cb.setToolTip(
            "Skip cloud models; use only local inference backends."
        )
        settings_layout.addWidget(self._local_only_cb)

        self._skip_review_cb = QCheckBox("Skip review")
        self._skip_review_cb.setToolTip(
            "Skip the review/correction loop after execution."
        )
        settings_layout.addWidget(self._skip_review_cb)

        self._settings_section.add_widget(settings_inner)
        layout.addWidget(self._settings_section)

        layout.addStretch()

        scroll.setWidget(panel)
        wrapper_layout.addWidget(scroll)
        return wrapper

    # ================================================================== #
    # Intent Combobox
    # ================================================================== #

    def _populate_intent_combo(self) -> None:
        """Populate the intent combobox with Auto, built-in, and analyzer intents."""
        self._intent_combo.blockSignals(True)
        self._intent_combo.clear()

        # First item: Auto (classify)
        self._intent_combo.addItem("Auto (classify)", None)

        # Separator
        self._intent_combo.insertSeparator(self._intent_combo.count())

        # Group label: Built-in (disabled)
        self._intent_combo.addItem("Built-in")
        builtin_label_idx = self._intent_combo.count() - 1
        model = self._intent_combo.model()
        item = model.item(builtin_label_idx)
        if item is not None:
            item.setEnabled(False)

        # Built-in intents
        for defn in IntentRegistry.BUILTIN_INTENTS:
            self._intent_combo.addItem(
                f"  {defn.name}", defn.name,
            )

        # Analyzer-declared intents
        try:
            config = self._api._config  # noqa: SLF001
            registry = build_intent_registry(config)
            analyzer_id = config.get_active_analyzer() or "aurarouter-default"

            # Collect analyzer-specific intents (source != "builtin")
            analyzer_intents = [
                defn for defn in registry.get_all()
                if defn.source != "builtin"
            ]
            if analyzer_intents:
                # Get display name for the analyzer
                analyzer_data = config.catalog_get(analyzer_id)
                display_name = analyzer_id
                if analyzer_data:
                    display_name = analyzer_data.get("display_name", analyzer_id)

                self._intent_combo.insertSeparator(self._intent_combo.count())
                self._intent_combo.addItem(f"Analyzer: {display_name}")
                analyzer_label_idx = self._intent_combo.count() - 1
                item = model.item(analyzer_label_idx)
                if item is not None:
                    item.setEnabled(False)

                for defn in analyzer_intents:
                    self._intent_combo.addItem(
                        f"  {defn.name}", defn.name,
                    )
        except Exception:
            pass  # Gracefully degrade if config unavailable

        self._intent_combo.setCurrentIndex(0)
        self._intent_combo.blockSignals(False)

    def refresh_intents(self, _analyzer_id: str = "") -> None:
        """Rebuild the intent combobox (e.g. after analyzer change).

        Parameters
        ----------
        _analyzer_id:
            Ignored — present so the method can be connected directly to
            an ``analyzer_changed(str)`` signal.
        """
        self._populate_intent_combo()

    def get_selected_intent(self) -> str | None:
        """Return the selected intent name, or ``None`` for Auto."""
        return self._intent_combo.currentData()

    # ================================================================== #
    # Signal Wiring
    # ================================================================== #

    def _wire_signals(self) -> None:
        self.execute_requested.connect(self._on_execute)
        self.new_requested.connect(self._on_new)
        self.cancel_requested.connect(self._on_cancel)

    # ================================================================== #
    # Public API
    # ================================================================== #

    def set_api(self, api: AuraRouterAPI) -> None:
        """Replace the API instance (e.g. after environment switch)."""
        self._api = api

    def is_executing(self) -> bool:
        """Return True if a task is currently running."""
        return self._executing

    # ================================================================== #
    # Execution
    # ================================================================== #

    def _on_execute(self) -> None:
        task = self._task_input.toPlainText().strip()
        if not task:
            self._status_bar.setText("Error: Task description is empty.")
            return

        self._output_display.clear()
        self._dag_idle_label.setVisible(False)
        self._dag_summary.setText("")
        self._dag_visualizer.reset()
        # Re-expand after reset (reset collapses it)
        self._dag_visualizer._expanded = True  # noqa: SLF001
        self._dag_visualizer._canvas.setVisible(True)  # noqa: SLF001
        self._dag_visualizer.setMaximumHeight(16777215)  # noqa: SLF001

        self._set_executing(True)
        self._status_bar.setText("Running inference...")

        self._worker = WorkspaceWorker(
            api=self._api,
            task=task,
            context=self._context_input.get_context(),
            output_format=self._format_combo.currentText(),
            local_only=self._local_only_cb.isChecked(),
            skip_review=self._skip_review_cb.isChecked(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        # Wire worker signals
        self._thread.started.connect(self._worker.run)

        self._worker.intent_detected.connect(self._on_intent)
        self._worker.plan_generated.connect(self._on_plan)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_completed.connect(self._on_step_completed)
        self._worker.token_received.connect(self._on_token_received)
        self._worker.model_tried.connect(self._dag_visualizer.on_model_tried)
        self._worker.review_result.connect(self._on_review)

        # DAG trace signals
        self._worker.trace_node_added.connect(self._dag_visualizer.add_node)
        self._worker.trace_node_updated.connect(self._dag_visualizer.update_node)
        self._worker.intent_detected.connect(
            lambda intent, _comp: self._dag_visualizer.on_intent_detected(intent)
        )

        # Terminal signals
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Thread cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_cancel(self) -> None:
        """Cancel a running execution."""
        if self._worker:
            self._worker.cancel()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
            self._cleanup_thread()
            self._set_executing(False)
            self._status_bar.setText("Execution cancelled.")

    def _on_new(self) -> None:
        """Clear all inputs for a fresh prompt."""
        self._task_input.clear()
        self._context_input.clear()
        self._output_display.clear()
        self._dag_visualizer.reset()
        # Restore idle DAG
        self._dag_visualizer._expanded = True  # noqa: SLF001
        self._dag_visualizer._canvas.setVisible(True)  # noqa: SLF001
        self._dag_visualizer._header.setVisible(False)  # noqa: SLF001
        self._dag_visualizer.setMaximumHeight(16777215)
        idle_trace = _build_idle_trace()
        self._dag_visualizer._trace = idle_trace  # noqa: SLF001
        self._dag_visualizer._canvas.set_trace(idle_trace)  # noqa: SLF001
        self._dag_idle_label.setVisible(True)
        self._dag_summary.setText("")
        self._status_bar.setText("Ready")

    # ================================================================== #
    # Worker signal slots
    # ================================================================== #

    def _on_intent(self, intent: str, complexity: int) -> None:
        selected = self.get_selected_intent()
        if selected:
            intent_label = f"Intent: {selected} (explicit)"
        else:
            intent_label = f"Intent: {intent} (classified)"
        if intent in ("SIMPLE_CODE", "DIRECT"):
            self._status_bar.setText(f"{intent_label} -- generating response...")
        else:
            self._status_bar.setText(
                f"{intent_label} (complexity {complexity}) -- generating plan..."
            )

    def _on_plan(self, plan: list) -> None:
        self._status_bar.setText(f"Plan generated: {len(plan)} steps")

    def _on_step_started(self, index: int, description: str) -> None:
        self._status_bar.setText(f"Step {index + 1}: {description}")
        # Insert header into output display for multi-step tasks
        self._output_display.append(f"\n# --- Step {index + 1}: {description} ---\n")

    def _on_step_completed(self, index: int, result: str) -> None:
        # If generation failed, append the error message. Otherwise, the
        # tokens have already been inserted via _on_token_received.
        if "Failed." in result:
            self._output_display.append(result)

    def _on_token_received(self, token: str) -> None:
        self._output_display.insertPlainText(token)
        self._output_display.ensureCursorVisible()

    def _on_review(self, verdict: str, feedback: str) -> None:
        self._status_bar.setText(f"Review: {verdict}")

    def _on_finished(self, result: str) -> None:
        if not self._output_display.toPlainText():
            self._output_display.setPlainText(result)

        self._set_executing(False)

        # Build summary bar
        trace = self._dag_visualizer._trace  # noqa: SLF001
        summary = trace.summary() if trace else ""
        model_count = len({
            n.model_id for n in trace.nodes.values()
            if n.model_id
        }) if trace else 0
        total_tokens = sum(
            n.input_tokens + n.output_tokens
            for n in trace.nodes.values()
        ) if trace else 0
        extra_parts = []
        if model_count:
            extra_parts.append(f"{model_count} model{'s' if model_count != 1 else ''}")
        if total_tokens:
            extra_parts.append(f"{total_tokens:,} tokens")
        if extra_parts:
            summary += " \u00b7 " + " \u00b7 ".join(extra_parts)
        self._dag_summary.setText(summary)

        self._status_bar.setText("Done.")

        # Save to history
        task = self._task_input.toPlainText().strip()
        if task:
            self._add_to_history(
                task,
                self._output_display.toPlainText(),
                context=self._context_input.get_context(),
            )

    def _on_error(self, message: str) -> None:
        self._output_display.setPlainText(f"ERROR: {message}")
        self._set_executing(False)
        self._status_bar.setText(f"Error: {message}")

    def _cleanup_thread(self) -> None:
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    # ================================================================== #
    # UI state helpers
    # ================================================================== #

    def _set_executing(self, running: bool) -> None:
        self._executing = running
        self._execute_btn.setEnabled(not running)
        self._cancel_btn.setVisible(running)
        self._task_input.setReadOnly(running)

    # ================================================================== #
    # Clipboard
    # ================================================================== #

    def _on_copy_output(self) -> None:
        text = self._output_display.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)
                self._status_bar.setText("Copied to clipboard.")
        else:
            self._status_bar.setText("Nothing to copy.")

    # ================================================================== #
    # History management
    # ================================================================== #

    @staticmethod
    def _load_history() -> list[dict]:
        try:
            if _HISTORY_PATH.is_file():
                data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data[:_HISTORY_MAX]
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

    def _add_to_history(
        self, task: str, result: str, context: str = "",
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "task": task,
            "result": result[:4000],
            "context": context[:2000],
            "timestamp": ts,
            "format": self._format_combo.currentText(),
        }
        # Remove duplicate if same task already exists.
        self._history = [h for h in self._history if h.get("task") != task]
        self._history.insert(0, entry)
        self._history = self._history[:_HISTORY_MAX]
        self._save_history()
        self._populate_history_list()

    def _populate_history_list(self, filter_text: str = "") -> None:
        self._history_list.clear()
        q = filter_text.strip().lower()
        for entry in self._history:
            task = entry.get("task", "")
            if q and q not in task.lower():
                continue
            first_line = task.split("\n")[0][:60]
            ts = entry.get("timestamp", "")
            rel = _relative_time(ts) if ts else ""

            # Build display text
            display = first_line
            if rel:
                display += f"  ({rel})"

            item = QListWidgetItem(display)
            item.setToolTip(task)
            item.setData(Qt.ItemDataRole.UserRole, entry)

            # Intent badge (if stored)
            self._history_list.addItem(item)

    def _on_history_filter(self, text: str) -> None:
        self._populate_history_list(text)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, dict):
            return
        self._task_input.setPlainText(entry.get("task", ""))
        result = entry.get("result", "")
        if result:
            self._output_display.setPlainText(result)
        # Restore output format if stored
        fmt = entry.get("format", "")
        if fmt:
            idx = self._format_combo.findText(fmt)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)

    def _on_clear_history(self) -> None:
        self._history.clear()
        self._save_history()
        self._populate_history_list()
        self._status_bar.setText("History cleared.")
