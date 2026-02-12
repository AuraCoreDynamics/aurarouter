"""Real-time routing pipeline visualization widget."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# Stage labels (generic, not code-specific).
_STAGE_LABELS = {
    "router": "Classifier",
    "reasoning": "Planner",
    "coding": "Worker",
}


class _StageBox(QFrame):
    """Visual box representing a single pipeline stage."""

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label = label
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setMinimumWidth(150)
        self.setMinimumHeight(64)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._title = QLabel(label)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self._title)

        self._model_label = QLabel("--")
        self._model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self._model_label)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._status_label)

        self._attempts: list[tuple[str, bool, float]] = []
        self._apply_idle_style()

    def reset(self) -> None:
        """Reset to initial idle state."""
        self._model_label.setText("--")
        self._status_label.setText("")
        self._attempts.clear()
        self._apply_idle_style()

    def set_active(self) -> None:
        """Mark stage as currently running."""
        self._status_label.setText("running...")
        self._status_label.setStyleSheet("color: #1565c0; font-size: 10px;")
        self.setStyleSheet(
            "QFrame { border: 2px solid #1565c0; background-color: #e3f2fd; }"
        )

    def record_attempt(self, model_id: str, success: bool, elapsed: float) -> None:
        """Record a model attempt for this stage."""
        self._attempts.append((model_id, success, elapsed))

        if success:
            self._model_label.setText(model_id)
            self._status_label.setText(f"done {elapsed:.1f}s")
            self._status_label.setStyleSheet("color: #388e3c; font-size: 10px;")
            self.setStyleSheet(
                "QFrame { border: 2px solid #388e3c; background-color: #e8f5e9; }"
            )
        else:
            # Show failed attempt with strike-through hint.
            self._model_label.setText(f"{model_id} (failed)")
            self._model_label.setStyleSheet(
                "color: #d32f2f; text-decoration: line-through; font-size: 10px;"
            )
            self._status_label.setText(f"failed {elapsed:.1f}s")
            self._status_label.setStyleSheet("color: #d32f2f; font-size: 10px;")
            self.setStyleSheet(
                "QFrame { border: 2px solid #f9a825; background-color: #fff8e1; }"
            )

    def set_skipped(self) -> None:
        """Mark stage as skipped (e.g. direct intent skips planning)."""
        self._status_label.setText("skipped")
        self._status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.setStyleSheet(
            "QFrame { border: 1px solid #bdbdbd; background-color: #f5f5f5; }"
        )

    def _apply_idle_style(self) -> None:
        self.setStyleSheet(
            "QFrame { border: 1px solid #bdbdbd; background-color: #fafafa; }"
        )
        self._model_label.setStyleSheet("color: gray; font-size: 10px;")
        self._status_label.setStyleSheet("font-size: 10px;")

    def get_tooltip_info(self) -> str:
        """Build a tooltip summarizing all attempts."""
        if not self._attempts:
            return f"{self._label}: no attempts"
        lines = [f"{self._label}:"]
        for model_id, success, elapsed in self._attempts:
            status = "OK" if success else "FAIL"
            lines.append(f"  {model_id}: {status} ({elapsed:.2f}s)")
        return "\n".join(lines)


class RoutingVisualizer(QWidget):
    """Horizontal pipeline visualization: Classifier -> Planner -> Worker.

    Updated in real-time via :meth:`on_model_tried` connected to
    ``InferenceWorker.model_tried``.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stages: dict[str, _StageBox] = {}

        for role, display_label in _STAGE_LABELS.items():
            box = _StageBox(display_label, parent=self)
            self._stages[role] = box
            layout.addWidget(box)

            # Add arrow between stages (except after last).
            if role != list(_STAGE_LABELS.keys())[-1]:
                arrow = QLabel("\u2192")  # right arrow
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("font-size: 18px; color: #757575;")
                layout.addWidget(arrow)

        self.setMaximumHeight(80)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all stages to idle."""
        for box in self._stages.values():
            box.reset()

    def set_stage_active(self, role: str) -> None:
        """Mark a stage as currently executing."""
        if role in self._stages:
            self._stages[role].set_active()

    def set_stage_skipped(self, role: str) -> None:
        """Mark a stage as skipped."""
        if role in self._stages:
            self._stages[role].set_skipped()

    def on_model_tried(
        self, role: str, model_id: str, success: bool, elapsed: float
    ) -> None:
        """Slot: called when a model attempt completes.

        Connect to ``InferenceWorker.model_tried(str, str, bool, float)``.
        """
        if role in self._stages:
            self._stages[role].record_attempt(model_id, success, elapsed)

    def on_intent_detected(self, intent: str) -> None:
        """Update visualization based on detected intent.

        For direct (SIMPLE_CODE) tasks, the planner stage is skipped.
        """
        if intent == "SIMPLE_CODE":
            if "reasoning" in self._stages:
                self._stages["reasoning"].set_skipped()
