"""DAG-based execution visualization widget.

Replaces the old ``RoutingVisualizer`` with a dynamic DAG that builds
as execution plays out.  Collapsed by default (one-line summary),
expandable to show the full DAG with clickable nodes.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.execution_trace import (
    ExecutionTrace,
    ModelAttempt,
    NodeStatus,
    TraceNode,
)

# ------------------------------------------------------------------
# Colours keyed by node status
# ------------------------------------------------------------------

_STATUS_BORDER: dict[NodeStatus, QColor] = {
    NodeStatus.PENDING: QColor("#bdbdbd"),
    NodeStatus.RUNNING: QColor("#1565c0"),
    NodeStatus.SUCCESS: QColor("#388e3c"),
    NodeStatus.FAILED: QColor("#d32f2f"),
    NodeStatus.SKIPPED: QColor("#9e9e9e"),
}

_STATUS_BG: dict[NodeStatus, QColor] = {
    NodeStatus.PENDING: QColor("#fafafa"),
    NodeStatus.RUNNING: QColor("#e3f2fd"),
    NodeStatus.SUCCESS: QColor("#e8f5e9"),
    NodeStatus.FAILED: QColor("#ffebee"),
    NodeStatus.SKIPPED: QColor("#f5f5f5"),
}

# ------------------------------------------------------------------
# Layout constants
# ------------------------------------------------------------------

_NODE_W = 140
_NODE_H = 52
_H_GAP = 40
_V_GAP = 16
_MARGIN = 12


# ------------------------------------------------------------------
# Node detail dialog
# ------------------------------------------------------------------

class _NodeDetailDialog(QDialog):
    """Pop-up showing telemetry for a single DAG node."""

    def __init__(self, node: TraceNode, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Node: {node.label}")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        layout.addRow("Label:", QLabel(node.label))
        layout.addRow("Role:", QLabel(node.role))
        layout.addRow("Status:", QLabel(node.status.value))
        if node.model_id:
            layout.addRow("Model:", QLabel(node.model_id))
        layout.addRow("Elapsed:", QLabel(f"{node.elapsed_s:.2f}s"))
        layout.addRow("Input tokens:", QLabel(str(node.input_tokens)))
        layout.addRow("Output tokens:", QLabel(str(node.output_tokens)))

        if node.attempts:
            layout.addRow(QLabel(""))
            layout.addRow("Attempts:", QLabel(f"{len(node.attempts)} total"))
            for i, att in enumerate(node.attempts):
                status = "OK" if att.success else "FAIL"
                text = f"{att.model_id}: {status} ({att.elapsed_s:.2f}s)"
                if att.error:
                    text += f" \u2014 {att.error}"
                layout.addRow(f"  #{i + 1}:", QLabel(text))

        if node.result_preview:
            preview = QTextEdit()
            preview.setReadOnly(True)
            preview.setPlainText(node.result_preview)
            preview.setMaximumHeight(120)
            layout.addRow("Result:", preview)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addRow(close_btn)


# ------------------------------------------------------------------
# QPainter canvas
# ------------------------------------------------------------------

class _DAGCanvas(QWidget):
    """Custom widget that paints the execution DAG."""

    node_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._trace = ExecutionTrace()
        self._node_rects: dict[str, QRectF] = {}
        self.setMouseTracking(True)
        self._hover_node: Optional[str] = None

    def set_trace(self, trace: ExecutionTrace) -> None:
        self._trace = trace
        self._layout_nodes()
        self.update()

    # ---- layout ----

    def _layout_nodes(self) -> None:
        """Left-to-right topological layout: columns by depth, rows within."""
        self._node_rects.clear()
        if not self._trace.nodes:
            self.setMinimumSize(0, 0)
            return

        # BFS to compute max-depth of each node.
        depths: dict[str, int] = {}
        roots = self._trace.get_roots()
        queue: list[tuple[str, int]] = [(r.id, 0) for r in roots]
        while queue:
            nid, d = queue.pop(0)
            if nid in depths:
                depths[nid] = max(depths[nid], d)
            else:
                depths[nid] = d
            for child in self._trace.get_children(nid):
                queue.append((child.id, d + 1))

        # Group nodes into columns by depth.
        columns: dict[int, list[str]] = {}
        for nid, d in depths.items():
            columns.setdefault(d, []).append(nid)

        max_col = max(columns.keys()) if columns else 0
        for col_idx in range(max_col + 1):
            col_nodes = columns.get(col_idx, [])
            x = _MARGIN + col_idx * (_NODE_W + _H_GAP)
            for row_idx, nid in enumerate(col_nodes):
                y = _MARGIN + row_idx * (_NODE_H + _V_GAP)
                self._node_rects[nid] = QRectF(x, y, _NODE_W, _NODE_H)

        if self._node_rects:
            max_x = max(r.right() for r in self._node_rects.values()) + _MARGIN
            max_y = max(r.bottom() for r in self._node_rects.values()) + _MARGIN
        else:
            max_x = max_y = 0
        self.setMinimumSize(int(max_x), int(max_y))

    # ---- painting ----

    def paintEvent(self, event) -> None:
        if not self._trace.nodes:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Edges (behind nodes).
        edge_pen = QPen(QColor("#757575"), 1.5)
        painter.setPen(edge_pen)
        for node in self._trace.nodes.values():
            child_rect = self._node_rects.get(node.id)
            if child_rect is None:
                continue
            for pid in node.parent_ids:
                parent_rect = self._node_rects.get(pid)
                if parent_rect is None:
                    continue
                x1 = parent_rect.right()
                y1 = parent_rect.center().y()
                x2 = child_rect.left()
                y2 = child_rect.center().y()
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                self._draw_arrowhead(painter, x1, y1, x2, y2)

        # Nodes.
        for nid, rect in self._node_rects.items():
            node = self._trace.nodes.get(nid)
            if node is not None:
                self._draw_node(painter, node, rect)

        painter.end()

    def _draw_node(
        self, painter: QPainter, node: TraceNode, rect: QRectF
    ) -> None:
        border = _STATUS_BORDER.get(node.status, QColor("#bdbdbd"))
        bg = _STATUS_BG.get(node.status, QColor("#fafafa"))

        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        painter.fillPath(path, bg)

        pen_w = 2.0 if node.id == self._hover_node else 1.5
        painter.setPen(QPen(border, pen_w))
        painter.drawRoundedRect(rect, 6, 6)

        # Label (top line).
        painter.setPen(QPen(QColor("#212121")))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        label_rect = QRectF(rect.x() + 4, rect.y() + 4, rect.width() - 8, 18)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, node.label)

        # Sub-text (bottom line).
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#616161")))
        sub_rect = QRectF(rect.x() + 4, rect.y() + 24, rect.width() - 8, 16)
        sub = node.model_id or node.status.value
        if node.elapsed_s > 0:
            sub += f" ({node.elapsed_s:.1f}s)"
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, sub)

    @staticmethod
    def _draw_arrowhead(
        painter: QPainter,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        p1x = x2 - size * math.cos(angle - 0.4)
        p1y = y2 - size * math.sin(angle - 0.4)
        p2x = x2 - size * math.cos(angle + 0.4)
        p2y = y2 - size * math.sin(angle + 0.4)

        path = QPainterPath()
        path.moveTo(x2, y2)
        path.lineTo(p1x, p1y)
        path.lineTo(p2x, p2y)
        path.closeSubpath()
        painter.fillPath(path, QColor("#757575"))

    # ---- mouse interaction ----

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        for nid, rect in self._node_rects.items():
            if rect.contains(pos):
                self.node_clicked.emit(nid)
                return

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        old_hover = self._hover_node
        self._hover_node = None
        for nid, rect in self._node_rects.items():
            if rect.contains(pos):
                self._hover_node = nid
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                break
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if old_hover != self._hover_node:
            self.update()


# ------------------------------------------------------------------
# Public widget
# ------------------------------------------------------------------

class DAGVisualizer(QWidget):
    """Execution DAG visualization, collapsed by default.

    Public API
    ----------
    reset()              — clear the trace
    add_node(data)       — add a node from a dict
    update_node(id, upd) — update an existing node
    on_model_tried(...)  — compat slot for InferenceWorker.model_tried
    on_intent_detected() — compat slot for intent signal
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._trace = ExecutionTrace()
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- collapsed header row ----
        self._header = QWidget()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(4, 2, 4, 2)

        self._toggle_btn = QPushButton("\u25b6")
        self._toggle_btn.setFixedSize(24, 24)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.clicked.connect(self._toggle_expanded)
        header_layout.addWidget(self._toggle_btn)

        self._summary_label = QLabel("Execution trace")
        self._summary_label.setStyleSheet("color: #616161; font-size: 11px;")
        header_layout.addWidget(self._summary_label, 1)

        layout.addWidget(self._header)

        # ---- expanded canvas ----
        self._canvas = _DAGCanvas()
        self._canvas.node_clicked.connect(self._on_node_clicked)
        self._canvas.setVisible(False)
        self._canvas.setMaximumHeight(200)
        layout.addWidget(self._canvas)

        self.setMaximumHeight(32)

    # ==================================================================
    # Public API
    # ==================================================================

    def reset(self) -> None:
        """Clear the trace and collapse."""
        self._trace = ExecutionTrace()
        self._expanded = False
        self._canvas.setVisible(False)
        self._canvas.set_trace(self._trace)
        self._toggle_btn.setText("\u25b6")
        self._summary_label.setText("Execution trace")
        self.setMaximumHeight(32)

    def add_node(self, data: dict) -> None:
        """Add a node from a dict emitted by InferenceWorker."""
        node = TraceNode(
            id=data["id"],
            label=data["label"],
            role=data["role"],
            status=NodeStatus(data.get("status", "pending")),
            parent_ids=data.get("parent_ids", []),
            model_id=data.get("model_id"),
            elapsed_s=data.get("elapsed_s", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            result_preview=data.get("result_preview", ""),
        )
        self._trace.add_node(node)
        self._refresh()

    def update_node(self, node_id: str, updates: dict) -> None:
        """Update fields on an existing node."""
        node = self._trace.nodes.get(node_id)
        if node is None:
            return

        if "status" in updates:
            node.status = NodeStatus(updates["status"])
        if "model_id" in updates:
            node.model_id = updates["model_id"]
        if "elapsed_s" in updates:
            node.elapsed_s = updates["elapsed_s"]
        if "input_tokens" in updates:
            node.input_tokens = updates["input_tokens"]
        if "output_tokens" in updates:
            node.output_tokens = updates["output_tokens"]
        if "result_preview" in updates:
            node.result_preview = updates["result_preview"]
        if "attempts" in updates:
            node.attempts = [
                ModelAttempt(**a) if isinstance(a, dict) else a
                for a in updates["attempts"]
            ]
        self._refresh()

    # ------------------------------------------------------------------
    # Compatibility slots (match old RoutingVisualizer interface)
    # ------------------------------------------------------------------

    def on_model_tried(
        self, role: str, model_id: str, success: bool, elapsed: float
    ) -> None:
        """Record a model attempt on the most recent RUNNING node for *role*."""
        target: TraceNode | None = None
        for node in reversed(list(self._trace.nodes.values())):
            if node.role == role and node.status == NodeStatus.RUNNING:
                target = node
                break
        if target is None:
            for node in reversed(list(self._trace.nodes.values())):
                if node.role == role:
                    target = node
                    break
        if target is None:
            return

        target.attempts.append(
            ModelAttempt(model_id=model_id, success=success, elapsed_s=elapsed)
        )
        if success:
            target.model_id = model_id
            target.elapsed_s = elapsed
            target.status = NodeStatus.SUCCESS
        self._refresh()

    def on_intent_detected(self, intent: str) -> None:
        """Mark classify node done; skip reasoning if SIMPLE_CODE."""
        for node in self._trace.nodes.values():
            if node.role == "router":
                node.result_preview = intent
                break

        if intent == "SIMPLE_CODE":
            for node in self._trace.nodes.values():
                if (
                    node.role == "reasoning"
                    and node.status == NodeStatus.PENDING
                ):
                    node.status = NodeStatus.SKIPPED
        self._refresh()

    # ==================================================================
    # Internal
    # ==================================================================

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._canvas.setVisible(self._expanded)
        self._toggle_btn.setText("\u25bc" if self._expanded else "\u25b6")
        self.setMaximumHeight(232 if self._expanded else 32)

    def _refresh(self) -> None:
        self._canvas.set_trace(self._trace)
        self._summary_label.setText(
            self._trace.summary() or "Execution trace"
        )

    def _on_node_clicked(self, node_id: str) -> None:
        node = self._trace.nodes.get(node_id)
        if node is None:
            return
        dlg = _NodeDetailDialog(node, parent=self)
        dlg.exec()
