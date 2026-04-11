"""Timeline widget for displaying ordered events."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Typography


@dataclass
class TimelineEntry:
    """A single entry in a timeline."""

    timestamp: str
    title: str
    status: str = "pending"  # pending, running, success, failed, skipped
    detail: str = ""
    role_color: str = ""
    diff_before: str = ""
    diff_after: str = ""
    metadata: dict = field(default_factory=dict)


class _EntryWidget(QFrame):
    """Widget for a single timeline entry."""

    def __init__(
        self,
        entry: TimelineEntry,
        index: int,
        palette: ColorPalette,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._entry = entry
        self._index = index
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 4, 4, 4)
        layout.setSpacing(2)

        # Header row
        header_text = f"{entry.timestamp}  {self._status_icon(entry.status)}  {entry.title}"
        # Add dual-coding icon for expert roles
        role_icon = self._role_icon(entry.role_color, palette)
        if role_icon:
            header_text = role_icon + " " + header_text

        self._header = QLabel(header_text, self)
        color = self._status_color(entry.status)
        self._header.setStyleSheet(
            f"font-size: {Typography.size_small}pt; color: {color};"
        )
        layout.addWidget(self._header)

        # Detail (hidden by default)
        if entry.detail:
            self._detail = QLabel(entry.detail, self)
            self._detail.setWordWrap(True)
            self._detail.setStyleSheet(
                f"font-size: {Typography.size_small}pt; "
                f"color: {palette.text_secondary}; margin-left: 16px;"
            )
            self._detail.hide()
            layout.addWidget(self._detail)
        else:
            self._detail = None

        # Diff view (hidden, shows on expand when diff data present)
        if entry.diff_before and entry.diff_after:
            self._diff_widget = self._build_diff_widget(entry, palette)
            self._diff_widget.hide()
            layout.addWidget(self._diff_widget)
            self._header.setCursor(Qt.PointingHandCursor)
            self._header.mousePressEvent = lambda _: self._toggle()
        else:
            self._diff_widget = None
            if entry.detail:
                self._header.setCursor(Qt.PointingHandCursor)
                self._header.mousePressEvent = lambda _: self._toggle()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        if self._detail:
            self._detail.setVisible(self._expanded)
        if self._diff_widget:
            self._diff_widget.setVisible(self._expanded)

    def _build_diff_widget(self, entry: TimelineEntry, palette: ColorPalette) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(1)
        diff = list(difflib.unified_diff(
            entry.diff_before.splitlines(),
            entry.diff_after.splitlines(),
            lineterm="",
            n=2,
        ))
        for line in diff[2:]:  # skip the --- +++ header lines
            lbl = QLabel(line, container)
            lbl.setWordWrap(True)
            lbl.setFont(lbl.font())
            if line.startswith("+"):
                lbl.setStyleSheet(
                    f"font-family: monospace; font-size: {Typography.size_small}pt; "
                    f"background: {palette.success}22; color: {palette.success};"
                )
            elif line.startswith("-"):
                lbl.setStyleSheet(
                    f"font-family: monospace; font-size: {Typography.size_small}pt; "
                    f"background: {palette.error}22; color: {palette.error};"
                )
            else:
                lbl.setStyleSheet(
                    f"font-family: monospace; font-size: {Typography.size_small}pt; "
                    f"color: {palette.text_secondary};"
                )
            layout.addWidget(lbl)
        return container

    def _status_icon(self, status: str) -> str:
        return {
            "pending": "○",
            "running": "●",
            "success": "✓",
            "failed": "✗",
            "skipped": "–",
        }.get(status, "○")

    def _status_color(self, status: str) -> str:
        return {
            "pending": self._palette.status_pending,
            "running": self._palette.status_running,
            "success": self._palette.status_success,
            "failed": self._palette.status_failed,
            "skipped": self._palette.status_skipped,
        }.get(status, self._palette.text_secondary)

    def _role_icon(self, role_color: str, palette: ColorPalette) -> str:
        """Return dual-coded icon for known expert role colors."""
        if role_color == palette.monologue_generator:
            return "✏"   # Generator: pen (purple)
        if role_color == palette.monologue_critic:
            return "🔍"  # Critic: magnifying glass (orange)
        if role_color == palette.monologue_refiner:
            return "↻"   # Refiner: cycle (teal)
        return ""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        # Draw node circle and connector
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = 10, 12
        r = 5
        node_color = self._entry.role_color or self._status_color(self._entry.status)
        painter.setBrush(QColor(node_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        # Connector line
        if self.height() > 24:
            painter.setPen(QColor(self._palette.border))
            painter.drawLine(cx, cy + r, cx, self.height())
        painter.end()


class TimelineWidget(QWidget):
    """Vertical timeline with status-colored nodes."""

    entry_clicked = Signal(int)

    def __init__(self, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._entries: list[TimelineEntry] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self._scroll)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(4, 4, 4, 4)
        self._container_layout.setSpacing(0)
        self._container_layout.addStretch()
        self._scroll.setWidget(self._container)

    def set_entries(self, entries: list[TimelineEntry]) -> None:
        self._entries = list(entries)
        self._rebuild()

    def append_entry(self, entry: TimelineEntry) -> None:
        self._entries.append(entry)
        idx = len(self._entries) - 1
        # Insert before the stretch at the end
        widget = _EntryWidget(entry, idx, self._palette, self._container)
        self._container_layout.insertWidget(self._container_layout.count() - 1, widget)

    def clear(self) -> None:
        self._entries.clear()
        self._rebuild()

    def _rebuild(self) -> None:
        # Remove all except the trailing stretch
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, entry in enumerate(self._entries):
            widget = _EntryWidget(entry, i, self._palette, self._container)
            self._container_layout.insertWidget(i, widget)
