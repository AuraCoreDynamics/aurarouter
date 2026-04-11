"""Complexity badge widget."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Typography


class ComplexityBadge(QWidget):
    """Pill badge showing complexity score (1–10) with color coding."""

    def __init__(self, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._score: int = 0
        self._tooltip_extra: str = ""
        self.setFixedSize(32, 20)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_score(self, score: int, tooltip_extra: str = "") -> None:
        self._score = max(1, min(10, score))
        self._tooltip_extra = tooltip_extra
        tooltip = f"Complexity: {self._score}/10"
        if self._score >= 8:
            tooltip += " — Triggers monologue reasoning"
        elif self._score >= 7:
            tooltip += " — May trigger speculative decoding"
        if tooltip_extra:
            tooltip += f"\n{tooltip_extra}"
        self.setToolTip(tooltip)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._score:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Color by band
        if self._score <= 3:
            color = QColor(self._palette.success)
        elif self._score <= 6:
            color = QColor(self._palette.warning)
        else:
            color = QColor(self._palette.error)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 6, 6)

        # Text
        painter.setPen(QColor(self._palette.text_inverse))
        font = painter.font()
        font.setPointSize(Typography.size_small)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, str(self._score))
        painter.end()
