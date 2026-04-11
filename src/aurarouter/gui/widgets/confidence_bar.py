"""Confidence bar widget."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Typography


class ConfidenceBar(QWidget):
    """Horizontal bar with color gradient showing a 0.0–1.0 confidence score."""

    def __init__(self, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._score: float = 0.0
        self._label_text: str = ""
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addStretch()
        self._score_label = QLabel("—", self)
        self._score_label.setStyleSheet(
            f"font-size: {Typography.size_small}pt; color: {self._palette.text_secondary};"
        )
        layout.addWidget(self._score_label)

    def set_score(self, score: float, label: str = "") -> None:
        self._score = max(0.0, min(1.0, score))
        self._label_text = label
        display = f"{self._score:.2f}"
        if label:
            display = f"{label}: {display}"
        self._score_label.setText(display)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Bar area (leave space for label)
        label_w = self._score_label.sizeHint().width() + 8
        bar_rect = self.rect().adjusted(0, 4, -label_w, -4)
        painter.fillRect(bar_rect, QColor(self._palette.bg_tertiary))

        if self._score > 0:
            # Color: red (0) → yellow (0.5) → green (1.0)
            if self._score <= 0.5:
                r, g = 255, int(255 * self._score * 2)
            else:
                r, g = int(255 * (1.0 - self._score) * 2), 255
            color = QColor(r, g, 50)
            fill_w = int(bar_rect.width() * self._score)
            painter.fillRect(bar_rect.adjusted(0, 0, -(bar_rect.width() - fill_w), 0), color)
        painter.end()
