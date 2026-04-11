"""Token pressure gauge widget for AuraRouter GUI."""
from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Typography


class TokenPressureGauge(QWidget):
    """Horizontal bar showing token usage vs context limit.

    Zones: 0-60% green (safe), 60-80% yellow (caution),
           80-95% red (pressure), >95% critical (pulsing).
    """

    condense_requested = Signal()
    critical_pressure_alert = Signal(str)  # session_id

    def __init__(self, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._ratio: float = 0.0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._context_limit: int = 0
        self._critical: bool = False
        self._pulse_opacity: float = 1.0
        self._session_id: str = ""

        self.setFixedHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel("— tokens", self)
        self._label.setStyleSheet(
            f"font-size: {Typography.size_small}pt; color: {self._palette.text_secondary};"
        )
        layout.addWidget(self._label)
        layout.addStretch()

        self._warn_label = QLabel("", self)
        self._warn_label.setStyleSheet(
            f"font-size: {Typography.size_small}pt; color: {self._palette.error};"
        )
        self._warn_label.setCursor(Qt.PointingHandCursor)
        self._warn_label.hide()
        self._warn_label.mousePressEvent = lambda _: self.condense_requested.emit()
        layout.addWidget(self._warn_label)

        self._animation = QPropertyAnimation(self, b"pulse_opacity")
        self._animation.setDuration(800)
        self._animation.setStartValue(0.6)
        self._animation.setEndValue(1.0)
        self._animation.setLoopCount(-1)  # infinite

    # ------------------------------------------------------------------
    # Qt property for animation
    # ------------------------------------------------------------------

    def _get_pulse_opacity(self) -> float:
        return self._pulse_opacity

    def _set_pulse_opacity(self, value: float) -> None:
        self._pulse_opacity = value
        self.update()

    pulse_opacity = Property(float, _get_pulse_opacity, _set_pulse_opacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_stats(self, input_tokens: int, output_tokens: int, context_limit: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._context_limit = context_limit
        total = input_tokens + output_tokens
        ratio = total / context_limit if context_limit > 0 else 0.0
        self.set_pressure(min(ratio, 1.0))
        label = f"{total / 1000:.1f}K / {context_limit / 1000:.0f}K tokens ({ratio * 100:.0f}%)"
        self._label.setText(label)

    def set_pressure(self, ratio: float) -> None:
        self._ratio = max(0.0, min(1.0, ratio))
        was_critical = self._critical
        self._critical = self._ratio > 0.95
        if self._critical and not was_critical:
            self._animation.start()
            self._warn_label.setText("⚠ Critical — Click to Condense Now")
            self._warn_label.show()
            self.critical_pressure_alert.emit(self._session_id)
        elif not self._critical and was_critical:
            self._animation.stop()
            self._pulse_opacity = 1.0
            self._warn_label.hide()
        self.update()

    def set_critical(self, active: bool) -> None:
        self.set_pressure(0.96 if active else 0.0)

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bar_rect = self.rect().adjusted(0, 28, 0, -4)

        # Background
        painter.fillRect(bar_rect, QColor(self._palette.bg_tertiary))

        if self._ratio <= 0:
            return

        # Fill color by zone
        if self._ratio <= 0.60:
            color = QColor(self._palette.success)
        elif self._ratio <= 0.80:
            color = QColor(self._palette.warning)
        else:
            color = QColor(self._palette.error)

        if self._critical:
            color.setAlphaF(self._pulse_opacity)

        fill_width = int(bar_rect.width() * self._ratio)
        fill_rect = bar_rect.adjusted(0, 0, -(bar_rect.width() - fill_width), 0)
        painter.fillRect(fill_rect, color)
        painter.end()
