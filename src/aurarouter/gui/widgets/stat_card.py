"""Stat card widget — a compact summary tile with title, value, and optional accent."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
    get_palette,
)


class StatCard(QFrame):
    """A compact stat card showing a title, large value, and optional subtitle.

    Parameters
    ----------
    title:
        Small secondary header text.
    value:
        Large display value (e.g. ``"1,234"``).
    subtitle:
        Optional smaller text below the value.
    accent_color:
        Optional palette colour key or colour string for a left-edge stripe.
        When provided the stripe uses ``accent_color`` directly if it starts
        with ``#``, otherwise it is looked up on the palette.
    palette:
        The :class:`ColorPalette` to use.  Defaults to the dark palette.
    parent:
        Optional parent widget.
    """

    def __init__(
        self,
        title: str = "",
        value: str = "",
        subtitle: str = "",
        accent_color: Optional[str] = None,
        palette: Optional[ColorPalette] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._apply_frame_style(accent_color)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        layout.setSpacing(SPACING.xs)

        # Title
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px; "
            f"font-weight: bold; "
            f"background: transparent;"
        )
        layout.addWidget(self._title_label)

        # Value
        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(
            f"color: {self._palette.text_primary}; "
            f"font-size: {TYPOGRAPHY.size_h1}px; "
            f"font-weight: bold; "
            f"background: transparent;"
        )
        layout.addWidget(self._value_label)

        # Subtitle
        self._subtitle_label = QLabel(subtitle)
        self._subtitle_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px; "
            f"background: transparent;"
        )
        self._subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self._subtitle_label)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title_label.setText(text)

    def set_value(self, text: str) -> None:
        self._value_label.setText(text)

    def set_subtitle(self, text: str) -> None:
        self._subtitle_label.setText(text)
        self._subtitle_label.setVisible(bool(text))

    def title(self) -> str:
        return self._title_label.text()

    def value(self) -> str:
        return self._value_label.text()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_frame_style(self, accent_color: Optional[str]) -> None:
        p = self._palette
        border_left = ""
        if accent_color:
            colour = accent_color if accent_color.startswith("#") else getattr(p, accent_color, p.accent)
            border_left = f"border-left: 3px solid {colour};"

        self.setStyleSheet(
            f"StatCard {{"
            f"  background-color: {p.bg_secondary};"
            f"  border: 1px solid {p.border};"
            f"  border-radius: {RADIUS.md}px;"
            f"  {border_left}"
            f"}}"
        )
