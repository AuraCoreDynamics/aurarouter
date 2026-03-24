"""Help tooltip widget — small [?] button showing rich tooltip on hover/click."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QToolTip, QWidget

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
)


class HelpTooltip(QPushButton):
    """Small ``[?]`` button that shows a rich tooltip on hover or click.

    Parameters
    ----------
    help_text:
        The text (may contain basic HTML) shown in the tooltip.
    palette:
        Colour palette to use for styling.
    """

    def __init__(
        self,
        help_text: str = "",
        palette: Optional[ColorPalette] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("?", parent)
        self._palette = palette or DARK_PALETTE
        self._help_text = help_text

        self.setFixedSize(20, 20)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setToolTip(help_text)

        self._apply_style()
        self.clicked.connect(self._show_tooltip)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_help_text(self, text: str) -> None:
        """Update the tooltip text."""
        self._help_text = text
        self.setToolTip(text)

    def help_text(self) -> str:
        return self._help_text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _show_tooltip(self) -> None:
        """Show the tooltip at the button's position on click."""
        pos = self.mapToGlobal(self.rect().bottomLeft())
        QToolTip.showText(pos, self._help_text, self)

    def _apply_style(self) -> None:
        p = self._palette
        self.setStyleSheet(
            f"HelpTooltip {{"
            f"  background-color: {p.bg_tertiary};"
            f"  color: {p.text_secondary};"
            f"  border: 1px solid {p.border};"
            f"  border-radius: {RADIUS.lg}px;"
            f"  font-size: {TYPOGRAPHY.size_small}px;"
            f"  font-weight: bold;"
            f"}}"
            f"HelpTooltip:hover {{"
            f"  background-color: {p.bg_hover};"
            f"  color: {p.accent};"
            f"}}"
        )
