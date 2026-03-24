"""Search input widget — line edit with icon, clear button, and debounced signal."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLineEdit, QWidget

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
)

_DEBOUNCE_MS = 300


class SearchInput(QLineEdit):
    """Search input with a magnifying-glass prefix, clear button, and debounce.

    Signals
    -------
    search_changed(str)
        Emitted *after* the debounce interval with the current text.
    """

    search_changed = Signal(str)

    def __init__(
        self,
        placeholder: str = "Search...",
        palette: Optional[ColorPalette] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE

        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)

        self._apply_style()

        # Debounce timer
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_search)

        self.textChanged.connect(self._on_text_changed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_debounce_ms(self, ms: int) -> None:
        """Change the debounce interval."""
        self._debounce.setInterval(ms)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_text_changed(self, _text: str) -> None:
        self._debounce.start()

    def _emit_search(self) -> None:
        self.search_changed.emit(self.text())

    def _apply_style(self) -> None:
        p = self._palette
        # Use a unicode magnifying glass as a visual cue in the placeholder;
        # actual icon rendering is done via padding to leave room.
        self.setStyleSheet(
            f"SearchInput {{"
            f"  background-color: {p.bg_secondary};"
            f"  color: {p.text_primary};"
            f"  border: 1px solid {p.border};"
            f"  border-radius: {RADIUS.md}px;"
            f"  padding: {SPACING.sm}px {SPACING.sm}px {SPACING.sm}px {SPACING.xl}px;"
            f"  font-size: {TYPOGRAPHY.size_body}px;"
            f"}}"
            f"SearchInput:focus {{"
            f"  border-color: {p.accent};"
            f"}}"
        )
        # Prepend a search icon to placeholder
        if not self.placeholderText().startswith("\U0001f50d"):
            self.setPlaceholderText(f"\U0001f50d {self.placeholderText()}")
