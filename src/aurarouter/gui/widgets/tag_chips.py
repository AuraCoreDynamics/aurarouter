"""Tag chips widget — horizontal flow of coloured pills."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
)


class _Chip(QFrame):
    """A single coloured pill."""

    removed = Signal(str)

    def __init__(
        self,
        text: str,
        color: str,
        editable: bool,
        palette: ColorPalette,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._palette = palette

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        layout.setSpacing(SPACING.xs)

        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {palette.text_primary}; "
            f"font-size: {TYPOGRAPHY.size_small}px; "
            f"background: transparent;"
        )
        layout.addWidget(lbl)

        if editable:
            close_btn = QPushButton("\u00d7")
            close_btn.setFixedSize(16, 16)
            close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            close_btn.setStyleSheet(
                f"QPushButton {{"
                f"  background: transparent;"
                f"  color: {palette.text_secondary};"
                f"  border: none;"
                f"  font-size: {TYPOGRAPHY.size_body}px;"
                f"  font-weight: bold;"
                f"}}"
                f"QPushButton:hover {{"
                f"  color: {palette.error};"
                f"}}"
            )
            close_btn.clicked.connect(lambda: self.removed.emit(self._text))
            layout.addWidget(close_btn)

        self.setStyleSheet(
            f"_Chip {{"
            f"  background-color: {color};"
            f"  border-radius: {RADIUS.lg}px;"
            f"  border: none;"
            f"}}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    @property
    def tag_text(self) -> str:
        return self._text


class TagChips(QWidget):
    """Horizontal flow of coloured tag pills.

    Signals
    -------
    tag_removed(str)
        Emitted when a tag is removed in editable mode.
    """

    tag_removed = Signal(str)

    def __init__(
        self,
        editable: bool = False,
        palette: Optional[ColorPalette] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._editable = editable
        self._chips: list[_Chip] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(SPACING.sm)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tag(self, text: str, color: Optional[str] = None) -> None:
        """Add a tag chip.

        Parameters
        ----------
        text:
            Label for the chip.
        color:
            Background colour.  Falls back to a muted version of the
            palette's ``bg_tertiary``.
        """
        bg = color or self._palette.bg_tertiary
        chip = _Chip(text, bg, self._editable, self._palette, self)
        chip.removed.connect(self._on_chip_removed)
        self._chips.append(chip)
        self._layout.addWidget(chip)

    def remove_tag(self, text: str) -> None:
        """Remove a tag by its label text."""
        for chip in self._chips:
            if chip.tag_text == text:
                self._chips.remove(chip)
                self._layout.removeWidget(chip)
                chip.deleteLater()
                break

    def clear_tags(self) -> None:
        """Remove all tags."""
        for chip in self._chips:
            self._layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()

    def tags(self) -> list[str]:
        """Return the current list of tag labels."""
        return [c.tag_text for c in self._chips]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_chip_removed(self, text: str) -> None:
        self.remove_tag(text)
        self.tag_removed.emit(text)
