"""Sidebar navigation widget — vertical icon+label buttons with collapse animation."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
)

_COLLAPSED_WIDTH = 48
_EXPANDED_WIDTH = 180


class _NavButton(QPushButton):
    """Single navigation entry with icon and label text."""

    def __init__(
        self,
        key: str,
        icon_text: str,
        label: str,
        palette: ColorPalette,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.key = key
        self._icon_text = icon_text
        self._label = label
        self._palette = palette
        self._selected = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)
        self._update_text(expanded=True)
        self._apply_style()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def set_expanded(self, expanded: bool) -> None:
        self._update_text(expanded)

    def _update_text(self, expanded: bool) -> None:
        if expanded:
            self.setText(f"  {self._icon_text}  {self._label}")
            self.setToolTip("")
        else:
            self.setText(self._icon_text)
            self.setToolTip(self._label)

    def _apply_style(self) -> None:
        p = self._palette
        if self._selected:
            bg = p.bg_selected
            colour = p.text_primary
            border_left = f"border-left: 3px solid {p.accent};"
        else:
            bg = "transparent"
            colour = p.text_secondary
            border_left = "border-left: 3px solid transparent;"

        self.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {bg};"
            f"  color: {colour};"
            f"  border: none;"
            f"  {border_left}"
            f"  border-radius: {RADIUS.sm}px;"
            f"  text-align: left;"
            f"  padding-left: {SPACING.sm}px;"
            f"  font-size: {TYPOGRAPHY.size_body}px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {p.bg_hover};"
            f"}}"
        )


class SidebarNav(QFrame):
    """Vertical sidebar navigation with collapsible icon-only / full modes.

    Signals
    -------
    current_changed(str)
        Emitted when the user selects a different navigation item.
    """

    current_changed = Signal(str)

    def __init__(
        self,
        palette: Optional[ColorPalette] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._buttons: list[_NavButton] = []
        self._current_key: Optional[str] = None
        self._expanded = True

        self.setFixedWidth(_EXPANDED_WIDTH)
        self.setStyleSheet(
            f"SidebarNav {{"
            f"  background-color: {self._palette.bg_secondary};"
            f"  border-right: 1px solid {self._palette.separator};"
            f"}}"
        )

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, SPACING.sm, 0, SPACING.sm)
        self._layout.setSpacing(SPACING.xs)

        # Toggle button at the top
        self._toggle_btn = QPushButton("\u2630")
        self._toggle_btn.setFixedSize(32, 32)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  color: {self._palette.text_secondary};"
            f"  border: none;"
            f"  font-size: {TYPOGRAPHY.size_h2}px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {self._palette.bg_hover};"
            f"  border-radius: {RADIUS.sm}px;"
            f"}}"
        )
        self._toggle_btn.clicked.connect(self.toggle)
        self._layout.addWidget(self._toggle_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._layout.addSpacing(SPACING.sm)

        self._items_layout = QVBoxLayout()
        self._items_layout.setSpacing(SPACING.xs)
        self._layout.addLayout(self._items_layout)
        self._layout.addStretch()

        # Animation
        self._anim = QPropertyAnimation(self, b"minimumWidth")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim_max = QPropertyAnimation(self, b"maximumWidth")
        self._anim_max.setDuration(200)
        self._anim_max.setEasingCurve(QEasingCurve.Type.InOutCubic)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_item(self, key: str, icon_text: str, label: str) -> None:
        """Add a navigation entry.

        Parameters
        ----------
        key:
            Unique identifier emitted in ``current_changed``.
        icon_text:
            Single character or emoji used as the icon.
        label:
            Human-readable label shown when expanded.
        """
        btn = _NavButton(key, icon_text, label, self._palette, self)
        btn.set_expanded(self._expanded)
        btn.clicked.connect(lambda _checked=False, k=key: self._select(k))
        self._buttons.append(btn)
        self._items_layout.addWidget(btn)

        # Auto-select the first item.
        if self._current_key is None:
            self._select(key)

    def set_current(self, key: str) -> None:
        """Programmatically select a navigation item."""
        self._select(key, emit=False)

    def toggle(self) -> None:
        """Toggle between expanded and collapsed modes."""
        if self._expanded:
            self._animate_to(_COLLAPSED_WIDTH)
        else:
            self._animate_to(_EXPANDED_WIDTH)
        self._expanded = not self._expanded
        for btn in self._buttons:
            btn.set_expanded(self._expanded)

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select(self, key: str, emit: bool = True) -> None:
        if key == self._current_key:
            return
        self._current_key = key
        for btn in self._buttons:
            btn.set_selected(btn.key == key)
        if emit:
            self.current_changed.emit(key)

    def _animate_to(self, width: int) -> None:
        self._anim.stop()
        self._anim_max.stop()

        self._anim.setStartValue(self.width())
        self._anim.setEndValue(width)
        self._anim_max.setStartValue(self.width())
        self._anim_max.setEndValue(width)

        self._anim.start()
        self._anim_max.start()
