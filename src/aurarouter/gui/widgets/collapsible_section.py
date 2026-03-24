"""Collapsible section widget — header bar with animated expand/collapse."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
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


class CollapsibleSection(QWidget):
    """A section with a clickable header that expands/collapses its content.

    Usage::

        section = CollapsibleSection("Details")
        section.add_widget(my_content_widget)
    """

    def __init__(
        self,
        title: str = "",
        initially_expanded: bool = False,
        palette: Optional[ColorPalette] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._expanded = initially_expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Header bar ----
        self._header = QFrame()
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {self._palette.bg_tertiary};"
            f"  border: 1px solid {self._palette.border};"
            f"  border-radius: {RADIUS.sm}px;"
            f"}}"
        )
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self._arrow = QLabel("\u25bc" if initially_expanded else "\u25b6")
        self._arrow.setFixedWidth(16)
        self._arrow.setStyleSheet(
            f"color: {self._palette.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_body}px; "
            f"background: transparent; "
            f"border: none;"
        )
        header_layout.addWidget(self._arrow)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            f"color: {self._palette.text_primary}; "
            f"font-size: {TYPOGRAPHY.size_body}px; "
            f"font-weight: bold; "
            f"background: transparent; "
            f"border: none;"
        )
        header_layout.addWidget(self._title_label, 1)

        # Make the entire header clickable
        self._header.mousePressEvent = lambda _event: self.toggle()
        root.addWidget(self._header)

        # ---- Content area ----
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(
            SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm
        )
        self._content_layout.setSpacing(SPACING.sm)
        self._content.setVisible(initially_expanded)
        root.addWidget(self._content)

        # Animation on the content's maximumHeight
        self._anim = QPropertyAnimation(self._content, b"maximumHeight")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        if not initially_expanded:
            self._content.setMaximumHeight(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the collapsible content area."""
        self._content_layout.addWidget(widget)

    def set_content_layout(self, layout) -> None:
        """Replace the content layout entirely."""
        # Clear old layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        # Transfer widgets from new layout
        # (QWidget can only have one layout, so we move children)
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                self._content_layout.addWidget(item.widget())

    def toggle(self) -> None:
        """Toggle the expanded/collapsed state."""
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self) -> None:
        """Expand the content area."""
        if self._expanded:
            return
        self._expanded = True
        self._arrow.setText("\u25bc")
        self._content.setVisible(True)

        # Animate from 0 to sizeHint height
        target_height = self._content.sizeHint().height()
        if target_height < 20:
            target_height = 200  # fallback
        self._anim.stop()
        self._anim.setStartValue(0)
        self._anim.setEndValue(target_height)
        self._anim.start()

    def collapse(self) -> None:
        """Collapse the content area."""
        if not self._expanded:
            return
        self._expanded = False
        self._arrow.setText("\u25b6")

        self._anim.stop()
        self._anim.setStartValue(self._content.height())
        self._anim.setEndValue(0)
        self._anim.finished.connect(self._on_collapse_done)
        self._anim.start()

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def set_title(self, title: str) -> None:
        self._title_label.setText(title)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_collapse_done(self) -> None:
        self._anim.finished.disconnect(self._on_collapse_done)
        if not self._expanded:
            self._content.setVisible(False)
