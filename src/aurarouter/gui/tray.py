"""System tray icon for AuraRouter.

Provides :class:`AuraRouterTrayIcon`, a ``QSystemTrayIcon`` subclass with
a context menu (Show/Restore, Exit) and double-click-to-restore behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from aurarouter.gui.theme import DARK_PALETTE

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow


def _generate_tray_icon() -> QIcon:
    """Create a simple programmatic tray icon using the accent colour."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Outer circle — accent colour
    painter.setBrush(QColor(DARK_PALETTE.accent))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, size - 8, size - 8)

    # Inner "A" letter
    painter.setPen(QColor("#ffffff"))
    font = painter.font()
    font.setPixelSize(36)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "A")

    painter.end()
    return QIcon(pixmap)


class AuraRouterTrayIcon(QSystemTrayIcon):
    """System tray icon with Show/Restore and Exit actions."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(_generate_tray_icon(), parent=window)
        self._window = window
        self.setToolTip("AuraRouter \u2014 Running")

        # Context menu
        menu = QMenu()
        restore_action = QAction("Show / Restore", menu)
        restore_action.triggered.connect(self._restore_window)
        menu.addAction(restore_action)
        menu.addSeparator()
        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self._exit_app)
        menu.addAction(exit_action)
        self.setContextMenu(menu)

        # Double-click restores the window
        self.activated.connect(self._on_activated)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._restore_window()

    def _restore_window(self) -> None:
        self._window.showNormal()
        self._window.activateWindow()

    def _exit_app(self) -> None:
        from PySide6.QtWidgets import QApplication

        self.hide()
        QApplication.quit()
