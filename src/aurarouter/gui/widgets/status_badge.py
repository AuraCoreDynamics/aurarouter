"""Status badge widget — rounded pill with icon and text, auto-coloured by mode."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
)

# Preset modes → (icon, palette attribute for background, palette attribute for text)
_PRESETS: dict[str, tuple[str, str, str]] = {
    "running":   ("\u25b6", "status_running", "text_inverse"),
    "stopped":   ("\u25a0", "status_pending", "text_primary"),
    "error":     ("\u2716", "status_failed", "text_inverse"),
    "healthy":   ("\u2714", "status_success", "text_inverse"),
    "unhealthy": ("\u26a0", "error", "text_inverse"),
    "paused":    ("\u23f8", "warning", "text_inverse"),
    "loading":   ("\u21bb", "info", "text_inverse"),
}


class StatusBadge(QLabel):
    """Rounded pill badge showing an icon and status text.

    Parameters
    ----------
    mode:
        One of ``"running"``, ``"stopped"``, ``"error"``, ``"healthy"``,
        ``"unhealthy"``, ``"paused"``, ``"loading"``.
    text:
        Override display text (otherwise uses the mode name capitalised).
    palette:
        Colour palette to pull colours from.
    """

    def __init__(
        self,
        mode: str = "stopped",
        text: Optional[str] = None,
        palette: Optional[ColorPalette] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_mode(mode, text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str, text: Optional[str] = None) -> None:
        """Change the badge mode and optionally the display text."""
        self._mode = mode
        preset = _PRESETS.get(mode, ("\u2022", "bg_tertiary", "text_primary"))
        icon, bg_attr, fg_attr = preset

        bg_color = getattr(self._palette, bg_attr, self._palette.bg_tertiary)
        fg_color = getattr(self._palette, fg_attr, self._palette.text_primary)

        display = text or mode.capitalize()
        self.setText(f" {icon} {display} ")
        self._apply_style(bg_color, fg_color)

    def get_mode(self) -> str:
        return self._mode

    def background_color(self) -> str:
        """Return the current background colour string (for testing)."""
        return self._bg_color

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_style(self, bg: str, fg: str) -> None:
        self._bg_color = bg
        self.setStyleSheet(
            f"StatusBadge {{"
            f"  background-color: {bg};"
            f"  color: {fg};"
            f"  border-radius: {RADIUS.lg}px;"
            f"  padding: {SPACING.xs}px {SPACING.md}px;"
            f"  font-size: {TYPOGRAPHY.size_small}px;"
            f"  font-weight: bold;"
            f"}}"
        )
