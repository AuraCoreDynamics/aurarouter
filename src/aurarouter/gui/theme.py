"""Design system tokens and stylesheet generator for AuraRouter GUI.

Provides frozen dataclasses for colours, typography, and spacing, plus
:func:`apply_theme` which generates and installs a comprehensive QSS
stylesheet on the running ``QApplication``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


# ======================================================================
# Colour palette
# ======================================================================

@dataclass(frozen=True)
class ColorPalette:
    """Semantic colour tokens for the AuraRouter design system."""

    # Surface
    bg_primary: str = "#1e1e2e"
    bg_secondary: str = "#181825"
    bg_tertiary: str = "#313244"
    bg_hover: str = "#45475a"
    bg_selected: str = "#585b70"

    # Text
    text_primary: str = "#cdd6f4"
    text_secondary: str = "#a6adc8"
    text_disabled: str = "#6c7086"
    text_inverse: str = "#1e1e2e"

    # Semantic
    accent: str = "#89b4fa"
    success: str = "#a6e3a1"
    warning: str = "#f9e2af"
    error: str = "#f38ba8"
    info: str = "#89dceb"

    # Status
    status_pending: str = "#6c7086"
    status_running: str = "#89dceb"
    status_success: str = "#a6e3a1"
    status_failed: str = "#f38ba8"
    status_skipped: str = "#585b70"

    # Tiers
    tier_local: str = "#a6e3a1"
    tier_cloud: str = "#cba6f7"
    tier_grid: str = "#89dceb"

    # Chrome
    border: str = "#45475a"
    separator: str = "#313244"


DARK_PALETTE = ColorPalette()

LIGHT_PALETTE = ColorPalette(
    # Surface
    bg_primary="#eff1f5",
    bg_secondary="#e6e9ef",
    bg_tertiary="#ccd0da",
    bg_hover="#bcc0cc",
    bg_selected="#acb0be",
    # Text
    text_primary="#4c4f69",
    text_secondary="#5c5f77",
    text_disabled="#9ca0b0",
    text_inverse="#eff1f5",
    # Semantic
    accent="#1e66f5",
    success="#40a02b",
    warning="#df8e1d",
    error="#d20f39",
    info="#04a5e5",
    # Status
    status_pending="#9ca0b0",
    status_running="#04a5e5",
    status_success="#40a02b",
    status_failed="#d20f39",
    status_skipped="#acb0be",
    # Tiers
    tier_local="#40a02b",
    tier_cloud="#8839ef",
    tier_grid="#04a5e5",
    # Chrome
    border="#bcc0cc",
    separator="#ccd0da",
)


# ======================================================================
# Typography
# ======================================================================

@dataclass(frozen=True)
class Typography:
    """Font family and size tokens."""

    family_ui: str = "Segoe UI"
    family_mono: str = "Cascadia Code"
    family_mono_fallback: str = "Consolas"

    size_h1: int = 18
    size_h2: int = 14
    size_body: int = 11
    size_small: int = 9
    size_mono: int = 10


TYPOGRAPHY = Typography()


# ======================================================================
# Spacing & Radius
# ======================================================================

@dataclass(frozen=True)
class Spacing:
    """Spacing tokens in pixels."""

    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32


SPACING = Spacing()


@dataclass(frozen=True)
class Radius:
    """Border-radius tokens in pixels."""

    sm: int = 4
    md: int = 6
    lg: int = 8


RADIUS = Radius()


# ======================================================================
# Palette accessor by mode name
# ======================================================================

_PALETTES: dict[str, ColorPalette] = {
    "dark": DARK_PALETTE,
    "light": LIGHT_PALETTE,
}


def get_palette(mode: str = "dark") -> ColorPalette:
    """Return the palette for the given mode name."""
    return _PALETTES.get(mode, DARK_PALETTE)


# ======================================================================
# Stylesheet generator
# ======================================================================

def _generate_stylesheet(palette: ColorPalette) -> str:
    """Generate a comprehensive QSS string from *palette*."""
    p = palette
    t = TYPOGRAPHY
    s = SPACING
    r = RADIUS

    mono = f"'{t.family_mono}', '{t.family_mono_fallback}'"

    return f"""
/* ---- Base ---- */
QMainWindow, QWidget {{
    background-color: {p.bg_primary};
    color: {p.text_primary};
    font-family: '{t.family_ui}';
    font-size: {t.size_body}px;
}}

/* ---- QPushButton (default) ---- */
QPushButton {{
    background-color: {p.bg_tertiary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.lg}px;
    font-size: {t.size_body}px;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: {p.bg_hover};
}}
QPushButton:pressed {{
    background-color: {p.bg_selected};
}}
QPushButton:disabled {{
    color: {p.text_disabled};
    background-color: {p.bg_secondary};
    border-color: {p.separator};
}}

/* ---- QPushButton[objectName="primary"] ---- */
QPushButton#primary {{
    background-color: {p.accent};
    color: {p.text_inverse};
    border: 1px solid {p.accent};
    font-weight: bold;
}}
QPushButton#primary:hover {{
    background-color: {p.accent};
    opacity: 0.9;
}}
QPushButton#primary:disabled {{
    background-color: {p.bg_tertiary};
    color: {p.text_disabled};
    border-color: {p.border};
}}

/* ---- QPushButton[objectName="danger"] ---- */
QPushButton#danger {{
    background-color: {p.error};
    color: {p.text_inverse};
    border: 1px solid {p.error};
    font-weight: bold;
}}
QPushButton#danger:hover {{
    background-color: {p.error};
    opacity: 0.9;
}}

/* ---- QLineEdit ---- */
QLineEdit {{
    background-color: {p.bg_secondary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    padding: {s.sm}px;
    font-size: {t.size_body}px;
    selection-background-color: {p.accent};
    selection-color: {p.text_inverse};
}}
QLineEdit:focus {{
    border-color: {p.accent};
}}
QLineEdit:disabled {{
    color: {p.text_disabled};
    background-color: {p.bg_primary};
}}

/* ---- QTextEdit ---- */
QTextEdit {{
    background-color: {p.bg_secondary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    padding: {s.sm}px;
    font-family: {mono};
    font-size: {t.size_mono}px;
    selection-background-color: {p.accent};
    selection-color: {p.text_inverse};
}}
QTextEdit:focus {{
    border-color: {p.accent};
}}

/* ---- QComboBox ---- */
QComboBox {{
    background-color: {p.bg_secondary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    padding: {s.xs}px {s.sm}px;
    min-height: 22px;
}}
QComboBox:hover {{
    border-color: {p.accent};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {p.bg_secondary};
    color: {p.text_primary};
    selection-background-color: {p.bg_hover};
    selection-color: {p.text_primary};
    border: 1px solid {p.border};
}}

/* ---- QTableWidget ---- */
QTableWidget {{
    background-color: {p.bg_secondary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    gridline-color: {p.separator};
    selection-background-color: {p.bg_selected};
    selection-color: {p.text_primary};
}}
QTableWidget::item {{
    padding: {s.xs}px {s.sm}px;
}}
QTableWidget::item:selected {{
    background-color: {p.bg_selected};
}}

/* ---- QHeaderView ---- */
QHeaderView::section {{
    background-color: {p.bg_tertiary};
    color: {p.text_secondary};
    border: none;
    border-right: 1px solid {p.separator};
    border-bottom: 1px solid {p.separator};
    padding: {s.xs}px {s.sm}px;
    font-weight: bold;
    font-size: {t.size_small}px;
}}

/* ---- QSplitter ---- */
QSplitter::handle {{
    background-color: {p.separator};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ---- QScrollArea / QScrollBar ---- */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background-color: {p.bg_secondary};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background-color: {p.bg_hover};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {p.bg_selected};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background-color: {p.bg_secondary};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background-color: {p.bg_hover};
    border-radius: 5px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {p.bg_selected};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ---- QToolTip ---- */
QToolTip {{
    background-color: {p.bg_tertiary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    padding: {s.sm}px;
    font-size: {t.size_small}px;
}}

/* ---- QMenu ---- */
QMenu {{
    background-color: {p.bg_secondary};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    padding: {s.xs}px 0px;
}}
QMenu::item {{
    padding: {s.sm}px {s.lg}px;
}}
QMenu::item:selected {{
    background-color: {p.bg_hover};
}}
QMenu::separator {{
    height: 1px;
    background-color: {p.separator};
    margin: {s.xs}px {s.sm}px;
}}

/* ---- QCheckBox ---- */
QCheckBox {{
    color: {p.text_primary};
    spacing: {s.sm}px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    background-color: {p.bg_secondary};
}}
QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}
QCheckBox::indicator:hover {{
    border-color: {p.accent};
}}

/* ---- QProgressBar ---- */
QProgressBar {{
    background-color: {p.bg_tertiary};
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    text-align: center;
    color: {p.text_primary};
    font-size: {t.size_small}px;
    min-height: 14px;
}}
QProgressBar::chunk {{
    background-color: {p.accent};
    border-radius: {r.sm}px;
}}

/* ---- QLabel ---- */
QLabel {{
    background-color: transparent;
    color: {p.text_primary};
}}

/* ---- QGroupBox ---- */
QGroupBox {{
    border: 1px solid {p.border};
    border-radius: {r.md}px;
    margin-top: {s.lg}px;
    padding-top: {s.xl}px;
    font-weight: bold;
    color: {p.text_secondary};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 {s.sm}px;
    color: {p.text_secondary};
}}

/* ---- QTabWidget ---- */
QTabWidget::pane {{
    border: 1px solid {p.border};
    border-radius: {r.sm}px;
    background-color: {p.bg_primary};
}}
QTabBar::tab {{
    background-color: {p.bg_secondary};
    color: {p.text_secondary};
    border: 1px solid {p.border};
    border-bottom: none;
    padding: {s.sm}px {s.lg}px;
    margin-right: 2px;
    border-top-left-radius: {r.sm}px;
    border-top-right-radius: {r.sm}px;
}}
QTabBar::tab:selected {{
    background-color: {p.bg_primary};
    color: {p.text_primary};
    border-bottom: 2px solid {p.accent};
}}
QTabBar::tab:hover:!selected {{
    background-color: {p.bg_tertiary};
}}

/* ---- QStatusBar ---- */
QStatusBar {{
    background-color: {p.bg_secondary};
    color: {p.text_secondary};
    border-top: 1px solid {p.separator};
    font-size: {t.size_small}px;
}}
"""


# ======================================================================
# Public API
# ======================================================================

def apply_theme(app: "QApplication", mode: str = "dark") -> None:
    """Apply the design-system stylesheet to *app*.

    Parameters
    ----------
    app:
        The running ``QApplication`` instance.
    mode:
        ``"dark"`` (default) or ``"light"``.
    """
    palette = get_palette(mode)
    stylesheet = _generate_stylesheet(palette)
    app.setStyleSheet(stylesheet)
