"""Tests for the AuraRouter design system and reusable widget library.

Uses ``pytest-qt`` for widget testing.  Tests skip gracefully when
``pytestqt`` is not installed or no display is available.
"""

from __future__ import annotations

import pytest

pytestqt = pytest.importorskip("pytestqt")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel

from aurarouter.gui.theme import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
    apply_theme,
    get_palette,
)
from aurarouter.gui.widgets import (
    CollapsibleSection,
    HelpTooltip,
    SearchInput,
    SidebarNav,
    StatCard,
    StatusBadge,
    TagChips,
)


# ======================================================================
# Theme tests
# ======================================================================


class TestThemeModule:
    """Tests for theme.py dataclasses and stylesheet generator."""

    def test_dark_palette_is_frozen(self):
        with pytest.raises(AttributeError):
            DARK_PALETTE.bg_primary = "#000000"  # type: ignore[misc]

    def test_light_palette_differs(self):
        assert DARK_PALETTE.bg_primary != LIGHT_PALETTE.bg_primary

    def test_get_palette_dark(self):
        assert get_palette("dark") is DARK_PALETTE

    def test_get_palette_light(self):
        assert get_palette("light") is LIGHT_PALETTE

    def test_get_palette_unknown_falls_back(self):
        assert get_palette("neon") is DARK_PALETTE

    def test_typography_defaults(self):
        assert TYPOGRAPHY.family_ui == "Segoe UI"
        assert TYPOGRAPHY.size_h1 == 18
        assert TYPOGRAPHY.size_body == 11

    def test_spacing_values(self):
        assert SPACING.xs == 4
        assert SPACING.xxl == 32

    def test_radius_values(self):
        assert RADIUS.sm == 4
        assert RADIUS.lg == 8

    def test_apply_theme_dark(self, qtbot):
        app = QApplication.instance()
        apply_theme(app, "dark")
        ss = app.styleSheet()
        assert DARK_PALETTE.bg_primary in ss
        assert "QMainWindow" in ss

    def test_apply_theme_light(self, qtbot):
        app = QApplication.instance()
        apply_theme(app, "light")
        ss = app.styleSheet()
        assert LIGHT_PALETTE.bg_primary in ss

    def test_stylesheet_covers_key_widgets(self, qtbot):
        app = QApplication.instance()
        apply_theme(app, "dark")
        ss = app.styleSheet()
        for widget_type in [
            "QMainWindow", "QPushButton", "QLineEdit", "QTextEdit",
            "QComboBox", "QTableWidget", "QHeaderView", "QSplitter",
            "QScrollArea", "QToolTip", "QMenu", "QCheckBox",
            "QProgressBar", "QLabel", "QGroupBox",
        ]:
            assert widget_type in ss, f"{widget_type} missing from stylesheet"


# ======================================================================
# StatCard tests
# ======================================================================


class TestStatCard:
    def test_renders_title_and_value(self, qtbot):
        card = StatCard(title="Requests", value="42")
        qtbot.addWidget(card)
        assert card.title() == "Requests"
        assert card.value() == "42"

    def test_set_value(self, qtbot):
        card = StatCard(title="T", value="0")
        qtbot.addWidget(card)
        card.set_value("99")
        assert card.value() == "99"

    def test_subtitle_hidden_when_empty(self, qtbot):
        card = StatCard(title="T", value="0")
        qtbot.addWidget(card)
        assert not card._subtitle_label.isVisible()

    def test_subtitle_visible_when_set(self, qtbot):
        card = StatCard(title="T", value="0", subtitle="sub")
        qtbot.addWidget(card)
        assert card._subtitle_label.isVisible()

    def test_accent_stripe(self, qtbot):
        card = StatCard(title="T", value="0", accent_color="#ff0000")
        qtbot.addWidget(card)
        assert "border-left" in card.styleSheet()


# ======================================================================
# SidebarNav tests
# ======================================================================


class TestSidebarNav:
    def test_emits_current_changed(self, qtbot):
        nav = SidebarNav()
        qtbot.addWidget(nav)
        nav.add_item("home", "H", "Home")
        nav.add_item("settings", "S", "Settings")

        with qtbot.waitSignal(nav.current_changed, timeout=1000) as blocker:
            # Click the second button
            nav._buttons[1].click()

        assert blocker.args == ["settings"]

    def test_auto_selects_first(self, qtbot):
        nav = SidebarNav()
        qtbot.addWidget(nav)
        nav.add_item("home", "H", "Home")
        assert nav._current_key == "home"

    def test_set_current(self, qtbot):
        nav = SidebarNav()
        qtbot.addWidget(nav)
        nav.add_item("a", "A", "Alpha")
        nav.add_item("b", "B", "Beta")
        nav.set_current("b")
        assert nav._current_key == "b"

    def test_toggle_changes_width(self, qtbot):
        nav = SidebarNav()
        qtbot.addWidget(nav)
        assert nav.is_expanded
        nav.toggle()
        assert not nav.is_expanded


# ======================================================================
# SearchInput tests
# ======================================================================


class TestSearchInput:
    def test_debounced_signal(self, qtbot):
        search = SearchInput()
        qtbot.addWidget(search)
        search.set_debounce_ms(50)  # shorten for test

        with qtbot.waitSignal(search.search_changed, timeout=1000) as blocker:
            search.setText("hello")

        assert blocker.args == ["hello"]

    def test_clear_button_enabled(self, qtbot):
        search = SearchInput()
        qtbot.addWidget(search)
        assert search.isClearButtonEnabled()

    def test_placeholder(self, qtbot):
        search = SearchInput(placeholder="Find models...")
        qtbot.addWidget(search)
        assert "Find models" in search.placeholderText()


# ======================================================================
# StatusBadge tests
# ======================================================================


class TestStatusBadge:
    def test_running_mode_color(self, qtbot):
        badge = StatusBadge(mode="running")
        qtbot.addWidget(badge)
        assert badge.background_color() == DARK_PALETTE.status_running

    def test_error_mode_color(self, qtbot):
        badge = StatusBadge(mode="error")
        qtbot.addWidget(badge)
        assert badge.background_color() == DARK_PALETTE.status_failed

    def test_healthy_mode_color(self, qtbot):
        badge = StatusBadge(mode="healthy")
        qtbot.addWidget(badge)
        assert badge.background_color() == DARK_PALETTE.status_success

    def test_set_mode_changes_text(self, qtbot):
        badge = StatusBadge(mode="stopped")
        qtbot.addWidget(badge)
        badge.set_mode("running")
        assert "Running" in badge.text()

    def test_custom_text(self, qtbot):
        badge = StatusBadge(mode="loading", text="Initializing")
        qtbot.addWidget(badge)
        assert "Initializing" in badge.text()

    def test_light_palette(self, qtbot):
        badge = StatusBadge(mode="healthy", palette=LIGHT_PALETTE)
        qtbot.addWidget(badge)
        assert badge.background_color() == LIGHT_PALETTE.status_success


# ======================================================================
# CollapsibleSection tests
# ======================================================================


class TestCollapsibleSection:
    def test_initially_collapsed(self, qtbot):
        section = CollapsibleSection("Details")
        qtbot.addWidget(section)
        assert not section.is_expanded

    def test_initially_expanded(self, qtbot):
        section = CollapsibleSection("Details", initially_expanded=True)
        qtbot.addWidget(section)
        assert section.is_expanded

    def test_toggle(self, qtbot):
        section = CollapsibleSection("Details")
        qtbot.addWidget(section)
        assert not section.is_expanded
        section.toggle()
        assert section.is_expanded
        section.toggle()
        assert not section.is_expanded

    def test_add_widget(self, qtbot):
        section = CollapsibleSection("Details")
        qtbot.addWidget(section)
        lbl = QLabel("Content")
        section.add_widget(lbl)
        assert section._content_layout.count() == 1


# ======================================================================
# TagChips tests
# ======================================================================


class TestTagChips:
    def test_add_and_list(self, qtbot):
        chips = TagChips()
        qtbot.addWidget(chips)
        chips.add_tag("python")
        chips.add_tag("qt")
        assert chips.tags() == ["python", "qt"]

    def test_remove_tag(self, qtbot):
        chips = TagChips()
        qtbot.addWidget(chips)
        chips.add_tag("a")
        chips.add_tag("b")
        chips.remove_tag("a")
        assert chips.tags() == ["b"]

    def test_clear_tags(self, qtbot):
        chips = TagChips()
        qtbot.addWidget(chips)
        chips.add_tag("x")
        chips.clear_tags()
        assert chips.tags() == []

    def test_editable_emits_removed(self, qtbot):
        chips = TagChips(editable=True)
        qtbot.addWidget(chips)
        chips.add_tag("removeme")

        with qtbot.waitSignal(chips.tag_removed, timeout=1000) as blocker:
            # Simulate clicking the close button on the chip
            chips._chips[0].removed.emit("removeme")

        assert blocker.args == ["removeme"]


# ======================================================================
# HelpTooltip tests
# ======================================================================


class TestHelpTooltip:
    def test_text(self, qtbot):
        ht = HelpTooltip(help_text="This is help")
        qtbot.addWidget(ht)
        assert ht.help_text() == "This is help"
        assert ht.toolTip() == "This is help"

    def test_set_help_text(self, qtbot):
        ht = HelpTooltip()
        qtbot.addWidget(ht)
        ht.set_help_text("Updated")
        assert ht.help_text() == "Updated"
