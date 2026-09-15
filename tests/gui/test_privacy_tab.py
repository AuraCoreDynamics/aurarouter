"""Tests for the Privacy Audit Dashboard tab."""

import sys
import time

import pytest

from aurarouter.savings.privacy import (
    PrivacyEvent,
    PrivacyMatch,
    PrivacyStore,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_event(**overrides) -> PrivacyEvent:
    defaults = dict(
        timestamp="2025-06-15T10:00:00Z",
        model_id="gemini-2.0-flash",
        provider="google",
        matches=[
            PrivacyMatch(
                pattern_name="Email Address",
                severity="medium",
                matched_text="user***",
                position=10,
            )
        ],
        prompt_length=50,
        recommendation="Consider routing to a local model",
    )
    defaults.update(overrides)
    return PrivacyEvent(**defaults)


def _wait_for_refresh(tab, timeout_s: float = 5.0) -> None:
    """Spin the event loop until the worker thread completes."""
    from PySide6.QtCore import QCoreApplication

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if tab._thread is None:
            return
        time.sleep(0.01)


# Ensure a QApplication exists for widget tests.
_app = None


@pytest.fixture(autouse=True)
def _ensure_qapp():
    global _app
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        _app = QApplication(sys.argv)
    yield


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_tab_creation():
    """PrivacyAuditTab instantiates without error."""
    from aurarouter.gui.privacy_tab import PrivacyAuditTab

    tab = PrivacyAuditTab()
    assert tab is not None


def test_severity_filter_options():
    """Combo box contains All, High, Medium, Low."""
    from aurarouter.gui.privacy_tab import PrivacyAuditTab

    tab = PrivacyAuditTab()
    options = [
        tab._severity_combo.itemText(i)
        for i in range(tab._severity_combo.count())
    ]
    assert options == ["All", "High", "Medium", "Low"]


def test_refresh_with_empty_store(tmp_path):
    """All counts show 0 with an empty store."""
    from aurarouter.gui.privacy_tab import PrivacyAuditTab

    store = PrivacyStore(db_path=tmp_path / "usage.db")
    tab = PrivacyAuditTab()
    tab.set_data_source(store)
    tab.refresh()
    _wait_for_refresh(tab)

    assert tab._lbl_total.text() == "0"
    assert tab._lbl_high.text() == "0"
    assert tab._lbl_medium.text() == "0"
    assert tab._lbl_low.text() == "0"
    assert tab._pattern_table.rowCount() == 0
    assert tab._event_table.rowCount() == 0


def test_refresh_with_events(tmp_path):
    """Pre-populate store, verify summary counts match."""
    from aurarouter.gui.privacy_tab import PrivacyAuditTab

    store = PrivacyStore(db_path=tmp_path / "usage.db")

    # Event 1: medium severity (email).
    store.record(_make_event(
        timestamp="2025-06-15T10:00:00Z",
        matches=[PrivacyMatch("Email Address", "medium", "user***", 10)],
    ))

    # Event 2: high severity (API key + SSN).
    store.record(_make_event(
        timestamp="2025-06-15T11:00:00Z",
        matches=[
            PrivacyMatch("API Key", "high", "api_***", 5),
            PrivacyMatch("SSN", "high", "123-***", 30),
        ],
    ))

    # Event 3: low severity (private IP).
    store.record(_make_event(
        timestamp="2025-06-15T12:00:00Z",
        matches=[PrivacyMatch("Private IP Address", "low", "192.***", 0)],
    ))

    tab = PrivacyAuditTab()
    # Select "All Time" so date filtering doesn't exclude records.
    tab._range_combo.setCurrentText("All Time")
    tab.set_data_source(store)
    tab.refresh()
    _wait_for_refresh(tab)

    assert tab._lbl_total.text() == "3"
    assert tab._lbl_high.text() == "1"
    assert tab._lbl_medium.text() == "1"
    assert tab._lbl_low.text() == "1"
    assert tab._pattern_table.rowCount() == 4  # Email, API Key, SSN, Private IP
    assert tab._event_table.rowCount() == 3


def test_badge_appears_for_high_severity(tmp_path):
    """High-severity event row contains badge text."""
    from aurarouter.gui.privacy_tab import PrivacyAuditTab

    store = PrivacyStore(db_path=tmp_path / "usage.db")

    # High severity event.
    store.record(_make_event(
        timestamp="2025-06-15T10:00:00Z",
        matches=[PrivacyMatch("SSN", "high", "123-***", 0)],
    ))

    tab = PrivacyAuditTab()
    tab._range_combo.setCurrentText("All Time")
    tab.set_data_source(store)
    tab.refresh()
    _wait_for_refresh(tab)

    assert tab._event_table.rowCount() == 1
    badge_widget = tab._event_table.cellWidget(0, 5)
    assert badge_widget is not None
    assert "Consider Local Routing" in badge_widget.text()
