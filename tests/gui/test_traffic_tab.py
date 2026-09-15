"""Tests for the Token Traffic Monitor tab."""

import sys
import time

import pytest

from aurarouter.savings.models import UsageRecord
from aurarouter.savings.pricing import CostEngine, PricingCatalog
from aurarouter.savings.usage_store import UsageStore


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_record(**overrides) -> UsageRecord:
    defaults = dict(
        timestamp="2025-01-15T10:00:00Z",
        model_id="test-model",
        provider="ollama",
        role="router",
        intent="SIMPLE_CODE",
        input_tokens=100,
        output_tokens=200,
        elapsed_s=1.0,
        success=True,
        is_cloud=False,
    )
    defaults.update(overrides)
    return UsageRecord(**defaults)


def _make_sources(tmp_path):
    """Return (UsageStore, CostEngine) backed by a temp database."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    engine = CostEngine(PricingCatalog(), store)
    return store, engine


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
    """TokenTrafficTab instantiates without error."""
    from aurarouter.gui.traffic_tab import TokenTrafficTab

    tab = TokenTrafficTab()
    assert tab is not None


def test_time_range_options():
    """Combo box contains the expected time range options."""
    from aurarouter.gui.traffic_tab import TokenTrafficTab

    tab = TokenTrafficTab()
    options = [tab._range_combo.itemText(i) for i in range(tab._range_combo.count())]
    assert options == ["Last Hour", "Today", "This Week", "This Month", "All Time"]


def test_refresh_with_empty_store(tmp_path):
    """Refresh with an empty store shows zero values."""
    from aurarouter.gui.traffic_tab import TokenTrafficTab

    store, engine = _make_sources(tmp_path)
    tab = TokenTrafficTab()
    tab.set_data_sources(store, engine)
    tab.refresh()
    _wait_for_refresh(tab)

    assert tab._lbl_requests.text() == "0"
    assert tab._lbl_input.text() == "0"
    assert tab._lbl_output.text() == "0"
    assert tab._lbl_cost.text() == "$0.00"
    assert tab._provider_table.rowCount() == 0
    assert tab._intent_table.rowCount() == 0
    assert tab._role_table.rowCount() == 0


def test_refresh_with_data(tmp_path):
    """Pre-populated store produces matching summary values."""
    from aurarouter.gui.traffic_tab import TokenTrafficTab

    store, engine = _make_sources(tmp_path)
    store.record(_make_record(
        model_id="m1", provider="ollama", role="router",
        intent="SIMPLE_CODE", input_tokens=1000, output_tokens=2000,
    ))
    store.record(_make_record(
        model_id="m2", provider="google", role="coding",
        intent="COMPLEX_REASONING", input_tokens=3000, output_tokens=4000,
    ))

    tab = TokenTrafficTab()
    # Select "All Time" so date filtering doesn't exclude records.
    tab._range_combo.setCurrentText("All Time")
    tab.set_data_sources(store, engine)
    tab.refresh()
    _wait_for_refresh(tab)

    assert tab._lbl_requests.text() == "2"
    assert tab._lbl_input.text() == "4,000"
    assert tab._lbl_output.text() == "6,000"
    assert tab._provider_table.rowCount() == 2
    assert tab._intent_table.rowCount() == 2
    assert tab._role_table.rowCount() == 2
