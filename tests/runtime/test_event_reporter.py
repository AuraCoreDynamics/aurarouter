"""Tests for EventReporter and its integration with ComputeFabric."""
import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.event_reporter import EventReporter
from aurarouter.fabric import ComputeFabric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fabric(**kwargs) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}}
    cfg.config.update(kwargs)
    return ComputeFabric(cfg)


# ---------------------------------------------------------------------------
# EventReporter unit tests
# ---------------------------------------------------------------------------

def test_submit_calls_fn():
    """submit() executes the provided callable."""
    reporter = EventReporter(max_workers=2, max_queue_depth=10)
    called = threading.Event()
    reporter.submit(called.set)
    assert called.wait(timeout=2.0), "callable was not invoked"
    reporter.shutdown(wait=True)


def test_failing_callable_does_not_propagate():
    """A callable that raises must not propagate an exception to the caller."""
    reporter = EventReporter(max_workers=2, max_queue_depth=10)

    def _boom():
        raise RuntimeError("intentional failure")

    # Should return without raising
    reporter.submit(_boom)
    reporter.shutdown(wait=True)


def test_full_queue_drops_and_warns(caplog):
    """When max_queue_depth=1 and executor saturated, extras are dropped with a warning."""
    import logging

    # Use a barrier to hold the single worker thread so the queue saturates.
    barrier = threading.Barrier(2)

    def _block():
        barrier.wait(timeout=5)

    with caplog.at_level(logging.WARNING, logger="aurarouter.event_reporter"):
        reporter = EventReporter(max_workers=1, max_queue_depth=1)

        # Submit first task — occupies the only semaphore slot.
        reporter.submit(_block)
        # Give the thread a moment to start so the slot is consumed.
        time.sleep(0.05)

        # This one should be dropped.
        reporter.submit(lambda: None)

    assert any("dropping event" in r.message for r in caplog.records), (
        "Expected a drop warning in logs"
    )

    # Release the blocked worker then shut down cleanly.
    barrier.wait(timeout=5)
    reporter.shutdown(wait=True)


def test_shutdown_after_full_queue_no_exception():
    """shutdown() after a dropped event does not raise."""
    reporter = EventReporter(max_workers=1, max_queue_depth=1)
    barrier = threading.Barrier(2)

    reporter.submit(lambda: barrier.wait(timeout=5))
    time.sleep(0.05)
    reporter.submit(lambda: None)  # dropped

    barrier.wait(timeout=5)
    reporter.shutdown(wait=True)  # must not raise


def test_submit_after_shutdown_does_not_raise():
    """submit() after shutdown() must not raise (executor RuntimeError swallowed)."""
    reporter = EventReporter(max_workers=2, max_queue_depth=10)
    reporter.shutdown(wait=True)
    reporter.submit(lambda: None)  # should be silently dropped


# ---------------------------------------------------------------------------
# ComputeFabric integration tests
# ---------------------------------------------------------------------------

def test_fabric_uses_event_reporter_not_threading_thread():
    """50 _report_usage calls must go through event_reporter.submit, not raw Thread."""
    fabric = _make_fabric()

    # Replace the live reporter with a mock so we can count submit calls
    mock_reporter = MagicMock()
    fabric._event_reporter = mock_reporter

    # Patch prerequisites for _report_usage to proceed past its guards
    fabric._xlm_usage_limiter = MagicMock()
    fabric._xlm_usage_limiter.acquire.return_value = True
    fabric._config.is_xlm_usage_reporting_enabled = MagicMock(return_value=True)
    fabric._config.get_xlm_endpoint = MagicMock(return_value="http://fake-xlm")

    with patch("aurarouter.telemetry_config.is_external_telemetry_enabled", return_value=True):
        for _ in range(50):
            fabric._report_usage("coding", "m1", True, 0.1)

    assert mock_reporter.submit.call_count == 50


def test_fabric_close_calls_event_reporter_shutdown():
    """Fabric.close() must call event_reporter.shutdown()."""
    fabric = _make_fabric()
    mock_reporter = MagicMock()
    fabric._event_reporter = mock_reporter

    fabric.close()

    mock_reporter.shutdown.assert_called_once_with(wait=False)


def test_fabric_record_feedback_uses_event_reporter():
    """_record_feedback() should submit to event_reporter, not spawn a Thread."""
    feedback_store = MagicMock()
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}}
    fabric = ComputeFabric(cfg, feedback_store=feedback_store)

    mock_reporter = MagicMock()
    fabric._event_reporter = mock_reporter

    fabric._record_feedback("coding", "m1", True, 0.5)

    mock_reporter.submit.assert_called_once()
