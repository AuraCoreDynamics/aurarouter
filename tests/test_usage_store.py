"""Tests for the SQLite-backed UsageStore."""

import threading

import pytest

from aurarouter.savings.models import UsageRecord
from aurarouter.savings.usage_store import UsageStore


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


def test_store_creation(tmp_path):
    db_path = tmp_path / "usage.db"
    store = UsageStore(db_path=db_path)
    assert db_path.exists()


def test_record_and_query(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    rec = _make_record()
    store.record(rec)

    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert r.model_id == "test-model"
    assert r.provider == "ollama"
    assert r.input_tokens == 100
    assert r.output_tokens == 200
    assert r.elapsed_s == 1.0
    assert r.success is True
    assert r.is_cloud is False


def test_query_filters(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(_make_record(model_id="m1", provider="ollama", role="router"))
    store.record(_make_record(model_id="m2", provider="google", role="reasoning"))
    store.record(_make_record(model_id="m3", provider="claude", role="coding"))

    # Filter by model_id
    rows = store.query(model_id="m2")
    assert len(rows) == 1
    assert rows[0].model_id == "m2"

    # Filter by provider
    rows = store.query(provider="google")
    assert len(rows) == 1
    assert rows[0].provider == "google"

    # Filter by role
    rows = store.query(role="coding")
    assert len(rows) == 1
    assert rows[0].role == "coding"


def test_query_date_range(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(_make_record(timestamp="2025-01-10T00:00:00Z"))
    store.record(_make_record(timestamp="2025-01-15T00:00:00Z"))
    store.record(_make_record(timestamp="2025-01-20T00:00:00Z"))

    rows = store.query(start="2025-01-12T00:00:00Z", end="2025-01-18T00:00:00Z")
    assert len(rows) == 1
    assert rows[0].timestamp == "2025-01-15T00:00:00Z"


def test_aggregate_tokens(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(_make_record(model_id="m1", input_tokens=10, output_tokens=20))
    store.record(_make_record(model_id="m1", input_tokens=30, output_tokens=40))
    store.record(_make_record(model_id="m2", input_tokens=50, output_tokens=60))

    agg = store.aggregate_tokens()
    assert len(agg) == 2

    by_model = {r["model_id"]: r for r in agg}
    assert by_model["m1"]["input_tokens"] == 40
    assert by_model["m1"]["output_tokens"] == 60
    assert by_model["m1"]["total_tokens"] == 100
    assert by_model["m2"]["input_tokens"] == 50
    assert by_model["m2"]["output_tokens"] == 60
    assert by_model["m2"]["total_tokens"] == 110


def test_total_tokens(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(_make_record(input_tokens=100, output_tokens=200))
    store.record(_make_record(input_tokens=300, output_tokens=400))

    totals = store.total_tokens()
    assert totals["input_tokens"] == 400
    assert totals["output_tokens"] == 600
    assert totals["total_tokens"] == 1000


def test_total_tokens_empty(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    totals = store.total_tokens()
    assert totals["input_tokens"] == 0
    assert totals["output_tokens"] == 0
    assert totals["total_tokens"] == 0


def test_purge_before(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(_make_record(timestamp="2025-01-10T00:00:00Z"))
    store.record(_make_record(timestamp="2025-01-15T00:00:00Z"))
    store.record(_make_record(timestamp="2025-01-20T00:00:00Z"))

    deleted = store.purge_before("2025-01-16T00:00:00Z")
    assert deleted == 2

    rows = store.query()
    assert len(rows) == 1
    assert rows[0].timestamp == "2025-01-20T00:00:00Z"


def test_thread_safety(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(20):
                store.record(
                    _make_record(
                        model_id=f"thread-{thread_id}",
                        timestamp=f"2025-01-15T10:{thread_id:02d}:{i:02d}Z",
                    )
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"

    rows = store.query()
    assert len(rows) == 40  # 2 threads * 20 records each
