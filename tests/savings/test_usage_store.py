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
    store.record(_make_record(model_id="m2", provider="openapi", role="reasoning"))
    store.record(_make_record(model_id="m3", provider="llamacpp-server", role="coding"))

    # Filter by model_id
    rows = store.query(model_id="m2")
    assert len(rows) == 1
    assert rows[0].model_id == "m2"

    # Filter by provider
    rows = store.query(provider="openapi")
    assert len(rows) == 1
    assert rows[0].provider == "openapi"

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


def test_persistent_connection_reused(tmp_path):
    """Verify that multiple record() calls reuse the same SQLite connection."""
    store = UsageStore(db_path=tmp_path / "usage.db")

    # After init, _conn should already be set (from _init_db)
    conn_after_init = store._conn
    assert conn_after_init is not None

    # Record multiple entries
    store.record(_make_record(timestamp="2025-01-15T10:00:00Z"))
    conn_after_first = store._conn
    store.record(_make_record(timestamp="2025-01-15T11:00:00Z"))
    conn_after_second = store._conn
    store.record(_make_record(timestamp="2025-01-15T12:00:00Z"))
    conn_after_third = store._conn

    # All should be the exact same connection object
    assert conn_after_init is conn_after_first
    assert conn_after_first is conn_after_second
    assert conn_after_second is conn_after_third

    # Verify data is persisted correctly
    rows = store.query()
    assert len(rows) == 3


def test_close_and_reconnect(tmp_path):
    """Verify close() releases connection and subsequent ops reconnect."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(_make_record(timestamp="2025-01-15T10:00:00Z"))
    old_conn = store._conn

    store.close()
    assert store._conn is None

    # Next operation should create a new connection
    store.record(_make_record(timestamp="2025-01-15T11:00:00Z"))
    assert store._conn is not None
    assert store._conn is not old_conn

    rows = store.query()
    assert len(rows) == 2


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

def test_schema_migration(tmp_path):
    import sqlite3
    db_path = tmp_path / "usage.db"
    
    # Create old schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
    CREATE TABLE usage (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        model_id    TEXT    NOT NULL,
        provider    TEXT    NOT NULL,
        role        TEXT    NOT NULL,
        intent      TEXT    NOT NULL,
        input_tokens  INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        elapsed_s   REAL    NOT NULL,
        success     INTEGER NOT NULL,
        is_cloud    INTEGER NOT NULL
    )
    """)
    conn.commit()
    conn.close()

    # Init should trigger migration
    store = UsageStore(db_path=db_path)
    
    # Verify new columns exist
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(usage)")
    columns = [row[1] for row in cursor.fetchall()]
    assert "simulated_cost_avoided" in columns
    assert "complexity_score" in columns
    conn.close()

class DummyRoutingContext:
    def __init__(self, cost, complexity):
        self.simulated_cost_avoided = cost
        self.complexity_score = complexity

def test_record_with_roi(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    rec = _make_record()
    routing_ctx = DummyRoutingContext(cost=1.25, complexity=8)
    store.record(rec, routing_context=routing_ctx)

    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert r.simulated_cost_avoided == 1.25
    assert r.complexity_score == 8

def test_record_without_roi(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    rec = _make_record()
    store.record(rec)

    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert r.simulated_cost_avoided == 0.0
    assert r.complexity_score == 0

def test_integration_sanity_check(tmp_path):
    """Integration sanity check that simulates a hard-routed execution and verifies the entry appears in the UsageStore with non-zero metrics."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    
    # Simulate a hard-routed task that would have gone to cloud
    # The analyzer decides to route locally, calculating savings and complexity
    routing_ctx = DummyRoutingContext(cost=0.042, complexity=3)
    rec = _make_record(
        model_id="local-phi3",
        provider="ollama",
        is_cloud=False,
        input_tokens=1500,
        output_tokens=350,
        elapsed_s=2.5
    )
    
    store.record(rec, routing_context=routing_ctx)
    
    # Verify through query
    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert not r.is_cloud
    assert r.simulated_cost_avoided == 0.042
    assert r.complexity_score == 3

