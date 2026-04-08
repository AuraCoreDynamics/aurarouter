"""
End-to-end ROI telemetry validation harness for AuraRouter.

Verifies that:
- simulated_cost_avoided accumulates correctly in UsageStore for hard-routed (local) tasks.
- AuraRouterAPI.get_roi_metrics() surfaces accurate aggregated values.
- The UsageStore SQLite layer persists and re-reads the metrics faithfully.
- Mixed batches (simple + complex) produce the correct hard_route_percentage.

Run with: pytest tests/integration/validate_roi_telemetry.py -v
"""
from __future__ import annotations

import asyncio
import datetime
import sqlite3

import pytest
import pytest_asyncio

from aurarouter.api import AuraRouterAPI, APIConfig
from aurarouter.savings.models import UsageRecord
from aurarouter.savings.usage_store import UsageStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCAL_MODEL_ID = "qwen3:8b-local"
CLOUD_MODEL_ID = "gpt-4o-mini-cloud"
SIMULATED_COST_PER_CALL = 0.0025   # USD per hard-routed call deflected from cloud
SIMPLE_PROMPT_COMPLEXITY = 2       # triggers hard-route + local boost
COMPLEX_PROMPT_COMPLEXITY = 8      # no hard-route — dispatched to cloud

SIMPLE_PROMPTS = [
    "Write a Python function to reverse a string.",
    "What is the syntax for a for loop in Python?",
    "Write a bash one-liner to count lines in a file.",
    "How do I rename a variable in VS Code?",
    "Write a SQL SELECT with a WHERE clause.",
]
COMPLEX_PROMPTS = [
    "Design a distributed event-sourced architecture for a financial ledger system "
    "with split-brain resilience, CQRS, and eventual consistency guarantees.",
    "Explain the trade-offs between actor-model concurrency (Erlang/Akka) and "
    "structured concurrency (Go goroutines, Python asyncio) at scale.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _make_local_record(timestamp: str) -> UsageRecord:
    return UsageRecord(
        timestamp=timestamp,
        model_id=LOCAL_MODEL_ID,
        provider="ollama",
        role="coding",
        intent="SIMPLE_CODE",
        input_tokens=50,
        output_tokens=80,
        elapsed_s=0.8,
        success=True,
        is_cloud=False,
        simulated_cost_avoided=SIMULATED_COST_PER_CALL,
        complexity_score=SIMPLE_PROMPT_COMPLEXITY,
    )


def _make_cloud_record(timestamp: str) -> UsageRecord:
    return UsageRecord(
        timestamp=timestamp,
        model_id=CLOUD_MODEL_ID,
        provider="openapi",
        role="coding",
        intent="COMPLEX_REASONING",
        input_tokens=500,
        output_tokens=1200,
        elapsed_s=8.5,
        success=True,
        is_cloud=True,
        simulated_cost_avoided=0.0,
        complexity_score=COMPLEX_PROMPT_COMPLEXITY,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def seeded_store(tmp_path):
    """
    Yield a UsageStore pre-seeded with SIMPLE_PROMPTS (local, hard-routed) and
    COMPLEX_PROMPTS (cloud) records. Provides a known ground truth for ROI assertions.
    The caller must NOT close this store — teardown handles it.
    """
    db_path = tmp_path / "seeded_roi.db"
    store = UsageStore(db_path=db_path)
    ts = _now_iso()

    for _ in SIMPLE_PROMPTS:
        store.record(_make_local_record(ts))

    for _ in COMPLEX_PROMPTS:
        store.record(_make_cloud_record(ts))

    try:
        yield store
    finally:
        store.close()


def _api_with_store(store: UsageStore) -> AuraRouterAPI:
    """
    Build an AuraRouterAPI wired to a custom UsageStore.
    enable_savings=False prevents creation of a default store at ~/.auracore.
    """
    api = AuraRouterAPI(APIConfig(enable_savings=False))
    api._usage_store = store
    return api


# ---------------------------------------------------------------------------
# ROI metric accumulation
# ---------------------------------------------------------------------------

async def test_total_cost_avoided_equals_sum_of_seeded_records(seeded_store):
    """
    total_simulated_cost_avoided must equal COST_PER_CALL * N_SIMPLE exactly.
    This verifies the UsageStore accumulates correctly before any API layer.
    """
    expected = SIMULATED_COST_PER_CALL * len(SIMPLE_PROMPTS)

    records = await asyncio.to_thread(seeded_store.query)
    actual = sum(r.simulated_cost_avoided for r in records)

    assert abs(actual - expected) < 1e-6, (
        f"Expected total cost avoided={expected:.6f}, got {actual:.6f}"
    )


async def test_get_roi_metrics_total_matches_store(seeded_store):
    """
    AuraRouterAPI.get_roi_metrics() must return total_simulated_cost_avoided
    that matches direct summation from UsageStore.query().
    """
    expected = SIMULATED_COST_PER_CALL * len(SIMPLE_PROMPTS)

    api = _api_with_store(seeded_store)
    try:
        metrics = await asyncio.to_thread(api.get_roi_metrics, 1)
        assert abs(metrics.total_simulated_cost_avoided - expected) < 1e-6, (
            f"ROIMetrics reported {metrics.total_simulated_cost_avoided:.6f}, "
            f"expected {expected:.6f}"
        )
    finally:
        api._usage_store = None   # detach — fixture owns the store's lifecycle
        api.close()


async def test_hard_route_percentage_with_mixed_batch(seeded_store):
    """
    hard_route_percentage = (n_local / n_total) * 100.
    With N simple (local) and M complex (cloud) records the percentage must be exact.
    """
    n_local = len(SIMPLE_PROMPTS)
    n_total = len(SIMPLE_PROMPTS) + len(COMPLEX_PROMPTS)
    expected_pct = (n_local / n_total) * 100.0

    api = _api_with_store(seeded_store)
    try:
        metrics = await asyncio.to_thread(api.get_roi_metrics, 1)
        assert abs(metrics.hard_route_percentage - expected_pct) < 0.01, (
            f"hard_route_percentage={metrics.hard_route_percentage:.2f}%, "
            f"expected {expected_pct:.2f}%"
        )
    finally:
        api._usage_store = None
        api.close()


async def test_zero_cost_avoided_for_cloud_records(seeded_store):
    """
    Cloud records must have simulated_cost_avoided == 0.0.
    They are not deflected costs; they represent real cloud spend.
    """
    records = await asyncio.to_thread(seeded_store.query)
    cloud_records = [r for r in records if r.is_cloud]

    assert cloud_records, "No cloud records found — check seeded_store fixture"
    for rec in cloud_records:
        assert rec.simulated_cost_avoided == 0.0, (
            f"Cloud record for {rec.model_id} has non-zero "
            f"simulated_cost_avoided={rec.simulated_cost_avoided}"
        )


# ---------------------------------------------------------------------------
# UsageStore persistence (SQLite round-trip)
# ---------------------------------------------------------------------------

async def test_usagestore_persists_simulated_cost_avoided(tmp_path):
    """
    Write a record with a known simulated_cost_avoided, close the store,
    reopen it against the same file, and assert the value survived the round-trip.
    This guards against schema evolution breaking the column.
    """
    db_path = tmp_path / "persist_test.db"
    expected_cost = 0.0031415

    rec = UsageRecord(
        timestamp=_now_iso(),
        model_id=LOCAL_MODEL_ID,
        provider="ollama",
        role="coding",
        intent="SIMPLE_CODE",
        input_tokens=100,
        output_tokens=150,
        elapsed_s=0.5,
        success=True,
        is_cloud=False,
        simulated_cost_avoided=expected_cost,
        complexity_score=SIMPLE_PROMPT_COMPLEXITY,
    )

    store = UsageStore(db_path=db_path)
    await asyncio.to_thread(store.record, rec)
    store.close()

    # Reopen — simulates server restart
    reopened = UsageStore(db_path=db_path)
    try:
        rows = await asyncio.to_thread(reopened.query)
        assert len(rows) == 1
        assert abs(rows[0].simulated_cost_avoided - expected_cost) < 1e-7, (
            f"After reopen: cost={rows[0].simulated_cost_avoided}, expected={expected_cost}"
        )
        assert rows[0].complexity_score == SIMPLE_PROMPT_COMPLEXITY
    finally:
        reopened.close()


async def test_usagestore_schema_has_required_columns(tmp_path):
    """
    Verify the SQLite schema includes simulated_cost_avoided and complexity_score
    after initialization — guards against schema migration regressions.
    """
    db_path = tmp_path / "schema_check.db"
    store = UsageStore(db_path=db_path)
    store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("PRAGMA table_info(usage)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "simulated_cost_avoided" in columns, \
            "Column 'simulated_cost_avoided' missing from usage table"
        assert "complexity_score" in columns, \
            "Column 'complexity_score' missing from usage table"
    finally:
        conn.close()


async def test_only_local_records_contribute_to_cost_avoided(seeded_store):
    """
    Strict check: total_simulated_cost_avoided must equal COST_PER_CALL * N_simple,
    not more. Verifies cloud records (cost_avoided=0.0) are not contaminating the sum.
    """
    expected = SIMULATED_COST_PER_CALL * len(SIMPLE_PROMPTS)

    rows = await asyncio.to_thread(seeded_store.query)
    total = sum(r.simulated_cost_avoided for r in rows)

    assert abs(total - expected) < 1e-6, (
        f"Total cost avoided {total:.6f} != expected {expected:.6f}. "
        f"Cloud records may be contaminating the sum."
    )


async def test_recent_hard_routed_in_roi_metrics(seeded_store):
    """
    ROIMetrics.recent_hard_routed must contain exactly N_simple entries,
    each with the correct savings value.
    """
    api = _api_with_store(seeded_store)
    try:
        metrics = await asyncio.to_thread(api.get_roi_metrics, 1)
        assert len(metrics.recent_hard_routed) == len(SIMPLE_PROMPTS), (
            f"Expected {len(SIMPLE_PROMPTS)} recent hard-routed entries, "
            f"got {len(metrics.recent_hard_routed)}"
        )
        for entry in metrics.recent_hard_routed:
            assert abs(entry["savings"] - SIMULATED_COST_PER_CALL) < 1e-7, (
                f"recent_hard_routed entry has wrong savings: {entry}"
            )
    finally:
        api._usage_store = None
        api.close()
