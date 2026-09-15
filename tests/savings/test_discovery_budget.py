"""Tests for T9.1 RegistrationStore and T9.2 BudgetSyncStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aurarouter.budget_sync import (
    BudgetSyncStore,
    get_global_budget_fn,
    report_budget_sync_fn,
)
from aurarouter.errors import AuraError
from aurarouter.registration import (
    RegistrationStore,
    discovery_status_fn,
    registration_ready_fn,
    xlm_gate_check,
)


# ---------------------------------------------------------------------------
# T9.1 — RegistrationStore
# ---------------------------------------------------------------------------


class TestRegistrationStore:
    def test_announce_returns_uuid_and_capabilities(self):
        store = RegistrationStore()
        cid, caps = store.announce("auraxlm", ["rag", "moe"])
        assert len(cid) == 36  # UUID format
        assert "rag" in caps
        assert "moe" in caps

    def test_announce_overwrites_previous_registration(self):
        store = RegistrationStore()
        cid1, _ = store.announce("auraxlm", ["rag"])
        cid2, _ = store.announce("auraxlm", ["moe"])
        assert cid1 != cid2
        assert not store.is_operational("auraxlm")  # pending after re-announce

    def test_mark_ready_valid_catalog_id(self):
        store = RegistrationStore()
        cid, _ = store.announce("auraxlm", [])
        found = store.mark_ready(cid)
        assert found is True
        assert store.is_operational("auraxlm")

    def test_mark_ready_unknown_catalog_id_returns_false(self):
        store = RegistrationStore()
        found = store.mark_ready("does-not-exist")
        assert found is False

    def test_is_operational_pending_returns_false(self):
        store = RegistrationStore()
        store.announce("auraxlm", [])
        assert not store.is_operational("auraxlm")

    def test_is_operational_unregistered_returns_false(self):
        store = RegistrationStore()
        assert not store.is_operational("auraxlm")

    def test_get_status_empty(self):
        store = RegistrationStore()
        assert store.get_status() == []

    def test_get_status_after_announce_and_ready(self):
        store = RegistrationStore()
        cid, _ = store.announce("auraxlm", ["rag"])
        store.mark_ready(cid)
        statuses = store.get_status()
        assert len(statuses) == 1
        assert statuses[0]["name"] == "auraxlm"
        assert statuses[0]["status"] == "operational"
        assert statuses[0]["catalog_id"] == cid

    def test_multiple_services_tracked_independently(self):
        store = RegistrationStore()
        cid_xlm, _ = store.announce("auraxlm", ["rag"])
        cid_grid, _ = store.announce("auragrid", ["compute"])
        store.mark_ready(cid_xlm)
        # Only xlm is operational
        assert store.is_operational("auraxlm")
        assert not store.is_operational("auragrid")


# ---------------------------------------------------------------------------
# T9.1 — xlm_gate_check
# ---------------------------------------------------------------------------


class TestXlmGateCheck:
    def test_gate_blocks_unregistered(self):
        store = RegistrationStore()
        result = xlm_gate_check(store)
        assert result is not None
        err = json.loads(result)
        assert err["error_code"] == 1005
        assert err["category"] == "infrastructure"
        assert err["source_project"] == "aurarouter"

    def test_gate_blocks_pending(self):
        store = RegistrationStore()
        store.announce("auraxlm", [])
        assert xlm_gate_check(store) is not None

    def test_gate_passes_operational(self):
        store = RegistrationStore()
        cid, _ = store.announce("auraxlm", [])
        store.mark_ready(cid)
        assert xlm_gate_check(store) is None

    def test_gate_error_is_valid_aura_error(self):
        store = RegistrationStore()
        result = xlm_gate_check(store)
        assert result is not None
        # Must deserialize into a valid AuraError
        err = AuraError.model_validate_json(result)
        assert err.error_code == 1005


# ---------------------------------------------------------------------------
# T9.1 — MCP-callable helpers
# ---------------------------------------------------------------------------


class TestRegistrationMcpHelpers:
    def test_ready_fn_valid_catalog_id(self):
        store = RegistrationStore()
        cid, _ = store.announce("auraxlm", [])
        result = json.loads(registration_ready_fn(store, cid))
        assert result["ok"] is True
        assert result["catalog_id"] == cid

    def test_ready_fn_unknown_catalog_id(self):
        store = RegistrationStore()
        result = json.loads(registration_ready_fn(store, "bad-id"))
        assert "error" in result

    def test_ready_fn_empty_catalog_id(self):
        store = RegistrationStore()
        result = json.loads(registration_ready_fn(store, ""))
        assert "error" in result

    def test_discovery_status_fn_empty(self):
        store = RegistrationStore()
        result = json.loads(discovery_status_fn(store))
        assert result == {"services": []}

    def test_discovery_status_fn_after_full_handshake(self):
        store = RegistrationStore()
        cid, _ = store.announce("auraxlm", ["rag"])
        store.mark_ready(cid)
        result = json.loads(discovery_status_fn(store))
        svc = result["services"][0]
        assert svc["status"] == "operational"
        assert svc["name"] == "auraxlm"


# ---------------------------------------------------------------------------
# T9.2 — BudgetSyncStore
# ---------------------------------------------------------------------------


_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_NOW_ISO = _NOW.isoformat()


def _make_payload(source: str, input_tok: int = 100, output_tok: int = 50) -> str:
    return json.dumps(
        {
            "source": source,
            "period_start": _NOW_ISO,
            "period_end": _NOW_ISO,
            "token_spend": {"input": input_tok, "output": output_tok},
            "inference_cost_usd": 0.01,
            "compute_cost_usd": 0.0,
        }
    )


class TestBudgetSyncStore:
    def test_record_stores_message(self):
        store = BudgetSyncStore()
        msg = store.get_all()
        assert msg == []
        # report via helper
        report_budget_sync_fn(store, _make_payload("auraxlm"))
        assert len(store.get_all()) == 1

    def test_last_write_wins_per_source(self):
        store = BudgetSyncStore()
        report_budget_sync_fn(store, _make_payload("auraxlm", input_tok=100))
        report_budget_sync_fn(store, _make_payload("auraxlm", input_tok=999))
        assert len(store.get_all()) == 1
        assert store.get_all()[0].token_spend["input"] == 999

    def test_different_sources_stored_separately(self):
        store = BudgetSyncStore()
        report_budget_sync_fn(store, _make_payload("auraxlm"))
        report_budget_sync_fn(store, _make_payload("aurarouter"))
        assert len(store.get_all()) == 2

    def test_reset_clears_all(self):
        store = BudgetSyncStore()
        report_budget_sync_fn(store, _make_payload("auraxlm"))
        store.reset()
        assert store.get_all() == []


class TestReportBudgetSyncFn:
    def test_valid_payload_returns_ok(self):
        store = BudgetSyncStore()
        result = json.loads(report_budget_sync_fn(store, _make_payload("auraxlm")))
        assert result["ok"] is True
        assert result["source"] == "auraxlm"

    def test_invalid_json_returns_error(self):
        store = BudgetSyncStore()
        result = json.loads(report_budget_sync_fn(store, "not-json"))
        assert "error" in result

    def test_invalid_source_returns_error(self):
        store = BudgetSyncStore()
        payload = json.dumps(
            {
                "source": "unknown_service",  # not in Literal
                "period_start": _NOW_ISO,
                "period_end": _NOW_ISO,
                "token_spend": {"input": 0, "output": 0},
            }
        )
        result = json.loads(report_budget_sync_fn(store, payload))
        assert "error" in result

    def test_missing_required_field_returns_error(self):
        store = BudgetSyncStore()
        payload = json.dumps({"source": "auraxlm"})  # missing period_start etc.
        result = json.loads(report_budget_sync_fn(store, payload))
        assert "error" in result


class TestGetGlobalBudgetFn:
    def test_empty_store(self):
        store = BudgetSyncStore()
        result = json.loads(get_global_budget_fn(store))
        assert result["total_input_tokens"] == 0
        assert result["total_output_tokens"] == 0
        assert result["sources_reported"] == []

    def test_single_source(self):
        store = BudgetSyncStore()
        report_budget_sync_fn(store, _make_payload("auraxlm", input_tok=1000, output_tok=500))
        result = json.loads(get_global_budget_fn(store))
        assert result["total_input_tokens"] == 1000
        assert result["total_output_tokens"] == 500
        assert "auraxlm" in result["sources_reported"]

    def test_multiple_sources_aggregated(self):
        store = BudgetSyncStore()
        report_budget_sync_fn(store, _make_payload("auraxlm", input_tok=1000, output_tok=500))
        report_budget_sync_fn(store, _make_payload("aurarouter", input_tok=200, output_tok=100))
        result = json.loads(get_global_budget_fn(store))
        assert result["total_input_tokens"] == 1200
        assert result["total_output_tokens"] == 600
        assert set(result["sources_reported"]) == {"auraxlm", "aurarouter"}

    def test_total_costs_summed(self):
        store = BudgetSyncStore()
        p1 = json.dumps(
            {
                "source": "auraxlm",
                "period_start": _NOW_ISO,
                "period_end": _NOW_ISO,
                "token_spend": {"input": 0, "output": 0},
                "inference_cost_usd": 0.05,
                "compute_cost_usd": 0.01,
            }
        )
        p2 = json.dumps(
            {
                "source": "aurarouter",
                "period_start": _NOW_ISO,
                "period_end": _NOW_ISO,
                "token_spend": {"input": 0, "output": 0},
                "inference_cost_usd": 0.02,
                "compute_cost_usd": 0.0,
            }
        )
        report_budget_sync_fn(store, p1)
        report_budget_sync_fn(store, p2)
        result = json.loads(get_global_budget_fn(store))
        assert abs(result["total_inference_cost_usd"] - 0.07) < 1e-9
        assert abs(result["total_compute_cost_usd"] - 0.01) < 1e-9
