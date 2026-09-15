"""Tests for speculative decoding orchestrator (TG7)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import should_use_speculative, TriageResult
from aurarouter.savings.models import GenerateResult
from aurarouter.speculative import SpeculativeOrchestrator, SpeculativeSession


def _make_config(speculative_enabled=True, threshold=7, confidence=0.85):
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "drafter-3b": {"provider": "ollama", "model_name": "d", "endpoint": "http://x"},
            "verifier-70b": {"provider": "ollama", "model_name": "v", "endpoint": "http://x"},
        },
        "roles": {
            "coding": ["drafter-3b"],
            "reasoning": ["verifier-70b"],
            "router": ["drafter-3b"],
        },
        "system": {
            "speculative_decoding": speculative_enabled,
            "speculative_complexity_threshold": threshold,
            "notional_confidence_threshold": confidence,
            "speculative_timeout": 60.0,
        },
    }
    return cfg


def _make_fabric(cfg=None):
    return ComputeFabric(cfg or _make_config())


def _make_orchestrator(fabric=None, sovereignty_gate=None, triage_router=None):
    fb = fabric or _make_fabric()
    return SpeculativeOrchestrator(
        fabric=fb,
        mcp_registry=None,
        sovereignty_gate=sovereignty_gate,
        triage_router=triage_router,
    )


# --- Session lifecycle ---


class TestSpeculativeSession:
    def test_create_session_returns_active(self):
        orch = _make_orchestrator()
        session = orch.create_session("test task", "drafter-3b", "verifier-70b")

        assert session.status == "active"
        assert session.drafter_model == "drafter-3b"
        assert session.verifier_model == "verifier-70b"
        assert session.task == "test task"

    def test_session_id_is_unique(self):
        orch = _make_orchestrator()
        s1 = orch.create_session("t1", "d", "v")
        s2 = orch.create_session("t2", "d", "v")

        assert s1.session_id != s2.session_id

    def test_get_session_returns_created(self):
        orch = _make_orchestrator()
        session = orch.create_session("t", "d", "v")

        found = orch.get_session(session.session_id)
        assert found is session

    def test_get_session_unknown_returns_none(self):
        orch = _make_orchestrator()
        assert orch.get_session("nonexistent") is None

    def test_complete_session(self):
        orch = _make_orchestrator()
        session = orch.create_session("t", "d", "v")

        assert orch.complete_session(session.session_id)
        assert session.status == "completed"

    def test_active_sessions(self):
        orch = _make_orchestrator()
        s1 = orch.create_session("t1", "d", "v")
        s2 = orch.create_session("t2", "d", "v")
        orch.complete_session(s1.session_id)

        active = orch.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == s2.session_id

    def test_session_to_dict(self):
        session = SpeculativeSession(
            session_id="test-123",
            drafter_model="d",
            verifier_model="v",
            task="code",
        )
        d = session.to_dict()
        assert d["session_id"] == "test-123"
        assert d["status"] == "active"
        assert d["accepted_count"] == 0


# --- Configuration ---


class TestSpeculativeConfig:
    def test_is_enabled_true(self):
        orch = _make_orchestrator()
        assert orch.is_enabled()

    def test_is_enabled_false(self):
        cfg = _make_config(speculative_enabled=False)
        orch = _make_orchestrator(fabric=_make_fabric(cfg))
        assert not orch.is_enabled()

    def test_complexity_threshold_default(self):
        orch = _make_orchestrator()
        assert orch.complexity_threshold == 7

    def test_should_trigger_high_complexity(self):
        orch = _make_orchestrator()
        assert orch.should_trigger(8)

    def test_should_not_trigger_low_complexity(self):
        orch = _make_orchestrator()
        assert not orch.should_trigger(3)

    def test_confidence_threshold_configurable(self):
        cfg = _make_config(confidence=0.9)
        orch = _make_orchestrator(fabric=_make_fabric(cfg))
        assert orch.confidence_threshold == 0.9


# --- should_use_speculative routing helper ---


class TestShouldUseSpeculative:
    def test_enabled_high_complexity(self):
        fabric = _make_fabric()
        result = TriageResult(intent="COMPLEX_REASONING", complexity=8)
        assert should_use_speculative(fabric, result)

    def test_disabled_config(self):
        cfg = _make_config(speculative_enabled=False)
        fabric = _make_fabric(cfg)
        result = TriageResult(intent="COMPLEX_REASONING", complexity=8)
        assert not should_use_speculative(fabric, result)

    def test_low_complexity(self):
        fabric = _make_fabric()
        result = TriageResult(intent="SIMPLE_CODE", complexity=3)
        assert not should_use_speculative(fabric, result)

    def test_exact_threshold(self):
        fabric = _make_fabric()
        result = TriageResult(intent="COMPLEX_REASONING", complexity=7)
        assert should_use_speculative(fabric, result)


# --- Speculative execution ---


class TestSpeculativeExecution:
    def test_execute_speculative_with_drafter_success(self):
        fabric = _make_fabric()
        orch = _make_orchestrator(fabric=fabric)

        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="drafted output", model_id="drafter-3b"),
        ):
            result = asyncio.run(orch.execute_speculative("test task"))

        assert result is not None
        assert "content" in result
        assert result["content"] == "drafted output"

    def test_execute_speculative_drafter_failure_returns_none(self):
        fabric = _make_fabric()
        orch = _make_orchestrator(fabric=fabric)

        with patch.object(fabric, "execute", return_value=None):
            result = asyncio.run(orch.execute_speculative("test task"))

        assert result is None

    def test_notional_callback_invoked_on_high_confidence(self):
        fabric = _make_fabric(_make_config(confidence=0.0))  # Low threshold → always notional
        from aurarouter.savings.triage import TriageRouter, TriageRule

        triage = TriageRouter(rules=[TriageRule(max_complexity=10, preferred_role="reasoning")])
        orch = _make_orchestrator(fabric=fabric, triage_router=triage)

        notional_events = []

        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="draft", model_id="drafter-3b"),
        ):
            asyncio.run(orch.execute_speculative(
                "complex task",
                notional_callback=lambda e: notional_events.append(e),
            ))

        assert len(notional_events) >= 1
        assert notional_events[0]["status"] == "notional"

    def test_correction_callback_on_verification_rejection(self):
        fabric = _make_fabric()
        orch = _make_orchestrator(fabric=fabric)

        corrections = []

        # Patch _verify_draft to return rejection
        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="output", model_id="drafter-3b"),
        ), patch.object(
            orch, "_verify_draft",
            return_value={"accepted": False, "accepted_count": 0, "total_tokens": 5},
        ):
            asyncio.run(orch.execute_speculative(
                "task",
                correction_callback=lambda e: corrections.append(e),
            ))

        assert len(corrections) == 1
        assert corrections[0]["reason"] == "verifier_rejection"
