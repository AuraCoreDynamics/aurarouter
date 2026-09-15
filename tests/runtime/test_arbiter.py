"""Tests for the Collision Detector & Reasoning Arbiter (TG5).

Covers:
- ArbiterDecision.from_dict()
- build_arbiter_prompt()
- resolve_collisions() with mocked fabric
- Integration: broker collisions -> arbiter invocation
- Trace entries for collision/resolution/fallback events
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.broker import AnalyzerBid, BrokerResult
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import ArbiterDecision, build_arbiter_prompt, resolve_collisions
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fabric() -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return ComputeFabric(cfg)


def _make_collision_pair() -> tuple[AnalyzerBid, AnalyzerBid]:
    bid_a = AnalyzerBid(
        analyzer_id="analyzer-alpha",
        confidence=0.9,
        claimed_files=["src/main.py", "src/utils.py"],
        proposed_tasks=[{"task": "refactor main"}],
        role="coding",
    )
    bid_b = AnalyzerBid(
        analyzer_id="analyzer-beta",
        confidence=0.85,
        claimed_files=["src/utils.py", "src/helpers.py"],
        proposed_tasks=[{"task": "update utils"}],
        role="reasoning",
    )
    return bid_a, bid_b


def _make_broker_result_with_collisions() -> BrokerResult:
    bid_a, bid_b = _make_collision_pair()
    return BrokerResult(
        bids=[bid_a, bid_b],
        collisions=[(bid_a, bid_b)],
        merged_plan=None,
        execution_trace=["Broker: collision detected between analyzer-alpha and analyzer-beta"],
    )


# ---------------------------------------------------------------------------
# T5.1: ArbiterDecision.from_dict
# ---------------------------------------------------------------------------

class TestArbiterDecisionFromDict:
    def test_valid_dict(self):
        data = {
            "execution_order": [
                {"analyzer_id": "a1", "role": "coding", "tasks": ["t1"], "files": ["f1"]},
            ],
            "reasoning": "a1 is more specialized",
            "strategy": "sequential",
        }
        decision = ArbiterDecision.from_dict(data)
        assert decision.strategy == "sequential"
        assert decision.reasoning == "a1 is more specialized"
        assert len(decision.execution_order) == 1
        assert decision.execution_order[0]["analyzer_id"] == "a1"

    def test_missing_fields_uses_defaults(self):
        decision = ArbiterDecision.from_dict({})
        assert decision.execution_order == []
        assert decision.reasoning == ""
        assert decision.strategy == "winner_takes_all"

    def test_partial_dict(self):
        data = {"strategy": "split", "reasoning": "split work"}
        decision = ArbiterDecision.from_dict(data)
        assert decision.strategy == "split"
        assert decision.reasoning == "split work"
        assert decision.execution_order == []


# ---------------------------------------------------------------------------
# T5.2: build_arbiter_prompt
# ---------------------------------------------------------------------------

class TestBuildArbiterPrompt:
    def test_user_request_verbatim(self):
        bid_a, bid_b = _make_collision_pair()
        prompt = build_arbiter_prompt(
            "Refactor the authentication module",
            [(bid_a, bid_b)],
        )
        assert "Refactor the authentication module" in prompt

    def test_collision_formatting(self):
        bid_a, bid_b = _make_collision_pair()
        prompt = build_arbiter_prompt("some task", [(bid_a, bid_b)])
        assert "analyzer-alpha" in prompt
        assert "analyzer-beta" in prompt
        assert "0.9" in prompt
        assert "0.85" in prompt
        assert "src/utils.py" in prompt  # shared file

    def test_file_context_as_workspace_files(self):
        bid_a, bid_b = _make_collision_pair()
        file_ctx = [
            {"path": "src/main.py", "language": "python"},
            {"path": "src/utils.py", "language": "python"},
        ]
        prompt = build_arbiter_prompt("task", [(bid_a, bid_b)], file_context=file_ctx)
        assert "WORKSPACE FILES" in prompt
        assert "src/main.py" in prompt
        assert "(python)" in prompt

    def test_no_file_context(self):
        bid_a, bid_b = _make_collision_pair()
        prompt = build_arbiter_prompt("task", [(bid_a, bid_b)], file_context=None)
        assert "WORKSPACE FILES" not in prompt

    def test_multiple_collisions(self):
        bid_a = AnalyzerBid(analyzer_id="a1", confidence=0.8, claimed_files=["f1"])
        bid_b = AnalyzerBid(analyzer_id="a2", confidence=0.7, claimed_files=["f1"])
        bid_c = AnalyzerBid(analyzer_id="a3", confidence=0.9, claimed_files=["f1"])
        prompt = build_arbiter_prompt("task", [(bid_a, bid_b), (bid_a, bid_c)])
        assert "Collision 1" in prompt
        assert "Collision 2" in prompt

    def test_json_response_format_specified(self):
        bid_a, bid_b = _make_collision_pair()
        prompt = build_arbiter_prompt("task", [(bid_a, bid_b)])
        assert "execution_order" in prompt
        assert "strategy" in prompt
        assert "reasoning" in prompt


# ---------------------------------------------------------------------------
# T5.3: resolve_collisions
# ---------------------------------------------------------------------------

class TestResolveCollisions:
    def test_valid_json_response(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        arbiter_response = {
            "execution_order": [
                {"analyzer_id": "analyzer-alpha", "role": "coding", "tasks": ["refactor"], "files": ["src/main.py"]},
            ],
            "reasoning": "alpha has higher confidence and broader scope",
            "strategy": "winner_takes_all",
        }

        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text=json.dumps(arbiter_response)),
        ):
            decision = resolve_collisions(fabric, "refactor auth", broker_result)

        assert decision.strategy == "winner_takes_all"
        assert len(decision.execution_order) == 1
        assert decision.execution_order[0]["analyzer_id"] == "analyzer-alpha"
        assert decision.reasoning == "alpha has higher confidence and broader scope"

    def test_uses_reasoning_role(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        arbiter_response = {
            "execution_order": [],
            "reasoning": "ok",
            "strategy": "winner_takes_all",
        }

        with patch.object(fabric, "execute", return_value=GenerateResult(text=json.dumps(arbiter_response))) as mock_exec:
            resolve_collisions(fabric, "task", broker_result)
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[0][0] == "reasoning"  # first positional arg is role
            assert call_args[1].get("json_mode") is True

    def test_fallback_on_garbage_response(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        with patch.object(fabric, "execute", return_value=GenerateResult(text="not valid json {")):
            decision = resolve_collisions(fabric, "task", broker_result)

        # Should fall back to highest-confidence bid (analyzer-alpha at 0.9)
        assert decision.strategy == "winner_takes_all"
        assert len(decision.execution_order) == 1
        assert decision.execution_order[0]["analyzer_id"] == "analyzer-alpha"
        assert "Fallback" in decision.reasoning

    def test_fallback_on_none_response(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        with patch.object(fabric, "execute", return_value=None):
            decision = resolve_collisions(fabric, "task", broker_result)

        assert decision.strategy == "winner_takes_all"
        assert decision.execution_order[0]["analyzer_id"] == "analyzer-alpha"
        assert "Fallback" in decision.reasoning

    def test_fallback_on_exception(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        with patch.object(fabric, "execute", side_effect=RuntimeError("model down")):
            decision = resolve_collisions(fabric, "task", broker_result)

        assert decision.strategy == "winner_takes_all"
        assert "Fallback" in decision.reasoning

    def test_file_context_passed_to_prompt(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()
        file_ctx = [{"path": "a.py", "language": "python"}]

        arbiter_response = {
            "execution_order": [],
            "reasoning": "ok",
            "strategy": "split",
        }

        with patch.object(fabric, "execute", return_value=GenerateResult(text=json.dumps(arbiter_response))) as mock_exec:
            resolve_collisions(fabric, "task", broker_result, file_context=file_ctx)
            prompt_arg = mock_exec.call_args[0][1]
            assert "WORKSPACE FILES" in prompt_arg
            assert "a.py" in prompt_arg


# ---------------------------------------------------------------------------
# Trace tests
# ---------------------------------------------------------------------------

class TestArbiterTrace:
    def test_trace_on_successful_resolution(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        arbiter_response = {
            "execution_order": [{"analyzer_id": "a", "role": "coding"}],
            "reasoning": "picked a",
            "strategy": "sequential",
        }

        with patch.object(fabric, "execute", return_value=GenerateResult(text=json.dumps(arbiter_response))):
            resolve_collisions(fabric, "task", broker_result)

        trace_str = " ".join(broker_result.execution_trace)
        assert "resolving" in trace_str.lower()
        assert "resolved" in trace_str.lower()
        assert "sequential" in trace_str

    def test_trace_on_fallback(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        with patch.object(fabric, "execute", return_value=None):
            resolve_collisions(fabric, "task", broker_result)

        trace_str = " ".join(broker_result.execution_trace)
        assert "fallback" in trace_str.lower()

    def test_trace_preserves_existing_entries(self):
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()
        original_trace_len = len(broker_result.execution_trace)

        arbiter_response = {
            "execution_order": [],
            "reasoning": "ok",
            "strategy": "winner_takes_all",
        }

        with patch.object(fabric, "execute", return_value=GenerateResult(text=json.dumps(arbiter_response))):
            resolve_collisions(fabric, "task", broker_result)

        # Should have added entries, not replaced
        assert len(broker_result.execution_trace) > original_trace_len


# ---------------------------------------------------------------------------
# Integration: broker collisions -> arbiter in route_task
# ---------------------------------------------------------------------------

class TestArbiterIntegration:
    def test_route_task_invokes_arbiter_on_collisions(self):
        """When broker returns collisions, route_task calls resolve_collisions
        and executes the resulting plan steps through fabric."""
        import asyncio
        from aurarouter.mcp_tools import route_task

        fabric = _make_fabric()
        cfg = fabric._config
        cfg.config.setdefault("catalog", {})

        bid_a, bid_b = _make_collision_pair()
        mock_broker_result = BrokerResult(
            bids=[bid_a, bid_b],
            collisions=[(bid_a, bid_b)],
            merged_plan=None,
            execution_trace=[],
        )

        arbiter_decision = ArbiterDecision(
            execution_order=[
                {"analyzer_id": "analyzer-alpha", "role": "coding", "tasks": ["do stuff"]},
            ],
            reasoning="alpha wins",
            strategy="winner_takes_all",
        )

        # Patch at the broker module level (where route_task imports from)
        with patch("aurarouter.broker.broadcast_to_analyzers") as mock_broadcast, \
             patch("aurarouter.broker.merge_bids", return_value=mock_broker_result), \
             patch("aurarouter.routing.resolve_collisions", return_value=arbiter_decision) as mock_resolve, \
             patch.object(fabric, "execute", return_value=GenerateResult(text="executed output")):

            # Make broadcast_to_analyzers return a coroutine that yields bids
            async def _fake_broadcast(*a, **kw):
                return [bid_a, bid_b]
            mock_broadcast.side_effect = _fake_broadcast

            result = route_task(
                fabric, None, task="refactor auth", config=cfg,
                options={"routing_hints": ["python"]},
            )

            # Verify resolve_collisions was called
            mock_resolve.assert_called_once()
            call_args = mock_resolve.call_args
            assert call_args[0][1] == "refactor auth"  # user_request
            assert result == "executed output"

    def test_arbiter_decision_execution_produces_output(self):
        """Verify that when arbiter returns steps, they are executed through fabric."""
        fabric = _make_fabric()
        broker_result = _make_broker_result_with_collisions()

        arbiter_response = {
            "execution_order": [
                {"analyzer_id": "a1", "role": "coding", "tasks": ["task1"]},
                {"analyzer_id": "a2", "role": "reasoning", "tasks": ["task2"]},
            ],
            "reasoning": "split work",
            "strategy": "sequential",
        }

        with patch.object(fabric, "execute", return_value=GenerateResult(text=json.dumps(arbiter_response))):
            decision = resolve_collisions(fabric, "build feature", broker_result)

        assert decision.strategy == "sequential"
        assert len(decision.execution_order) == 2
