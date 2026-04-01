"""Tests for AuraMonologue reasoning loop (TG10)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.monologue import MonologueOrchestrator, MonologueResult, ReasoningStep
from aurarouter.routing import should_use_monologue, TriageResult
from aurarouter.savings.models import GenerateResult


def _make_config(monologue_enabled=True):
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "gen-model": {"provider": "ollama", "model_name": "g", "endpoint": "http://x"},
            "crit-model": {"provider": "ollama", "model_name": "c", "endpoint": "http://x"},
            "ref-model": {"provider": "ollama", "model_name": "r", "endpoint": "http://x"},
        },
        "roles": {
            "reasoning": ["gen-model"],
            "reviewer": ["crit-model"],
            "coding": ["ref-model"],
            "router": ["gen-model"],
        },
        "system": {
            "monologue": monologue_enabled,
        },
    }
    return cfg


def _make_fabric(cfg=None):
    return ComputeFabric(cfg or _make_config())


def _make_orchestrator(fabric=None, sovereignty_gate=None, rag_pipeline=None):
    fb = fabric or _make_fabric()
    return MonologueOrchestrator(
        fabric=fb,
        mcp_registry=None,
        sovereignty_gate=sovereignty_gate,
        rag_pipeline=rag_pipeline,
    )


class TestMonologueOrchestrator:
    def test_is_enabled_reads_config(self):
        orch = _make_orchestrator()
        assert orch.is_enabled() is True

    def test_is_disabled_by_default(self):
        orch = _make_orchestrator(fabric=_make_fabric(_make_config(False)))
        assert orch.is_enabled() is False

    def test_select_experts_returns_models(self):
        orch = _make_orchestrator()
        gen, crit, ref = orch._select_experts()
        assert gen == "gen-model"
        assert crit == "crit-model"
        assert ref == "ref-model"

    def test_text_similarity_identical(self):
        orch = _make_orchestrator()
        assert orch._compute_text_similarity("hello world", "hello world") == 1.0

    def test_text_similarity_disjoint(self):
        orch = _make_orchestrator()
        assert orch._compute_text_similarity("hello world", "foo bar") == 0.0

    def test_text_similarity_partial(self):
        orch = _make_orchestrator()
        sim = orch._compute_text_similarity("hello world foo", "hello bar foo")
        # Jaccard: {hello, foo} / {hello, world, foo, bar} = 2/4 = 0.5
        assert abs(sim - 0.5) < 0.01


class TestReasonLoop:
    def test_full_loop_generator_critic_refiner(self):
        """Full reasoning loop with all three experts producing output."""
        orch = _make_orchestrator()

        gen_result = GenerateResult(
            text="Generated reasoning trace", model_id="gen-model",
            provider="ollama", input_tokens=10, output_tokens=20,
        )
        crit_result = GenerateResult(
            text=json.dumps({"score": 0.9, "feedback": "Excellent"}),
            model_id="crit-model", provider="ollama",
            input_tokens=10, output_tokens=20,
        )
        ref_result = GenerateResult(
            text="Refined final answer", model_id="ref-model",
            provider="ollama", input_tokens=10, output_tokens=20,
        )

        call_count = {"n": 0}

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            call_count["n"] += 1
            if role == "reasoning":
                return gen_result
            elif role == "reviewer":
                return crit_result
            elif role == "coding":
                return ref_result
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("Solve world peace", max_iterations=1))

        assert result.session_id
        assert result.convergence_reason == "confidence_threshold"
        assert result.total_iterations >= 1
        assert len(result.reasoning_trace) >= 2  # generator + critic at minimum

    def test_convergence_by_max_iterations(self):
        """Converges due to max iterations when critic never approves."""
        orch = _make_orchestrator()
        counter = {"n": 0}

        crit_result = GenerateResult(
            text=json.dumps({"score": 0.3, "feedback": "Needs work"}),
            model_id="crit-model", provider="ollama",
            input_tokens=10, output_tokens=20,
        )

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            counter["n"] += 1
            if role == "reasoning":
                # Each iteration produces very different text to avoid similarity convergence
                return GenerateResult(
                    text=f"completely unique reasoning {counter['n']} with many distinct words iteration alpha bravo charlie",
                    model_id="gen-model", provider="ollama",
                    input_tokens=10, output_tokens=20,
                )
            elif role == "reviewer":
                return crit_result
            elif role == "coding":
                return GenerateResult(
                    text=f"refined output {counter['n']} delta echo foxtrot golf hotel india juliet",
                    model_id="ref-model", provider="ollama",
                    input_tokens=10, output_tokens=20,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("Hard problem", max_iterations=3))

        assert result.convergence_reason == "max_iterations"
        assert result.total_iterations == 3

    def test_convergence_by_output_similarity(self):
        """Converges when consecutive generator outputs are very similar."""
        orch = _make_orchestrator()

        gen_outputs = iter([
            "Start reasoning about the problem",
            "Start reasoning about the problem nicely",
            "Start reasoning about the problem nicely",
        ])
        crit_result = GenerateResult(
            text=json.dumps({"score": 0.5, "feedback": "OK"}),
            model_id="crit-model", provider="ollama",
            input_tokens=10, output_tokens=20,
        )
        ref_result = GenerateResult(
            text="Refined", model_id="ref-model",
            provider="ollama", input_tokens=10, output_tokens=20,
        )

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                text = next(gen_outputs, "same text")
                return GenerateResult(
                    text=text, model_id="gen-model",
                    provider="ollama", input_tokens=10, output_tokens=20,
                )
            elif role == "reviewer":
                return crit_result
            elif role == "coding":
                return ref_result
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("X", max_iterations=5))

        assert result.convergence_reason in ("output_similarity", "max_iterations")

    def test_generator_failure_produces_empty_output(self):
        """When generator returns None, the result still completes."""
        orch = _make_orchestrator()

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("X", max_iterations=1))

        assert result.convergence_reason == "max_iterations"
        assert result.final_output == ""

    def test_no_models_available(self):
        """When no models exist, converges immediately."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {},
            "roles": {"reasoning": [], "reviewer": [], "coding": []},
            "system": {"monologue": True},
        }
        orch = _make_orchestrator(fabric=ComputeFabric(cfg))

        result = asyncio.run(orch.reason("task"))
        assert result.convergence_reason == "no_models_available"


class TestShouldUseMonologue:
    def test_enabled_complex_reasoning_high_complexity(self):
        fb = _make_fabric()
        tr = TriageResult(intent="COMPLEX_REASONING", complexity=8)
        assert should_use_monologue(fb, tr) is True

    def test_disabled_returns_false(self):
        fb = _make_fabric(_make_config(False))
        tr = TriageResult(intent="COMPLEX_REASONING", complexity=8)
        assert should_use_monologue(fb, tr) is False

    def test_wrong_intent_returns_false(self):
        fb = _make_fabric()
        tr = TriageResult(intent="SIMPLE_CODE", complexity=8)
        assert should_use_monologue(fb, tr) is False

    def test_low_complexity_returns_false(self):
        fb = _make_fabric()
        tr = TriageResult(intent="COMPLEX_REASONING", complexity=7)
        assert should_use_monologue(fb, tr) is False

    def test_complexity_exactly_8_returns_true(self):
        fb = _make_fabric()
        tr = TriageResult(intent="COMPLEX_REASONING", complexity=8)
        assert should_use_monologue(fb, tr) is True


class TestMonologueResult:
    def test_to_dict_roundtrip(self):
        result = MonologueResult(
            session_id="abc123",
            final_output="answer",
            total_iterations=3,
            convergence_reason="confidence_threshold",
            total_latency_ms=100.5,
            nodes_idled=2,
        )
        d = result.to_dict()
        assert d["session_id"] == "abc123"
        assert d["final_output"] == "answer"
        assert d["nodes_idled"] == 2

    def test_reasoning_step_to_dict(self):
        step = ReasoningStep(
            step_id="s1", role="generator", model_id="m1",
            input_prompt="p", output="o", iteration=1,
        )
        d = step.to_dict()
        assert d["step_id"] == "s1"
        assert d["role"] == "generator"
