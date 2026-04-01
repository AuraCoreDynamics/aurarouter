"""Tests for AuraMonologue convergence detection (TG10)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.monologue import MonologueOrchestrator
from aurarouter.savings.models import GenerateResult


def _make_config():
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
        "system": {"monologue": True},
    }
    return cfg


def _make_orchestrator():
    return MonologueOrchestrator(
        fabric=ComputeFabric(_make_config()),
        mcp_registry=None,
        sovereignty_gate=None,
        rag_pipeline=None,
    )


class TestConvergenceByThreshold:
    def test_high_critic_score_converges_immediately(self):
        """Critic score >= threshold causes immediate convergence."""
        orch = _make_orchestrator()

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.95, "feedback": "Great"}),
                    model_id="crit-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="final", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", convergence_threshold=0.9))

        assert result.convergence_reason == "confidence_threshold"
        assert result.total_iterations == 1

    def test_below_threshold_continues(self):
        """Critic score below threshold doesn't converge."""
        orch = _make_orchestrator()
        iteration_counter = {"n": 0}

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                iteration_counter["n"] += 1
                return GenerateResult(
                    text=f"trace-{iteration_counter['n']}", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.3, "feedback": "Needs work"}),
                    model_id="crit-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=2))

        assert result.convergence_reason == "max_iterations"


class TestConvergenceBySimilarity:
    def test_identical_outputs_converge(self):
        """Two identical generator outputs trigger similarity convergence."""
        orch = _make_orchestrator()

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="identical output every time", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.5, "feedback": "Same"}),
                    model_id="crit-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="identical output every time", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=5))

        assert result.convergence_reason == "output_similarity"


class TestCriticScoring:
    def test_invalid_json_from_critic_uses_zero(self):
        """Malformed critic JSON results in score=0.0."""
        orch = _make_orchestrator()

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text="not json at all", model_id="crit-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=1))

        assert result.convergence_reason == "max_iterations"

    def test_critic_none_result_continues(self):
        """When critic returns None, score is 0 and loop continues."""
        orch = _make_orchestrator()
        counter = {"n": 0}

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            counter["n"] += 1
            if role == "reasoning":
                return GenerateResult(
                    text=f"unique trace {counter['n']} alpha bravo charlie delta echo",
                    model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return None
            elif role == "coding":
                return GenerateResult(
                    text=f"unique refined {counter['n']} foxtrot golf hotel india juliet",
                    model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=2))

        assert result.convergence_reason == "max_iterations"
