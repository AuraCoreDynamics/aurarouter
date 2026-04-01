"""Tests for MAS-score-gated node idling in AuraMonologue (TG10)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

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


class TestMasScoreGating:
    def test_low_relevancy_generator_is_idled(self):
        """Generator with MAS relevancy below threshold is skipped."""
        orch = _make_orchestrator()

        # Force _score_anchor to return low score for generator
        async def low_score(anchor_id, prompt):
            if "gen-" in anchor_id:
                return 0.2  # Below 0.4 threshold
            return 1.0

        gen_result = GenerateResult(
            text="should not be called", model_id="gen-model",
            provider="ollama", input_tokens=5, output_tokens=5,
        )

        execute_called = {"n": 0}

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            execute_called["n"] += 1
            return gen_result

        with patch.object(orch, "_score_anchor", side_effect=low_score), \
             patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=1))

        assert result.nodes_idled >= 1

    def test_low_relevancy_critic_is_idled(self):
        """Critic with MAS relevancy below threshold is skipped."""
        orch = _make_orchestrator()

        async def selective_score(anchor_id, prompt):
            if "crit-" in anchor_id:
                return 0.1  # Below threshold
            return 1.0

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch, "_score_anchor", side_effect=selective_score), \
             patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=1))

        assert result.nodes_idled >= 1
        # Critic was idled, so no critic step in trace
        critic_steps = [s for s in result.reasoning_trace if s.role == "critic"]
        assert len(critic_steps) == 0

    def test_nodes_idled_counter_accumulates(self):
        """Idled count accumulates across multiple iterations."""
        orch = _make_orchestrator()

        async def low_gen_score(anchor_id, prompt):
            if "gen-" in anchor_id:
                return 0.1
            return 1.0

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            return None

        with patch.object(orch, "_score_anchor", side_effect=low_gen_score), \
             patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=3))

        # Generator idled each of 3 iterations
        assert result.nodes_idled == 3

    def test_high_relevancy_not_idled(self):
        """Experts with relevancy above threshold are NOT idled."""
        orch = _make_orchestrator()

        async def high_score(anchor_id, prompt):
            return 0.9  # Well above 0.4

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.9, "feedback": "Good"}),
                    model_id="crit-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch, "_score_anchor", side_effect=high_score), \
             patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=1))

        assert result.nodes_idled == 0

    def test_threshold_edge_exactly_at_boundary(self):
        """Expert with relevancy exactly at threshold is NOT idled (>= check)."""
        orch = _make_orchestrator()

        async def exact_threshold(anchor_id, prompt):
            return 0.4  # Exactly at threshold

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.9, "feedback": "OK"}),
                    model_id="crit-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch, "_score_anchor", side_effect=exact_threshold), \
             patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("task", max_iterations=1))

        assert result.nodes_idled == 0

    def test_custom_mas_threshold(self):
        """Custom mas_relevancy_threshold is respected."""
        orch = _make_orchestrator()

        async def moderate_score(anchor_id, prompt):
            return 0.6

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="gen-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.5, "feedback": "OK"}),
                    model_id="crit-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined", model_id="ref-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        # Threshold at 0.7, score is 0.6 → should idle
        with patch.object(orch, "_score_anchor", side_effect=moderate_score), \
             patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(
                orch.reason("task", max_iterations=1, mas_relevancy_threshold=0.7)
            )

        assert result.nodes_idled >= 1
