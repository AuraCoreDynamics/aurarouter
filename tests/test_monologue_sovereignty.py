"""Tests for AuraMonologue sovereignty enforcement (TG10)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.monologue import MonologueOrchestrator
from aurarouter.savings.models import GenerateResult
from aurarouter.sovereignty import SovereigntyGate, SovereigntyResult, SovereigntyVerdict


def _make_config():
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "local-model": {
                "provider": "ollama", "model_name": "l",
                "endpoint": "http://x", "hosting_tier": "local",
            },
            "cloud-model": {
                "provider": "openapi", "model_name": "c",
                "endpoint": "http://cloud", "hosting_tier": "cloud",
            },
        },
        "roles": {
            "reasoning": ["local-model", "cloud-model"],
            "reviewer": ["local-model"],
            "coding": ["local-model"],
            "router": ["local-model"],
        },
        "system": {
            "monologue": True,
            "sovereignty_enforcement": True,
        },
    }
    return cfg


def _make_fabric():
    return ComputeFabric(_make_config())


class TestSovereigntyEnforcement:
    def test_blocked_verdict_returns_immediately(self):
        """BLOCKED sovereignty verdict prevents any reasoning."""
        gate = MagicMock(spec=SovereigntyGate)
        gate.evaluate.return_value = SovereigntyResult(
            verdict=SovereigntyVerdict.BLOCKED,
            reason="PII detected, no local models",
            matched_patterns=["ssn"],
        )

        orch = MonologueOrchestrator(
            fabric=_make_fabric(),
            mcp_registry=None,
            sovereignty_gate=gate,
            rag_pipeline=None,
        )

        result = asyncio.run(orch.reason("task with SSN 123-45-6789"))
        assert result.convergence_reason == "sovereignty_blocked"
        assert result.final_output == ""
        assert result.total_iterations == 0

    def test_sovereign_verdict_filters_to_local_models(self):
        """SOVEREIGN verdict filters role chains to local-only models."""
        gate = MagicMock(spec=SovereigntyGate)
        gate.evaluate.return_value = SovereigntyResult(
            verdict=SovereigntyVerdict.SOVEREIGN,
            reason="Sensitive content detected",
            matched_patterns=["email"],
        )
        gate.enforce.side_effect = lambda chain, config, result: [
            m for m in chain if "local" in m
        ]

        orch = MonologueOrchestrator(
            fabric=_make_fabric(),
            mcp_registry=None,
            sovereignty_gate=gate,
            rag_pipeline=None,
        )

        gen, crit, ref = orch._select_experts(
            gate.evaluate.return_value
        )
        assert gen == "local-model"
        assert "cloud" not in (gen or "")

    def test_sovereign_all_experts_use_local(self):
        """All experts must be local when sovereignty requires it."""
        gate = MagicMock(spec=SovereigntyGate)
        gate.evaluate.return_value = SovereigntyResult(
            verdict=SovereigntyVerdict.SOVEREIGN,
            reason="Sensitive",
            matched_patterns=[],
        )
        gate.enforce.side_effect = lambda chain, config, result: [
            m for m in chain if "local" in m
        ]

        orch = MonologueOrchestrator(
            fabric=_make_fabric(),
            mcp_registry=None,
            sovereignty_gate=gate,
            rag_pipeline=None,
        )

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if chain_override:
                # Verify chain only contains local models
                for m in chain_override:
                    assert "cloud" not in m, f"Cloud model {m} in sovereign chain"
            if role == "reasoning":
                return GenerateResult(
                    text="local trace", model_id="local-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.9, "feedback": "OK"}),
                    model_id="local-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="refined locally", model_id="local-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("sensitive task", max_iterations=1))

        assert result.final_output
        gate.enforce.assert_called()

    def test_open_verdict_allows_all_models(self):
        """OPEN sovereignty verdict allows cloud models."""
        gate = MagicMock(spec=SovereigntyGate)
        gate.evaluate.return_value = SovereigntyResult(
            verdict=SovereigntyVerdict.OPEN,
            reason="No sensitive content",
            matched_patterns=[],
        )
        gate.enforce.side_effect = lambda chain, config, result: chain

        orch = MonologueOrchestrator(
            fabric=_make_fabric(),
            mcp_registry=None,
            sovereignty_gate=gate,
            rag_pipeline=None,
        )

        gen, crit, ref = orch._select_experts(gate.evaluate.return_value)
        # First in reasoning chain is local-model (it's listed first)
        assert gen == "local-model"

    def test_trace_never_contains_cloud_model_in_sovereign_mode(self):
        """Reasoning trace model_ids must all be local under sovereign verdict."""
        gate = MagicMock(spec=SovereigntyGate)
        gate.evaluate.return_value = SovereigntyResult(
            verdict=SovereigntyVerdict.SOVEREIGN,
            reason="PII",
            matched_patterns=["ssn"],
        )
        gate.enforce.side_effect = lambda chain, config, result: [
            m for m in chain if "local" in m
        ]

        orch = MonologueOrchestrator(
            fabric=_make_fabric(),
            mcp_registry=None,
            sovereignty_gate=gate,
            rag_pipeline=None,
        )

        def mock_execute(role, prompt, json_mode=False, chain_override=None):
            if role == "reasoning":
                return GenerateResult(
                    text="trace", model_id="local-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            elif role == "reviewer":
                return GenerateResult(
                    text=json.dumps({"score": 0.9, "feedback": "OK"}),
                    model_id="local-model", provider="ollama",
                    input_tokens=5, output_tokens=5,
                )
            elif role == "coding":
                return GenerateResult(
                    text="local refined", model_id="local-model",
                    provider="ollama", input_tokens=5, output_tokens=5,
                )
            return None

        with patch.object(orch._fabric, "execute", side_effect=mock_execute):
            result = asyncio.run(orch.reason("SSN 123-45-6789", max_iterations=1))

        for step in result.reasoning_trace:
            assert "cloud" not in step.model_id, \
                f"Cloud model {step.model_id} found in sovereign trace"
