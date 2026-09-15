"""Tests for sovereignty enforcement in speculative decoding (TG7)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult
from aurarouter.sovereignty import SovereigntyGate, SovereigntyResult, SovereigntyVerdict
from aurarouter.speculative import SpeculativeOrchestrator


def _make_config():
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "local-3b": {
                "provider": "ollama",
                "model_name": "d",
                "endpoint": "http://x",
                "hosting_tier": "local",
            },
            "cloud-70b": {
                "provider": "openai",
                "model_name": "v",
                "endpoint": "http://x",
                "hosting_tier": "cloud",
            },
        },
        "roles": {
            "coding": ["local-3b", "cloud-70b"],
            "reasoning": ["cloud-70b", "local-3b"],
            "router": ["local-3b"],
        },
        "system": {
            "speculative_decoding": True,
            "speculative_complexity_threshold": 7,
            "sovereignty": True,
        },
    }
    return cfg


def _make_fabric(cfg=None):
    return ComputeFabric(cfg or _make_config())


class TestSovereigntyInSpeculative:
    def test_blocked_verdict_returns_error(self):
        """BLOCKED sovereignty verdict should prevent speculative execution."""
        fabric = _make_fabric()
        gate = SovereigntyGate(fabric.config)

        orch = SpeculativeOrchestrator(
            fabric=fabric,
            mcp_registry=None,
            sovereignty_gate=gate,
            triage_router=None,
        )

        # Mock sovereignty gate to return BLOCKED
        with patch.object(
            gate, "evaluate",
            return_value=SovereigntyResult(
                verdict=SovereigntyVerdict.BLOCKED,
                reason="classified content",
            ),
        ):
            result = asyncio.run(orch.execute_speculative("secret task"))

        assert result is not None
        assert result.get("error") == "blocked_by_sovereignty"

    def test_sovereign_verdict_filters_to_local(self):
        """SOVEREIGN verdict should filter both drafter and verifier to local models."""
        cfg = _make_config()
        fabric = _make_fabric(cfg)
        gate = SovereigntyGate(cfg)

        orch = SpeculativeOrchestrator(
            fabric=fabric,
            mcp_registry=None,
            sovereignty_gate=gate,
            triage_router=None,
        )

        with patch.object(
            gate, "evaluate",
            return_value=SovereigntyResult(
                verdict=SovereigntyVerdict.SOVEREIGN,
                reason="PII detected",
                matched_patterns=["ssn"],
            ),
        ), patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="local output", model_id="local-3b"),
        ) as mock_exec:
            result = asyncio.run(orch.execute_speculative("task with SSN 123-45-6789"))

        assert result is not None
        # All execute calls should use local model only
        for call in mock_exec.call_args_list:
            if "chain_override" in (call.kwargs or {}):
                chain = call.kwargs["chain_override"]
                for model_id in chain:
                    model_cfg = cfg.get_model_config(model_id)
                    assert model_cfg.get("hosting_tier") != "cloud", \
                        f"Cloud model {model_id} should be filtered in sovereign mode"

    def test_open_verdict_allows_all_models(self):
        """OPEN verdict should not filter any models."""
        fabric = _make_fabric()
        gate = SovereigntyGate(fabric.config)

        orch = SpeculativeOrchestrator(
            fabric=fabric,
            mcp_registry=None,
            sovereignty_gate=gate,
            triage_router=None,
        )

        with patch.object(
            gate, "evaluate",
            return_value=SovereigntyResult(verdict=SovereigntyVerdict.OPEN),
        ), patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="output", model_id="cloud-70b"),
        ):
            result = asyncio.run(orch.execute_speculative("harmless task"))

        assert result is not None

    def test_sovereignty_evaluated_for_both_models(self):
        """Sovereignty gate must be called to check the task prompt."""
        fabric = _make_fabric()
        gate = SovereigntyGate(fabric.config)

        orch = SpeculativeOrchestrator(
            fabric=fabric,
            mcp_registry=None,
            sovereignty_gate=gate,
            triage_router=None,
        )

        with patch.object(
            gate, "evaluate",
            return_value=SovereigntyResult(verdict=SovereigntyVerdict.OPEN),
        ) as mock_evaluate, patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="output", model_id="local-3b"),
        ):
            asyncio.run(orch.execute_speculative("test task"))

        # Sovereignty gate must have been called
        mock_evaluate.assert_called_once_with("test task")

    def test_no_sovereignty_gate_proceeds_normally(self):
        """Without a sovereignty gate, speculative execution should still work."""
        fabric = _make_fabric()
        orch = SpeculativeOrchestrator(
            fabric=fabric,
            mcp_registry=None,
            sovereignty_gate=None,
            triage_router=None,
        )

        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text="output", model_id="local-3b"),
        ):
            result = asyncio.run(orch.execute_speculative("task"))

        assert result is not None
        assert result["content"] == "output"
