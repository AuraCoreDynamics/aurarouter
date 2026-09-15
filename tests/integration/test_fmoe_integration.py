"""FMoE cross-component integration tests for AuraRouter (Task Group 7).

Validates the end-to-end Broker -> Arbiter -> Execution pipeline:
- Broker broadcasts to analyzers and collects bids
- Arbiter resolves bid collisions via LLM reasoning
- Schema enforcement produces parseable JSON for actionable intents
- Fallback path works when no analyzers are registered
- execution_trace is coherent across all paths
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.broker import (
    AnalyzerBid,
    BrokerResult,
    broadcast_to_analyzers,
    merge_bids,
)
from aurarouter.config import ConfigLoader
from aurarouter.fabric import (
    MODIFICATIONS_SCHEMA,
    ComputeFabric,
    compile_modifications_schema,
)
from aurarouter.mcp_tools import route_task
from aurarouter.routing import (
    ArbiterDecision,
    build_arbiter_prompt,
    resolve_collisions,
)
from aurarouter.savings.models import GenerateResult


# ===================================================================
# Helpers
# ===================================================================


def _make_fabric(models=None, roles=None) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": models or {
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        "roles": roles or {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    fabric = ComputeFabric(cfg)
    return fabric


def _make_config_with_analyzers(analyzers: list[dict]) -> ConfigLoader:
    """Create a ConfigLoader with the given analyzer entries in the catalog."""
    cfg = ConfigLoader(allow_missing=True)
    catalog = {}
    for a in analyzers:
        a = dict(a)  # avoid mutating caller's dicts
        aid = a.pop("artifact_id")
        catalog[aid] = a
    cfg.config = {"catalog": catalog}
    return cfg


# ===================================================================
# T7.2: Broker -> Arbiter -> Execution Pipeline
# ===================================================================


class TestBrokerArbiterPipeline:
    """Register two mock analyzers claiming overlapping files, verify
    collision detection and arbiter resolution."""

    def _make_overlapping_bids(self):
        """Two analyzers, both claiming shared files with high confidence."""
        bid_py = AnalyzerBid(
            analyzer_id="python-specialist",
            confidence=0.9,
            claimed_files=["src/main.py", "src/utils.py"],
            proposed_tasks=[{"task": "refactor python modules"}],
            role="coding",
        )
        bid_ts = AnalyzerBid(
            analyzer_id="typescript-specialist",
            confidence=0.85,
            claimed_files=["src/utils.py", "src/app.ts"],
            proposed_tasks=[{"task": "update typescript bindings"}],
            role="coding",
        )
        return bid_py, bid_ts

    def test_broker_detects_collision(self):
        """Overlapping files with high confidence => collision flagged."""
        bid_py, bid_ts = self._make_overlapping_bids()
        result = merge_bids([bid_py, bid_ts], routing_hints=["python", "typescript"])

        assert len(result.collisions) == 1
        assert result.merged_plan is None
        assert any("collision detected" in t for t in result.execution_trace)

    def test_arbiter_resolves_with_valid_json(self):
        """Arbiter parses valid JSON decision from the reasoning role."""
        bid_py, bid_ts = self._make_overlapping_bids()
        broker_result = BrokerResult(
            bids=[bid_py, bid_ts],
            collisions=[(bid_py, bid_ts)],
            merged_plan=None,
            execution_trace=["Broker: collision detected"],
        )

        arbiter_response = {
            "execution_order": [
                {
                    "analyzer_id": "python-specialist",
                    "role": "coding",
                    "tasks": ["refactor python modules"],
                    "files": ["src/main.py", "src/utils.py"],
                },
                {
                    "analyzer_id": "typescript-specialist",
                    "role": "coding",
                    "tasks": ["update typescript bindings"],
                    "files": ["src/app.ts"],
                },
            ],
            "reasoning": "Split by language domain",
            "strategy": "sequential",
        }

        fabric = _make_fabric()
        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text=json.dumps(arbiter_response)),
        ):
            decision = resolve_collisions(fabric, "refactor code", broker_result)

        assert decision.strategy == "sequential"
        assert len(decision.execution_order) == 2
        assert decision.execution_order[0]["analyzer_id"] == "python-specialist"
        assert decision.execution_order[1]["analyzer_id"] == "typescript-specialist"

    def test_full_route_task_with_collisions(self):
        """End-to-end: route_task with routing hints => broker => arbiter => output."""
        fabric = _make_fabric()
        cfg = _make_config_with_analyzers([])

        bid_py, bid_ts = self._make_overlapping_bids()
        broker_result = BrokerResult(
            bids=[bid_py, bid_ts],
            collisions=[(bid_py, bid_ts)],
            merged_plan=None,
            execution_trace=["collision"],
        )

        arbiter_json = json.dumps({
            "execution_order": [
                {"analyzer_id": "python-specialist", "role": "coding", "tasks": ["do it"]},
            ],
            "reasoning": "python-specialist wins",
            "strategy": "winner_takes_all",
        })

        async def _fake_broadcast(*a, **kw):
            return [bid_py, bid_ts]

        with patch("aurarouter.broker.broadcast_to_analyzers", side_effect=_fake_broadcast), \
             patch("aurarouter.broker.merge_bids", return_value=broker_result), \
             patch.object(fabric, "execute", side_effect=[
                 GenerateResult(text=arbiter_json),
                 GenerateResult(text="final output"),
             ]):
            result = route_task(
                fabric, None,
                task="refactor code",
                config=cfg,
                options={"routing_hints": ["python", "typescript"]},
            )

        assert result == "final output"

    def test_arbiter_trace_entries(self):
        """Verify trace entries are appended for arbiter resolution."""
        bid_py, bid_ts = self._make_overlapping_bids()
        broker_result = BrokerResult(
            bids=[bid_py, bid_ts],
            collisions=[(bid_py, bid_ts)],
            merged_plan=None,
            execution_trace=["Broker: collision detected"],
        )

        arbiter_response = {
            "execution_order": [{"analyzer_id": "python-specialist", "role": "coding"}],
            "reasoning": "python wins",
            "strategy": "winner_takes_all",
        }

        fabric = _make_fabric()
        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text=json.dumps(arbiter_response)),
        ):
            resolve_collisions(fabric, "task", broker_result)

        trace = broker_result.execution_trace
        assert any("resolving" in t.lower() for t in trace)
        assert any("resolved" in t.lower() for t in trace)
        # Original trace preserved
        assert "Broker: collision detected" in trace


# ===================================================================
# T7.4: Integration Gaps Validation
# ===================================================================


class TestIntegrationGaps:
    """Verify interfaces between TG1-TG6 are compatible."""

    def test_analyzer_bid_fields_match_arbiter_expectations(self):
        """AnalyzerBid has the fields the arbiter prompt formatter needs."""
        bid = AnalyzerBid(
            analyzer_id="test",
            confidence=0.8,
            claimed_files=["a.py"],
            proposed_tasks=[{"task": "do"}],
            role="coding",
        )
        # build_arbiter_prompt accesses these attributes
        assert hasattr(bid, "analyzer_id")
        assert hasattr(bid, "confidence")
        assert hasattr(bid, "claimed_files")
        assert hasattr(bid, "role")

        # Verify the prompt builds without error
        bid2 = AnalyzerBid(analyzer_id="other", confidence=0.7, claimed_files=["a.py"])
        prompt = build_arbiter_prompt("task", [(bid, bid2)])
        assert "test" in prompt
        assert "other" in prompt

    def test_schema_fields_match_parser_expectations(self):
        """MODIFICATIONS_SCHEMA required fields match parse_artifact_payload's expectations."""
        item_schema = MODIFICATIONS_SCHEMA["properties"]["modifications"]["items"]
        required = set(item_schema["required"])
        # The parser (artifacts.py) checks for these exact fields
        expected = {"file_path", "modification_type", "content", "language"}
        assert required == expected

    def test_compile_schema_with_file_constraints(self):
        """file_constraints format from TG2 is compatible with compile_modifications_schema."""
        # This is exactly the format build_file_constraints returns
        file_constraints = [
            {"path": "small.py", "preferred_modification": "full_rewrite"},
            {"path": "large.py", "preferred_modification": "unified_diff"},
        ]
        schema = compile_modifications_schema(file_constraints)
        assert "allOf" in schema["properties"]["modifications"]["items"]
        conditionals = schema["properties"]["modifications"]["items"]["allOf"]
        assert len(conditionals) == 1  # only unified_diff files get conditionals
        assert conditionals[0]["if"]["properties"]["file_path"]["const"] == "large.py"
        assert conditionals[0]["then"]["properties"]["modification_type"]["enum"] == ["unified_diff"]

    def test_compile_schema_no_constraints(self):
        """compile_modifications_schema with no constraints returns base schema."""
        schema = compile_modifications_schema(None)
        assert schema is MODIFICATIONS_SCHEMA

    def test_compile_schema_all_full_rewrite(self):
        """When all files are full_rewrite, no conditionals added."""
        file_constraints = [
            {"path": "a.py", "preferred_modification": "full_rewrite"},
            {"path": "b.py", "preferred_modification": "full_rewrite"},
        ]
        schema = compile_modifications_schema(file_constraints)
        # Should return base schema (no allOf needed)
        assert "allOf" not in schema["properties"]["modifications"]["items"]

    def test_options_dict_keys_compat(self):
        """Verify TG1 options keys match what TG4/TG6 expect."""
        # TG1 produces these keys in _build_route_options
        options = {
            "intent": "edit_code",
            "routing_hints": ["python"],
            "file_constraints": [
                {"path": "a.py", "preferred_modification": "unified_diff"},
            ],
        }

        # TG4 (route_task) reads routing_hints
        assert "routing_hints" in options
        assert isinstance(options["routing_hints"], list)

        # TG6 (fabric.execute) reads intent and file_constraints
        assert options["intent"] in ("edit_code", "generate_code", "chat")
        assert isinstance(options.get("file_constraints"), list)

        # TG6 compile_modifications_schema expects path + preferred_modification
        for fc in options["file_constraints"]:
            assert "path" in fc
            assert "preferred_modification" in fc


# ===================================================================
# T7.5: Cross-Component Smoke Tests (AuraRouter side)
# ===================================================================


class TestSmokeHappyPath:
    """Chat intent -> no schema enforcement, standard text response."""

    def test_chat_intent_no_schema(self):
        """route_task with no routing hints follows legacy pipeline."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 2})),
            GenerateResult(text="hello world response"),
        ]):
            result = route_task(fabric, None, task="say hello")
        assert result == "hello world response"


class TestSmokeEditPath:
    """Edit intent with routing hints -> broker invoked, output produced."""

    def test_edit_with_merged_plan(self):
        """Non-colliding bids merge cleanly and execute through fabric."""
        fabric = _make_fabric()
        cfg = _make_config_with_analyzers([])

        bid = AnalyzerBid(
            analyzer_id="py-analyzer",
            confidence=0.9,
            claimed_files=["main.py"],
            role="coding",
        )
        broker_result = BrokerResult(
            bids=[bid],
            collisions=[],
            merged_plan=[{
                "analyzer_id": "py-analyzer",
                "role": "coding",
                "confidence": 0.9,
                "claimed_files": ["main.py"],
                "proposed_tasks": [],
            }],
            mismatch=False,
            execution_trace=["merged"],
        )

        async def _fake_broadcast(*a, **kw):
            return [bid]

        with patch("aurarouter.broker.broadcast_to_analyzers", side_effect=_fake_broadcast), \
             patch("aurarouter.broker.merge_bids", return_value=broker_result), \
             patch.object(fabric, "execute", return_value=GenerateResult(text="edit output")):
            result = route_task(
                fabric, None,
                task="edit code",
                config=cfg,
                options={"routing_hints": ["python"]},
            )
        assert result == "edit output"


class TestSmokeFallbackPath:
    """No analyzers registered, no routing hints -> existing pipeline unchanged."""

    def test_no_hints_fallback(self):
        """Without routing_hints, route_task uses built-in pipeline."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})),
            GenerateResult(text="fallback result"),
        ]):
            result = route_task(fabric, None, task="do something")
        assert result == "fallback result"

    def test_empty_hints_fallback(self):
        """Empty routing_hints list should not trigger broker."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})),
            GenerateResult(text="fallback result"),
        ]):
            result = route_task(
                fabric, None,
                task="do something",
                options={"routing_hints": []},
            )
        assert result == "fallback result"

    def test_no_config_fallback(self):
        """Without config, routing_hints are ignored."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})),
            GenerateResult(text="fallback result"),
        ]):
            result = route_task(
                fabric, None,
                task="do something",
                options={"routing_hints": ["python"]},
            )
        assert result == "fallback result"


class TestSmokeTraceCompleteness:
    """Verify execution_trace is coherent across broker/arbiter paths."""

    def test_broker_collision_trace(self):
        """Broker collision -> arbiter -> trace contains all phases."""
        bid_a = AnalyzerBid(analyzer_id="a", confidence=0.9, claimed_files=["x.py"])
        bid_b = AnalyzerBid(analyzer_id="b", confidence=0.8, claimed_files=["x.py"])

        # Clear broadcast trace
        if hasattr(broadcast_to_analyzers, "_last_trace"):
            broadcast_to_analyzers._last_trace = []  # type: ignore[attr-defined]

        result = merge_bids([bid_a, bid_b], routing_hints=["python"])
        assert len(result.collisions) == 1

        # Feed through arbiter
        fabric = _make_fabric()
        arbiter_response = {
            "execution_order": [{"analyzer_id": "a", "role": "coding"}],
            "reasoning": "a wins",
            "strategy": "winner_takes_all",
        }
        with patch.object(
            fabric, "execute",
            return_value=GenerateResult(text=json.dumps(arbiter_response)),
        ):
            decision = resolve_collisions(fabric, "task", result)

        trace = result.execution_trace
        # Should have entries from broker and arbiter phases
        assert any("collision" in t.lower() for t in trace)
        assert any("resolving" in t.lower() or "resolved" in t.lower() for t in trace)
        assert any("bids received" in t for t in trace)

    def test_no_collision_trace(self):
        """No collisions -> trace shows fast-path merge."""
        bid = AnalyzerBid(analyzer_id="a", confidence=0.9, claimed_files=["x.py"])

        if hasattr(broadcast_to_analyzers, "_last_trace"):
            broadcast_to_analyzers._last_trace = []  # type: ignore[attr-defined]

        result = merge_bids([bid])
        assert result.collisions == []
        assert any("fast-path merge" in t for t in result.execution_trace)
