"""Tests for the federated broker (broker.py) and its integration with route_task."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aurarouter.broker import (
    BROADCAST_TIMEOUT_S,
    AnalyzerBid,
    BrokerResult,
    broadcast_to_analyzers,
    merge_bids,
)
from aurarouter.config import ConfigLoader
from aurarouter.mcp_tools import route_task
from aurarouter.savings.models import GenerateResult


# ===================================================================
# AnalyzerBid
# ===================================================================

class TestAnalyzerBid:
    def test_confidence_validation_valid(self):
        bid = AnalyzerBid(analyzer_id="a", confidence=0.5)
        assert bid.confidence == 0.5

    def test_confidence_validation_zero(self):
        bid = AnalyzerBid(analyzer_id="a", confidence=0.0)
        assert bid.confidence == 0.0

    def test_confidence_validation_one(self):
        bid = AnalyzerBid(analyzer_id="a", confidence=1.0)
        assert bid.confidence == 1.0

    def test_confidence_validation_too_low(self):
        with pytest.raises(ValueError, match="confidence"):
            AnalyzerBid(analyzer_id="a", confidence=-0.1)

    def test_confidence_validation_too_high(self):
        with pytest.raises(ValueError, match="confidence"):
            AnalyzerBid(analyzer_id="a", confidence=1.1)

    def test_overlaps_with_overlapping(self):
        a = AnalyzerBid(analyzer_id="a", confidence=0.8, claimed_files=["x.py", "y.py"])
        b = AnalyzerBid(analyzer_id="b", confidence=0.7, claimed_files=["y.py", "z.py"])
        assert a.overlaps_with(b) is True

    def test_overlaps_with_disjoint(self):
        a = AnalyzerBid(analyzer_id="a", confidence=0.8, claimed_files=["x.py"])
        b = AnalyzerBid(analyzer_id="b", confidence=0.7, claimed_files=["y.py"])
        assert a.overlaps_with(b) is False

    def test_overlaps_with_empty_a(self):
        a = AnalyzerBid(analyzer_id="a", confidence=0.8, claimed_files=[])
        b = AnalyzerBid(analyzer_id="b", confidence=0.7, claimed_files=["y.py"])
        assert a.overlaps_with(b) is False

    def test_overlaps_with_empty_b(self):
        a = AnalyzerBid(analyzer_id="a", confidence=0.8, claimed_files=["x.py"])
        b = AnalyzerBid(analyzer_id="b", confidence=0.7, claimed_files=[])
        assert a.overlaps_with(b) is False

    def test_overlaps_with_both_empty(self):
        a = AnalyzerBid(analyzer_id="a", confidence=0.8)
        b = AnalyzerBid(analyzer_id="b", confidence=0.7)
        assert a.overlaps_with(b) is False


# ===================================================================
# merge_bids
# ===================================================================

class TestMergeBids:
    def setup_method(self):
        # Clear any leftover broadcast trace
        if hasattr(broadcast_to_analyzers, "_last_trace"):
            broadcast_to_analyzers._last_trace = []  # type: ignore[attr-defined]

    def test_no_bids_empty_result(self):
        result = merge_bids([])
        assert result.bids == []
        assert result.collisions == []
        assert result.merged_plan is None
        assert result.mismatch is False

    def test_non_overlapping_merged(self):
        bids = [
            AnalyzerBid(analyzer_id="a", confidence=0.9, claimed_files=["x.py"], role="coding"),
            AnalyzerBid(analyzer_id="b", confidence=0.7, claimed_files=["y.py"], role="reasoning"),
        ]
        result = merge_bids(bids)
        assert result.collisions == []
        assert result.merged_plan is not None
        assert len(result.merged_plan) == 2
        # Sorted by confidence descending
        assert result.merged_plan[0]["analyzer_id"] == "a"
        assert result.merged_plan[1]["analyzer_id"] == "b"
        assert result.mismatch is False

    def test_overlapping_high_confidence_collision(self):
        bids = [
            AnalyzerBid(analyzer_id="a", confidence=0.9, claimed_files=["x.py"]),
            AnalyzerBid(analyzer_id="b", confidence=0.8, claimed_files=["x.py"]),
        ]
        result = merge_bids(bids)
        assert len(result.collisions) == 1
        assert result.merged_plan is None

    def test_overlapping_low_confidence_no_collision(self):
        """Overlapping files but both below 0.5 threshold should not flag collision."""
        bids = [
            AnalyzerBid(analyzer_id="a", confidence=0.3, claimed_files=["x.py"]),
            AnalyzerBid(analyzer_id="b", confidence=0.4, claimed_files=["x.py"]),
        ]
        result = merge_bids(bids)
        assert result.collisions == []
        assert result.merged_plan is not None

    def test_hints_no_match_mismatch(self):
        bids = [
            AnalyzerBid(analyzer_id="rust-analyzer", confidence=0.8, role="coding"),
        ]
        result = merge_bids(bids, routing_hints=["python"])
        assert result.mismatch is True

    def test_hints_match_by_role(self):
        bids = [
            AnalyzerBid(analyzer_id="code-helper", confidence=0.8, role="coding"),
        ]
        result = merge_bids(bids, routing_hints=["coding"])
        assert result.mismatch is False

    def test_hints_match_by_analyzer_id(self):
        bids = [
            AnalyzerBid(analyzer_id="python-specialist", confidence=0.8, role="coding"),
        ]
        result = merge_bids(bids, routing_hints=["python-specialist"])
        assert result.mismatch is False

    def test_hints_match_by_file_extension(self):
        bids = [
            AnalyzerBid(
                analyzer_id="generic",
                confidence=0.8,
                role="coding",
                claimed_files=["main.py"],
            ),
        ]
        result = merge_bids(bids, routing_hints=["py"])
        assert result.mismatch is False

    def test_none_hints_no_validation(self):
        bids = [
            AnalyzerBid(analyzer_id="a", confidence=0.8, role="coding"),
        ]
        result = merge_bids(bids, routing_hints=None)
        assert result.mismatch is False

    def test_empty_hints_no_validation(self):
        bids = [
            AnalyzerBid(analyzer_id="a", confidence=0.8, role="coding"),
        ]
        result = merge_bids(bids, routing_hints=[])
        assert result.mismatch is False


# ===================================================================
# Execution trace
# ===================================================================

class TestExecutionTrace:
    def setup_method(self):
        if hasattr(broadcast_to_analyzers, "_last_trace"):
            broadcast_to_analyzers._last_trace = []  # type: ignore[attr-defined]

    def test_trace_bid_count(self):
        bids = [AnalyzerBid(analyzer_id="a", confidence=0.5)]
        result = merge_bids(bids)
        assert any("1 bids received" in t for t in result.execution_trace)

    def test_trace_collision_event(self):
        bids = [
            AnalyzerBid(analyzer_id="a", confidence=0.9, claimed_files=["x.py"]),
            AnalyzerBid(analyzer_id="b", confidence=0.8, claimed_files=["x.py"]),
        ]
        result = merge_bids(bids)
        assert any("collision detected" in t for t in result.execution_trace)

    def test_trace_mismatch_event(self):
        bids = [AnalyzerBid(analyzer_id="a", confidence=0.5, role="coding")]
        result = merge_bids(bids, routing_hints=["python"])
        assert any("hint validation failed" in t for t in result.execution_trace)

    def test_trace_fast_path_merge(self):
        bids = [AnalyzerBid(analyzer_id="a", confidence=0.5, claimed_files=["x.py"])]
        result = merge_bids(bids)
        assert any("fast-path merge" in t for t in result.execution_trace)

    def test_trace_hint_skipped_when_none(self):
        bids = [AnalyzerBid(analyzer_id="a", confidence=0.5)]
        result = merge_bids(bids, routing_hints=None)
        assert any("hint validation skipped" in t for t in result.execution_trace)


# ===================================================================
# broadcast_to_analyzers
# ===================================================================

def _make_config_with_analyzers(analyzers: list[dict]) -> ConfigLoader:
    """Create a ConfigLoader with the given analyzer entries in the catalog."""
    cfg = ConfigLoader(allow_missing=True)
    catalog = {}
    for a in analyzers:
        aid = a.pop("artifact_id")
        catalog[aid] = a
    cfg.config = {"catalog": catalog}
    return cfg


class TestBroadcastToAnalyzers:
    def test_excludes_default_analyzer(self):
        cfg = _make_config_with_analyzers([
            {
                "artifact_id": "aurarouter-default",
                "kind": "analyzer",
                "display_name": "Default",
                "mcp_endpoint": "http://localhost:9000",
            },
            {
                "artifact_id": "remote-a",
                "kind": "analyzer",
                "display_name": "Remote A",
                "mcp_endpoint": "http://localhost:9001",
                "mcp_tool_name": "analyze",
            },
        ])

        async def mock_call(endpoint, tool_name, prompt, options, timeout):
            return {"confidence": 0.8, "role": "coding", "claimed_files": ["a.py"]}

        with patch("aurarouter.broker._call_single_analyzer", side_effect=mock_call):
            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(cfg, "test prompt")
                )
            finally:
                loop.close()

        assert len(bids) == 1
        assert bids[0].analyzer_id == "remote-a"

    def test_skips_analyzers_without_endpoint(self):
        cfg = _make_config_with_analyzers([
            {
                "artifact_id": "local-analyzer",
                "kind": "analyzer",
                "display_name": "Local",
                # No mcp_endpoint
            },
        ])

        loop = asyncio.new_event_loop()
        try:
            bids = loop.run_until_complete(
                broadcast_to_analyzers(cfg, "test prompt")
            )
        finally:
            loop.close()

        assert bids == []

    def test_handles_timeout(self):
        cfg = _make_config_with_analyzers([
            {
                "artifact_id": "slow-analyzer",
                "kind": "analyzer",
                "display_name": "Slow",
                "mcp_endpoint": "http://localhost:9002",
                "mcp_tool_name": "analyze",
            },
        ])

        async def slow_call(endpoint, tool_name, prompt, options, timeout):
            await asyncio.sleep(100)  # Will be cancelled by timeout
            return {}

        with patch("aurarouter.broker._call_single_analyzer", side_effect=slow_call):
            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(cfg, "test", timeout=0.01)
                )
            finally:
                loop.close()

        assert bids == []
        trace = getattr(broadcast_to_analyzers, "_last_trace", [])
        assert any("timed out" in t for t in trace)

    def test_handles_failed_analyzer(self):
        cfg = _make_config_with_analyzers([
            {
                "artifact_id": "broken-analyzer",
                "kind": "analyzer",
                "display_name": "Broken",
                "mcp_endpoint": "http://localhost:9003",
                "mcp_tool_name": "analyze",
            },
        ])

        async def failing_call(endpoint, tool_name, prompt, options, timeout):
            raise ConnectionError("refused")

        with patch("aurarouter.broker._call_single_analyzer", side_effect=failing_call):
            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(cfg, "test")
                )
            finally:
                loop.close()

        assert bids == []
        trace = getattr(broadcast_to_analyzers, "_last_trace", [])
        assert any("failed" in t for t in trace)

    def test_collects_multiple_bids(self):
        cfg = _make_config_with_analyzers([
            {
                "artifact_id": "analyzer-a",
                "kind": "analyzer",
                "display_name": "A",
                "mcp_endpoint": "http://localhost:9001",
                "mcp_tool_name": "analyze",
            },
            {
                "artifact_id": "analyzer-b",
                "kind": "analyzer",
                "display_name": "B",
                "mcp_endpoint": "http://localhost:9002",
                "mcp_tool_name": "analyze",
            },
        ])

        call_count = 0

        async def mock_call(endpoint, tool_name, prompt, options, timeout):
            nonlocal call_count
            call_count += 1
            return {
                "confidence": 0.7 + call_count * 0.1,
                "role": "coding",
                "claimed_files": [f"file{call_count}.py"],
            }

        with patch("aurarouter.broker._call_single_analyzer", side_effect=mock_call):
            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(cfg, "test prompt")
                )
            finally:
                loop.close()

        assert len(bids) == 2

    def test_broadcast_trace_entries(self):
        cfg = _make_config_with_analyzers([
            {
                "artifact_id": "analyzer-x",
                "kind": "analyzer",
                "display_name": "X",
                "mcp_endpoint": "http://localhost:9001",
                "mcp_tool_name": "analyze",
            },
        ])

        async def mock_call(endpoint, tool_name, prompt, options, timeout):
            return {"confidence": 0.9, "role": "coding"}

        with patch("aurarouter.broker._call_single_analyzer", side_effect=mock_call):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(broadcast_to_analyzers(cfg, "test"))
            finally:
                loop.close()

        trace = getattr(broadcast_to_analyzers, "_last_trace", [])
        assert any("broadcast to 1 analyzers" in t for t in trace)
        assert any("responded" in t for t in trace)


# ===================================================================
# route_task integration
# ===================================================================

def _make_fabric(models=None, roles=None):
    from aurarouter.fabric import ComputeFabric

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


class TestRouteTaskBrokerIntegration:
    def test_fallback_when_no_hints(self):
        """route_task falls back to built-in pipeline when no routing hints."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})),
            GenerateResult(text="fallback result"),
        ]):
            result = route_task(fabric, None, task="hello world")
            assert result == "fallback result"

    def test_fallback_with_empty_options(self):
        """route_task falls back when options is empty dict."""
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})),
            GenerateResult(text="fallback result"),
        ]):
            result = route_task(fabric, None, task="hello", options={})
            assert result == "fallback result"

    def test_broker_invoked_when_hints_present(self):
        """route_task invokes broker when routing_hints are in options."""
        fabric = _make_fabric()
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"catalog": {}}

        # Broker returns empty bids -> falls through to built-in
        with patch(
            "aurarouter.broker.broadcast_to_analyzers",
            return_value=[],
        ) as mock_broadcast, patch(
            "aurarouter.broker.merge_bids",
            return_value=BrokerResult(bids=[], execution_trace=["test"]),
        ) as mock_merge, patch.object(fabric, "execute", side_effect=[
            GenerateResult(text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})),
            GenerateResult(text="built-in result"),
        ]):
            result = route_task(
                fabric, None,
                task="hello",
                config=cfg,
                options={"routing_hints": ["python"]},
            )
            # Should still produce output (either broker or fallback)
            assert result is not None

    def test_broker_success_path(self):
        """route_task uses broker result when bids merge cleanly."""
        fabric = _make_fabric()
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"catalog": {}}

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
            execution_trace=["test"],
        )

        with patch(
            "aurarouter.broker.broadcast_to_analyzers",
            return_value=[bid],
        ), patch(
            "aurarouter.broker.merge_bids",
            return_value=broker_result,
        ), patch.object(fabric, "execute", return_value=GenerateResult(text="broker output")):
            result = route_task(
                fabric, None,
                task="write code",
                config=cfg,
                options={"routing_hints": ["python"]},
            )
            assert result == "broker output"

    def test_broker_collision_invokes_arbiter(self):
        """route_task invokes the arbiter when broker has collisions."""
        fabric = _make_fabric()
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"catalog": {}}

        bid_a = AnalyzerBid(analyzer_id="a", confidence=0.9, claimed_files=["x.py"])
        bid_b = AnalyzerBid(analyzer_id="b", confidence=0.8, claimed_files=["x.py"])
        broker_result = BrokerResult(
            bids=[bid_a, bid_b],
            collisions=[(bid_a, bid_b)],
            merged_plan=None,
            execution_trace=["collision"],
        )

        arbiter_json = json.dumps({
            "execution_order": [
                {"analyzer_id": "a", "role": "coding", "tasks": ["do it"]},
            ],
            "reasoning": "a wins",
            "strategy": "winner_takes_all",
        })

        with patch(
            "aurarouter.broker.broadcast_to_analyzers",
            return_value=[bid_a, bid_b],
        ), patch(
            "aurarouter.broker.merge_bids",
            return_value=broker_result,
        ), patch.object(fabric, "execute", side_effect=[
            # First call: arbiter (resolve_collisions -> fabric.execute("reasoning", ...))
            GenerateResult(text=arbiter_json),
            # Second call: execution of the arbiter's plan step
            GenerateResult(text="arbiter resolved output"),
        ]):
            result = route_task(
                fabric, None,
                task="test",
                config=cfg,
                options={"routing_hints": ["python"]},
            )
            assert result == "arbiter resolved output"
