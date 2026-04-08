"""T7.1 — Serialization Contract Validation.

Validates that _aura_routing_context serializes identically across Python
(RoutingContext dataclass) and C# (AuraRoutingContext sealed record with
[JsonPropertyName] attributes), satisfying the Cross-Sovereignty Reflection Gate.

CANONICAL 9-FIELD SCHEMA
─────────────────────────
  strategy              string
  confidence_score      float
  complexity_score      int
  selected_route        string
  analyzer_chain        array[string]
  intent                string
  hard_routed           bool
  simulated_cost_avoided float  (defaults to 0.0)
  metadata              object
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from aurarouter.analyzer_protocol import RoutingContext

# ── Canonical keys (ground truth) ────────────────────────────────────────────

CANONICAL_KEYS: list[str] = [
    "strategy",
    "confidence_score",
    "complexity_score",
    "selected_route",
    "analyzer_chain",
    "intent",
    "hard_routed",
    "simulated_cost_avoided",
    "metadata",
]

# Path to the C# AuraRoutingContext source file
_CS_FILE = Path(__file__).parents[4] / "auraxlm" / "src" / "AuraXLM.Abstractions" / "Models" / "AuraRoutingContext.cs"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_csharp_json_property_names(cs_source: str) -> list[str]:
    """Extract all [JsonPropertyName("...")] values from C# source."""
    return re.findall(r'\[JsonPropertyName\("([^"]+)"\)\]', cs_source)


def _make_canonical_context(**overrides) -> RoutingContext:
    defaults = dict(
        strategy="vector",
        confidence_score=0.92,
        complexity_score=2,
        selected_route="coding",
        analyzer_chain=["edge-complexity", "onnx-vector"],
        intent="SIMPLE_CODE",
        hard_routed=True,
        simulated_cost_avoided=0.00041,
    )
    defaults.update(overrides)
    return RoutingContext(**defaults)


# ── T7.1a: Python RoutingContext field names ──────────────────────────────────


class TestPythonRoutingContextSerialization:
    def test_all_canonical_keys_present_in_asdict(self):
        """`asdict()` output must contain exactly the 9 canonical keys."""
        ctx = _make_canonical_context()
        serialized = asdict(ctx)
        for key in CANONICAL_KEYS:
            assert key in serialized, f"Missing canonical key: {key!r}"

    def test_no_extra_keys_in_asdict(self):
        """No undocumented keys should be present in serialized output."""
        ctx = _make_canonical_context()
        serialized = asdict(ctx)
        extra = set(serialized.keys()) - set(CANONICAL_KEYS)
        assert not extra, f"Unexpected extra keys: {extra}"

    def test_strategy_is_string(self):
        ctx = _make_canonical_context(strategy="pipeline")
        assert isinstance(asdict(ctx)["strategy"], str)

    def test_confidence_score_is_float(self):
        ctx = _make_canonical_context(confidence_score=0.92)
        serialized = asdict(ctx)
        assert isinstance(serialized["confidence_score"], float)

    def test_complexity_score_is_int(self):
        ctx = _make_canonical_context(complexity_score=4)
        serialized = asdict(ctx)
        assert isinstance(serialized["complexity_score"], int)

    def test_hard_routed_is_bool(self):
        ctx = _make_canonical_context(hard_routed=True)
        serialized = asdict(ctx)
        assert isinstance(serialized["hard_routed"], bool)
        assert serialized["hard_routed"] is True

    def test_simulated_cost_avoided_is_float(self):
        ctx = _make_canonical_context(simulated_cost_avoided=0.00041)
        serialized = asdict(ctx)
        assert isinstance(serialized["simulated_cost_avoided"], float)
        assert abs(serialized["simulated_cost_avoided"] - 0.00041) < 1e-7

    def test_simulated_cost_avoided_default_is_zero(self):
        ctx = RoutingContext(
            strategy="vector",
            confidence_score=0.8,
            complexity_score=3,
            selected_route="coding",
            analyzer_chain=[],
            intent="SIMPLE_CODE",
        )
        assert ctx.simulated_cost_avoided == 0.0
        assert isinstance(ctx.simulated_cost_avoided, float)

    def test_analyzer_chain_is_list_of_strings(self):
        ctx = _make_canonical_context(analyzer_chain=["edge-complexity", "onnx-vector"])
        serialized = asdict(ctx)
        chain = serialized["analyzer_chain"]
        assert isinstance(chain, list)
        assert all(isinstance(s, str) for s in chain)

    def test_metadata_is_dict(self):
        ctx = _make_canonical_context()
        serialized = asdict(ctx)
        assert isinstance(serialized["metadata"], dict)

    def test_json_round_trip_preserves_all_values(self):
        ctx = _make_canonical_context()
        serialized = asdict(ctx)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        assert parsed["strategy"] == ctx.strategy
        assert abs(parsed["confidence_score"] - ctx.confidence_score) < 1e-6
        assert parsed["complexity_score"] == ctx.complexity_score
        assert parsed["hard_routed"] == ctx.hard_routed
        assert abs(parsed["simulated_cost_avoided"] - ctx.simulated_cost_avoided) < 1e-7
        assert parsed["analyzer_chain"] == ctx.analyzer_chain


# ── T7.1b: C# [JsonPropertyName] attributes match canonical schema ────────────


class TestCSharpJsonPropertyNames:
    @pytest.fixture(autouse=True)
    def _require_cs_file(self):
        if not _CS_FILE.exists():
            pytest.skip(f"C# source not found at {_CS_FILE}")

    def _get_cs_keys(self) -> list[str]:
        return _parse_csharp_json_property_names(_CS_FILE.read_text(encoding="utf-8"))

    def test_all_canonical_keys_present_in_csharp(self):
        """C# [JsonPropertyName] values must cover all 9 canonical keys."""
        cs_keys = self._get_cs_keys()
        for key in CANONICAL_KEYS:
            assert key in cs_keys, (
                f"Canonical key {key!r} not found in C# [JsonPropertyName] attributes. "
                f"Found: {cs_keys}"
            )

    def test_no_extra_keys_in_csharp(self):
        """C# should not declare undocumented JSON fields."""
        cs_keys = self._get_cs_keys()
        extra = set(cs_keys) - set(CANONICAL_KEYS)
        assert not extra, f"Unexpected extra C# JSON keys: {extra}"

    def test_all_keys_are_snake_case(self):
        """All JSON property names must be snake_case (no camelCase)."""
        cs_keys = self._get_cs_keys()
        for key in cs_keys:
            # snake_case: lowercase letters, underscores only (no uppercase)
            assert key == key.lower(), f"Key {key!r} is not lowercase/snake_case"
            assert re.match(r'^[a-z][a-z0-9_]*$', key), f"Key {key!r} is not valid snake_case"


# ── T7.1c: Round-trip: Python → JSON → C# format verification ────────────────


class TestRoundTripPythonToCSharpFormat:
    def test_round_trip_json_matches_csharp_schema(self):
        """Python JSON output should be deserializable with C# property names."""
        ctx = _make_canonical_context(
            strategy="pipeline",
            confidence_score=0.91,
            complexity_score=3,
            hard_routed=False,
            simulated_cost_avoided=0.0,
            metadata={"source": "test"},
        )
        serialized = asdict(ctx)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        # All canonical C# JSON property names should parse correctly
        assert parsed["strategy"] == "pipeline"
        assert abs(parsed["confidence_score"] - 0.91) < 1e-6
        assert parsed["complexity_score"] == 3
        assert parsed["selected_route"] == "coding"
        assert isinstance(parsed["analyzer_chain"], list)
        assert parsed["intent"] == "SIMPLE_CODE"
        assert parsed["hard_routed"] is False
        assert parsed["simulated_cost_avoided"] == 0.0
        assert isinstance(parsed["metadata"], dict)

    def test_simulated_cost_avoided_float_precision(self):
        """simulated_cost_avoided must survive JSON round-trip without int coercion."""
        ctx = _make_canonical_context(simulated_cost_avoided=0.00041)
        parsed = json.loads(json.dumps(asdict(ctx)))
        assert isinstance(parsed["simulated_cost_avoided"], float)
        # JSON may lose some precision but should not become 0
        assert parsed["simulated_cost_avoided"] > 0.0

    def test_hard_routed_false_serializes_as_false(self):
        """hard_routed=False must serialize to JSON false, not null or 0."""
        ctx = _make_canonical_context(hard_routed=False)
        json_str = json.dumps(asdict(ctx))
        assert '"hard_routed": false' in json_str

    def test_hard_routed_true_serializes_as_true(self):
        """hard_routed=True must serialize to JSON true, not 1 or 'true'."""
        ctx = _make_canonical_context(hard_routed=True)
        json_str = json.dumps(asdict(ctx))
        assert '"hard_routed": true' in json_str

    def test_empty_analyzer_chain_serializes_as_array(self):
        """Empty analyzer_chain must serialize as [] not null."""
        ctx = _make_canonical_context(analyzer_chain=[])
        parsed = json.loads(json.dumps(asdict(ctx)))
        assert parsed["analyzer_chain"] == []
        assert isinstance(parsed["analyzer_chain"], list)

    def test_metadata_default_serializes_as_object(self):
        """metadata default must serialize as {} not null or []."""
        ctx = RoutingContext(
            strategy="vector",
            confidence_score=0.9,
            complexity_score=2,
            selected_route="coding",
            analyzer_chain=[],
            intent="DIRECT",
        )
        parsed = json.loads(json.dumps(asdict(ctx)))
        assert parsed["metadata"] == {}
        assert isinstance(parsed["metadata"], dict)
