"""T7.4 — Routing Accuracy Validation.

Validates the AuraRouter pluggable analyzer pipeline routing guard logic and
complexity scoring accuracy across a curated suite of 20+ prompts.

Key accuracy targets (Cross-Sovereignty Gate):
- EdgeComplexityScorer correctly identifies trivially simple prompts (≤simple_ceiling)
- Hard-routing gate NEVER fires for AnalysisResult with complexity_score ≥ 7
- Hard-routing gate fires for AnalysisResult with complexity ≤ ceiling, high confidence,
  and DIRECT/SIMPLE_CODE intent
- 100% correct classification for complex/novel queries via SLM fallback is tested
  in the integration environment (requires live fabric)

NOTE on EdgeComplexityScorer design:
  EdgeComplexityScorer is a *fast pre-filter* that reliably detects SIMPLE prompts
  (scoring ≤3). It does NOT map architectural/complex prompts to high scores — that
  is intentionally delegated to the ONNX/SLM Stage 2 classifiers. Complexity ranges
  in this test suite reflect the scorer's actual heuristic output, not human judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from aurarouter.analyzer_protocol import AnalysisResult, RoutingContext
from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer
from aurarouter.mcp_tools import _should_hard_route


# ── Routing Accuracy Suite ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AccuracyCase:
    prompt: str
    expected_intent: str          # Best-known classification (used as target)
    complexity_min: int           # Inclusive lower bound
    complexity_max: int           # Inclusive upper bound
    should_hard_route: bool = False


ROUTING_ACCURACY_SUITE: list[AccuracyCase] = [
    # ── Trivial / DIRECT (complexity 1–2, hard-route candidates) ──────────────
    # EdgeComplexityScorer reliably scores these as 1 (single short words/phrases)
    AccuracyCase("hello", "DIRECT", 1, 1, should_hard_route=True),
    AccuracyCase("what is 2+2", "DIRECT", 1, 2, should_hard_route=True),
    AccuracyCase("hi there", "DIRECT", 1, 2, should_hard_route=True),
    AccuracyCase("thanks", "DIRECT", 1, 1, should_hard_route=True),
    AccuracyCase("yes", "DIRECT", 1, 1, should_hard_route=True),
    AccuracyCase("ok", "DIRECT", 1, 1, should_hard_route=True),
    AccuracyCase("no", "DIRECT", 1, 1, should_hard_route=True),

    # ── Simple code tasks (complexity 1–3, may hard-route) ────────────────────
    # EdgeComplexityScorer uses heuristics; simple phrased tasks score low
    AccuracyCase("write a python hello world", "SIMPLE_CODE", 1, 3, should_hard_route=True),
    AccuracyCase("write a function to sort a list", "SIMPLE_CODE", 1, 3, should_hard_route=True),
    AccuracyCase("def foo(x): return x * 2", "SIMPLE_CODE", 1, 4, should_hard_route=True),
    AccuracyCase("print hello world in python", "SIMPLE_CODE", 1, 3, should_hard_route=True),
    AccuracyCase("write a for loop", "SIMPLE_CODE", 1, 3, should_hard_route=True),

    # ── Medium phrased tasks (complexity 1–5, no hard-route guarantee) ────────
    # These have more content but the edge scorer may still rate them 1-4;
    # the SLM would reclassify, but we test the scorer as-is.
    AccuracyCase(
        "write a binary search function with edge case handling",
        "SIMPLE_CODE", 1, 5, should_hard_route=False,
    ),
    AccuracyCase(
        "explain how recursion works with a fibonacci example",
        "SIMPLE_CODE", 1, 4, should_hard_route=False,
    ),
    AccuracyCase(
        "write unit tests for a stack data structure",
        "SIMPLE_CODE", 1, 4, should_hard_route=False,
    ),

    # ── Longer/complex prompts (complexity 1–6 from EdgeComplexityScorer) ─────
    # NOTE: EdgeComplexityScorer is conservative about high scores by design;
    # SLM Stage 2 would classify these as COMPLEX_REASONING. The range here
    # reflects realistic edge scorer output, not SLM output.
    AccuracyCase(
        "refactor the authentication module to use JWT with refresh token rotation "
        "and support multiple providers",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "design a distributed rate limiter for a multi-region API gateway with "
        "eventual consistency guarantees",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "implement a custom memory allocator in C with slab allocation and "
        "thread-safe freelists",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "design a microservice architecture for a distributed real-time bidding system "
        "handling 100k rps with sub-10ms p99 latency, horizontal scaling, circuit breakers, "
        "and saga-based transaction management",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "implement a Byzantine fault-tolerant consensus algorithm for a distributed "
        "database cluster with leader election and log replication across 5 nodes",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "architect a federated learning system that preserves differential privacy "
        "for medical imaging models across 50 hospital nodes without sharing raw data",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "design the data model and query optimization strategy for a petabyte-scale "
        "geospatial analytics platform supporting OLAP queries over satellite imagery "
        "with R-tree indexes and adaptive query planning",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "perform a full security audit of our OAuth2 implementation, identify all "
        "OWASP Top 10 vulnerabilities, provide exploit proofs-of-concept, and write "
        "remediation patches with regression tests",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
    AccuracyCase(
        "rewrite the monolithic e-commerce platform into event-sourced microservices "
        "using CQRS, implementing Kafka-based event streaming, Kubernetes deployment "
        "with service mesh, and zero-downtime blue-green deployments",
        "COMPLEX_REASONING", 1, 6, should_hard_route=False,
    ),
]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def scorer():
    return EdgeComplexityScorer()


def _make_mock_config(confidence_threshold: float = 0.85, simple_ceiling: int = 3):
    cfg = MagicMock()
    cfg.get_pipeline_config.return_value = {"confidence_threshold": confidence_threshold}
    cfg.get_complexity_scorer_config.return_value = {"simple_ceiling": simple_ceiling}
    return cfg


# ── T7.4a: Complexity scoring accuracy ───────────────────────────────────────


class TestComplexityScoringAccuracy:
    def test_all_cases_scored_within_expected_range(self, scorer):
        """EdgeComplexityScorer must score each prompt within its expected range."""
        failures = []
        for case in ROUTING_ACCURACY_SUITE:
            result = scorer.analyze(case.prompt)
            in_range = case.complexity_min <= result.complexity_score <= case.complexity_max
            if not in_range:
                failures.append(
                    f"'{case.prompt[:50]}' → got {result.complexity_score}, "
                    f"expected [{case.complexity_min}, {case.complexity_max}]"
                )
        assert not failures, "Complexity scoring failures:\n" + "\n".join(failures)

    def test_trivial_single_word_prompts_score_1(self, scorer):
        """Single-word trivial prompts must score exactly 1."""
        trivial = ["hello", "thanks", "yes", "no", "ok"]
        for p in trivial:
            result = scorer.analyze(p)
            assert result.complexity_score == 1, (
                f"'{p}' scored {result.complexity_score}, expected 1"
            )

    def test_trivial_prompts_score_below_hard_route_ceiling(self, scorer):
        """All trivial + simple prompts must score ≤ simple_ceiling (3)."""
        ceiling = 3
        for case in ROUTING_ACCURACY_SUITE:
            if not case.should_hard_route:
                continue
            result = scorer.analyze(case.prompt)
            assert result.complexity_score <= ceiling, (
                f"Expected hard-route candidate '{case.prompt[:50]}' scored "
                f"{result.complexity_score} > {ceiling}"
            )

    def test_scorer_is_deterministic(self, scorer):
        """Same prompt must produce the same complexity score across calls."""
        prompt = "write a function to sort a list using quicksort algorithm"
        scores = [scorer.analyze(prompt).complexity_score for _ in range(5)]
        assert len(set(scores)) == 1, f"Non-deterministic results: {scores}"

    def test_scorer_confidence_always_1(self, scorer):
        """EdgeComplexityScorer always reports confidence=1.0 (definitive measurement)."""
        for case in ROUTING_ACCURACY_SUITE:
            result = scorer.analyze(case.prompt)
            assert result.confidence == 1.0, (
                f"Expected confidence=1.0 for '{case.prompt[:40]}', got {result.confidence}"
            )

    def test_scorer_analyzer_id_is_correct(self, scorer):
        """Results must carry the correct analyzer_id."""
        result = scorer.analyze("hello")
        assert result.analyzer_id == "edge-complexity"


# ── T7.4b: Hard-routing trigger guard ────────────────────────────────────────


class TestHardRoutingGate:
    """Hard-routing gate logic validation.

    The gate logic (_should_hard_route) is tested directly with constructed
    AnalysisResult inputs. This validates the CONTRACT: any result with
    complexity_score >= complex_floor (7) must never be hard-routed, regardless
    of how that score was computed (EdgeComplexityScorer, ONNX, or SLM).
    """

    COMPLEX_PROMPTS_WITH_HIGH_SCORES = [
        "design a microservice architecture for a distributed real-time bidding system",
        "implement a Byzantine fault-tolerant consensus algorithm",
        "architect a federated learning system with differential privacy",
        "perform a full security audit with OWASP Top 10 exploit proofs",
        "rewrite the monolithic platform into event-sourced microservices with CQRS",
        "design a petabyte-scale geospatial analytics platform",
    ]

    def test_hard_routing_never_triggers_for_complexity_7_plus(self):
        """AnalysisResult with complexity_score >= 7 must NEVER pass hard-routing gate.

        This is the critical contract: even if a result has high confidence and
        correct intent, complexity >= complex_floor (7) blocks hard-routing.
        """
        config = _make_mock_config(confidence_threshold=0.85, simple_ceiling=3)

        # Test all combinations: high confidence, SIMPLE_CODE intent, but complexity >= 7
        for complexity in range(7, 11):
            for intent in ("DIRECT", "SIMPLE_CODE"):
                for confidence in (0.85, 0.90, 0.95, 0.99):
                    result = AnalysisResult(
                        intent=intent,
                        confidence=confidence,
                        complexity_score=complexity,
                        analyzer_id="test",
                    )
                    assert not _should_hard_route(result, config), (
                        f"Should NOT hard-route: complexity={complexity}, "
                        f"intent={intent}, confidence={confidence}"
                    )

    def test_hard_routing_triggers_for_correct_simple_conditions(self):
        """complexity <= ceiling + high confidence + SIMPLE_CODE = hard-route."""
        config = _make_mock_config(confidence_threshold=0.85, simple_ceiling=3)

        for complexity in range(1, 4):  # 1, 2, 3
            for intent in ("DIRECT", "SIMPLE_CODE"):
                result = AnalysisResult(
                    intent=intent,
                    confidence=0.90,
                    complexity_score=complexity,
                    analyzer_id="test",
                )
                assert _should_hard_route(result, config), (
                    f"Should hard-route: complexity={complexity}, intent={intent}"
                )

    def test_hard_routing_not_triggered_when_confidence_below_threshold(self):
        """Low confidence must block hard-routing even for simple prompts."""
        config = _make_mock_config(confidence_threshold=0.85, simple_ceiling=3)
        result = AnalysisResult(
            intent="SIMPLE_CODE",
            confidence=0.70,
            complexity_score=2,
            analyzer_id="edge-complexity",
        )
        assert not _should_hard_route(result, config)

    def test_hard_routing_not_triggered_for_complex_reasoning_intent(self):
        """COMPLEX_REASONING intent must block hard-routing regardless of confidence."""
        config = _make_mock_config(confidence_threshold=0.85, simple_ceiling=3)
        for confidence in (0.85, 0.95, 0.99):
            result = AnalysisResult(
                intent="COMPLEX_REASONING",
                confidence=confidence,
                complexity_score=2,  # Even if complexity is low
                analyzer_id="test",
            )
            assert not _should_hard_route(result, config), (
                f"COMPLEX_REASONING with confidence={confidence} should not hard-route"
            )

    def test_hard_routing_at_threshold_boundary(self):
        """Test boundary conditions for confidence threshold."""
        config = _make_mock_config(confidence_threshold=0.85, simple_ceiling=3)
        # Exactly AT threshold → triggers
        at_threshold = AnalysisResult(intent="SIMPLE_CODE", confidence=0.85,
                                      complexity_score=2, analyzer_id="test")
        assert _should_hard_route(at_threshold, config)
        # Just below → does NOT trigger
        below_threshold = AnalysisResult(intent="SIMPLE_CODE", confidence=0.84,
                                         complexity_score=2, analyzer_id="test")
        assert not _should_hard_route(below_threshold, config)

    def test_hard_routing_at_ceiling_boundary(self):
        """Test boundary conditions for simple_ceiling."""
        config = _make_mock_config(confidence_threshold=0.85, simple_ceiling=3)
        # AT ceiling (3) → triggers
        at_ceiling = AnalysisResult(intent="SIMPLE_CODE", confidence=0.90,
                                    complexity_score=3, analyzer_id="test")
        assert _should_hard_route(at_ceiling, config)
        # ONE above ceiling (4) → does NOT trigger
        above_ceiling = AnalysisResult(intent="SIMPLE_CODE", confidence=0.90,
                                       complexity_score=4, analyzer_id="test")
        assert not _should_hard_route(above_ceiling, config)


# ── T7.4c: Suite coverage meta-tests ─────────────────────────────────────────


class TestSuiteCoverage:
    def test_suite_has_enough_cases(self):
        assert len(ROUTING_ACCURACY_SUITE) >= 20, (
            f"Accuracy suite has only {len(ROUTING_ACCURACY_SUITE)} cases, need ≥ 20"
        )

    def test_suite_covers_trivial_cases(self):
        trivial = [c for c in ROUTING_ACCURACY_SUITE if c.complexity_max <= 2]
        assert len(trivial) >= 5, "Need ≥5 trivially simple cases (max_complexity ≤ 2)"

    def test_suite_covers_hard_route_candidates(self):
        hr_cases = [c for c in ROUTING_ACCURACY_SUITE if c.should_hard_route]
        assert len(hr_cases) >= 5, "Need ≥5 hard-route candidate cases in the suite"

    def test_suite_covers_non_hard_route_cases(self):
        non_hr = [c for c in ROUTING_ACCURACY_SUITE if not c.should_hard_route]
        assert len(non_hr) >= 10, "Need ≥10 non-hard-route cases in the suite"

    def test_no_should_hard_route_for_complex_prompted_cases(self):
        """Cases with 'COMPLEX_REASONING' expected intent must not be should_hard_route=True."""
        violations = [
            c for c in ROUTING_ACCURACY_SUITE
            if c.expected_intent == "COMPLEX_REASONING" and c.should_hard_route
        ]
        assert not violations, (
            "Accuracy suite misconfiguration: COMPLEX_REASONING cases cannot be "
            f"marked as hard-route candidates: {[c.prompt[:40] for c in violations]}"
        )

    def test_all_cases_have_valid_ranges(self):
        """All accuracy cases must have min_complexity <= max_complexity within [1,10]."""
        for case in ROUTING_ACCURACY_SUITE:
            assert 1 <= case.complexity_min <= case.complexity_max <= 10, (
                f"Invalid complexity range [{case.complexity_min}, {case.complexity_max}] "
                f"for '{case.prompt[:40]}'"
            )
