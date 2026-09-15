"""Tests for EdgeComplexityScorer (TG3).

Verifies scoring behavior, performance, determinism, and protocol conformance.
Zero external dependencies.

TG3 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import time

import pytest

from aurarouter.analyzer_protocol import PromptAnalyzer
from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> EdgeComplexityScorer:
    return EdgeComplexityScorer()


# ---------------------------------------------------------------------------
# Complexity range tests
# ---------------------------------------------------------------------------


class TestComplexityScoring:
    """Verify score magnitudes match expectations."""

    def test_simple_prompt_scores_low(self, scorer: EdgeComplexityScorer) -> None:
        result = scorer.analyze("hello")
        assert result.complexity_score <= 2, f"Expected ≤2, got {result.complexity_score}"

    def test_hello_world_scores_low(self, scorer: EdgeComplexityScorer) -> None:
        result = scorer.analyze("hello world")
        assert result.complexity_score <= 3

    def test_complex_prompt_scores_high(self, scorer: EdgeComplexityScorer) -> None:
        prompt = (
            "Design a microservice architecture for a distributed transaction processing system. "
            "First, create the API gateway. Then integrate authentication and authorization. "
            "Finally, implement the saga pattern across all services. "
            "Do not use a monolith. Ensure eventual consistency without distributed locks. "
            "All components must handle concurrent failures gracefully."
        )
        result = scorer.analyze(prompt)
        assert result.complexity_score >= 7, f"Expected ≥7, got {result.complexity_score}"

    def test_medium_prompt_scores_mid(self, scorer: EdgeComplexityScorer) -> None:
        result = scorer.analyze("write a function to sort a list of integers")
        assert 2 <= result.complexity_score <= 6, f"Expected 2–6, got {result.complexity_score}"

    def test_code_blocks_increase_complexity(self, scorer: EdgeComplexityScorer) -> None:
        without_code = scorer.analyze("write a sorting function")
        with_code = scorer.analyze(
            "write a sorting function\n```python\ndef sort(items):\n    return sorted(items)\n```\n"
            "import itertools\nclass Sorter:\n    def run(self): pass"
        )
        assert with_code.complexity_score >= without_code.complexity_score

    def test_constraint_language_increases_complexity(self, scorer: EdgeComplexityScorer) -> None:
        simple = scorer.analyze("write a login function")
        constrained = scorer.analyze(
            "write a login function. "
            "Do not use bcrypt. Must not store passwords in plain text. "
            "Without using any third-party libraries. Ensure token refresh unless expired. "
            "Requires constant-time comparison."
        )
        assert constrained.complexity_score >= simple.complexity_score

    def test_multi_step_prompts_score_higher(self, scorer: EdgeComplexityScorer) -> None:
        simple = scorer.analyze("create a database")
        multi_step = scorer.analyze(
            "First, create the database schema. Then integrate the ORM. "
            "Next, write the migration scripts. Step 1: create tables. "
            "Step 2: add indexes. Finally, write integration tests."
        )
        assert multi_step.complexity_score > simple.complexity_score


# ---------------------------------------------------------------------------
# Return value guarantees
# ---------------------------------------------------------------------------


class TestReturnGuarantees:
    """Verify mandatory return value properties."""

    def test_analyze_always_returns_non_none(self, scorer: EdgeComplexityScorer) -> None:
        """Pre-filter must never abstain."""
        for prompt in ["", "hello", "a" * 5000]:
            result = scorer.analyze(prompt)
            assert result is not None, f"Returned None for prompt: {prompt[:20]!r}"

    def test_confidence_always_one(self, scorer: EdgeComplexityScorer) -> None:
        """Complexity measurement is always definitive."""
        result = scorer.analyze("some prompt")
        assert result.confidence == 1.0

    def test_complexity_within_bounds(self, scorer: EdgeComplexityScorer) -> None:
        """Complexity must be in [1, 10]."""
        prompts = [
            "",
            "hi",
            "write a function",
            "a" * 10000,
            "design distributed system architecture microservice integration api gateway",
        ]
        for p in prompts:
            r = scorer.analyze(p)
            assert 1 <= r.complexity_score <= 10, \
                f"Score {r.complexity_score} out of [1,10] for prompt: {p[:30]!r}"

    def test_empty_prompt(self, scorer: EdgeComplexityScorer) -> None:
        result = scorer.analyze("")
        assert result.complexity_score >= 1
        assert result.intent in ("DIRECT", "SIMPLE_CODE", "COMPLEX_REASONING")

    def test_analyzer_id(self, scorer: EdgeComplexityScorer) -> None:
        result = scorer.analyze("test")
        assert result.analyzer_id == "edge-complexity"

    def test_supports_always_true(self, scorer: EdgeComplexityScorer) -> None:
        for prompt in ["", "hello", "complex " * 100]:
            assert scorer.supports(prompt)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_deterministic(self, scorer: EdgeComplexityScorer) -> None:
        """Same prompt always produces same complexity score."""
        prompt = (
            "Refactor the authentication module to support OAuth2 and OpenID Connect. "
            "Integrate with the existing user service. Do not break existing sessions. "
            "First, create the OAuth2 provider interface. Then implement the adapters."
        )
        scores = {scorer.analyze(prompt).complexity_score for _ in range(5)}
        assert len(scores) == 1, f"Non-deterministic: got scores {scores}"


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_performance_under_1ms(self, scorer: EdgeComplexityScorer) -> None:
        """Scoring must complete in <1ms for any prompt."""
        prompt = "design a microservice architecture for distributed system"
        ITERATIONS = 100
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            scorer.analyze(prompt)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / ITERATIONS) * 1000
        assert avg_ms < 1.0, f"Average scoring time {avg_ms:.3f}ms exceeds 1ms"

    def test_very_long_prompt_no_performance_degradation(self, scorer: EdgeComplexityScorer) -> None:
        """10,000 char prompt still completes in <10ms."""
        long_prompt = ("design distributed microservice architecture " * 100)[:10000]
        start = time.perf_counter()
        scorer.analyze(long_prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10.0, f"Long prompt took {elapsed_ms:.3f}ms"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_conforms_to_protocol(self, scorer: EdgeComplexityScorer) -> None:
        assert isinstance(scorer, PromptAnalyzer)

    def test_priority_200(self, scorer: EdgeComplexityScorer) -> None:
        assert scorer.priority == 200

    def test_analyzer_id_correct(self, scorer: EdgeComplexityScorer) -> None:
        assert scorer.analyzer_id == "edge-complexity"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_custom_thresholds(self) -> None:
        scorer = EdgeComplexityScorer(simple_ceiling=2, complex_floor=8)
        # A medium prompt should suggest "SIMPLE_CODE" (conservative) when between thresholds
        result = scorer.analyze("write a function to count words in a string")
        assert 1 <= result.complexity_score <= 10
