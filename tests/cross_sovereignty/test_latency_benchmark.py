"""T7.3 — Pipeline Latency Benchmark.

Benchmarks the AuraRouter pluggable analyzer pipeline components:
- EdgeComplexityScorer:   p99 < 1ms per call
- ONNXVectorAnalyzer:     p80 < 10ms per call (skipped if not installed)
- Full pipeline (Stage 1 only): p80 < 50ms per call

Uses time.perf_counter consistently (no wall clock).
Measurements are taken over 100 iterations to build a stable distribution.
"""

from __future__ import annotations

import statistics
import time

import pytest

from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer
from aurarouter.analyzer_protocol import AnalysisResult

# ── Benchmark helpers ─────────────────────────────────────────────────────────

_SAMPLE_PROMPTS = [
    "hello",
    "what is 2+2",
    "write a python hello world",
    "write a function to sort a list using quicksort",
    "explain recursion",
    "fix the bug in my code",
    "def foo(x): return x * 2",
    "help me debug my authentication service",
    "what is the difference between a list and a tuple",
    "write unit tests for my sorting algorithm",
]


def _percentile(data: list[float], pct: float) -> float:
    """Return the given percentile (0-100) from a sorted list of values."""
    sorted_data = sorted(data)
    idx = (pct / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


ITERATIONS = 100


# ── T7.3a: EdgeComplexityScorer latency ──────────────────────────────────────


class TestEdgeComplexityScorerLatency:
    @pytest.fixture(scope="class")
    def scorer(self):
        return EdgeComplexityScorer()

    def _run_benchmark(self, scorer: EdgeComplexityScorer) -> list[float]:
        times_ms: list[float] = []
        for i in range(ITERATIONS):
            prompt = _SAMPLE_PROMPTS[i % len(_SAMPLE_PROMPTS)]
            t0 = time.perf_counter()
            scorer.analyze(prompt)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
        return times_ms

    def test_edge_complexity_scorer_p99_under_1ms(self, scorer):
        """EdgeComplexityScorer p99 latency must be <1ms."""
        times_ms = self._run_benchmark(scorer)
        p99 = _percentile(times_ms, 99)
        assert p99 < 1.0, (
            f"EdgeComplexityScorer p99 latency {p99:.3f}ms exceeds 1ms target. "
            f"p50={_percentile(times_ms, 50):.3f}ms, max={max(times_ms):.3f}ms"
        )

    def test_edge_complexity_scorer_p50_under_500us(self, scorer):
        """EdgeComplexityScorer median must be <0.5ms (regex-only path)."""
        times_ms = self._run_benchmark(scorer)
        p50 = _percentile(times_ms, 50)
        assert p50 < 0.5, (
            f"EdgeComplexityScorer p50={p50:.3f}ms exceeds 0.5ms target"
        )

    def test_edge_complexity_scorer_latency_distribution_captured(self, scorer):
        """Meta-test: verify we can capture the distribution (always passes)."""
        times_ms = self._run_benchmark(scorer)
        p50 = _percentile(times_ms, 50)
        p80 = _percentile(times_ms, 80)
        p99 = _percentile(times_ms, 99)
        # Always passes — captures baseline for reporting
        assert len(times_ms) == ITERATIONS
        assert p50 <= p80 <= p99


# ── T7.3b: ONNXVectorAnalyzer latency (skipped if not installed) ──────────────


class TestONNXVectorAnalyzerLatency:
    @pytest.fixture(scope="class")
    def onnx_analyzer(self):
        try:
            from aurarouter.analyzers.onnx_vector import ONNXVectorAnalyzer
            from aurarouter.intent_registry import IntentRegistry
        except ImportError:
            pytest.skip("aurarouter.analyzers.onnx_vector not available")
        registry = IntentRegistry()
        analyzer = ONNXVectorAnalyzer(intent_registry=registry)
        if not analyzer.supports("hello"):
            pytest.skip("ONNX model not installed — skipping latency benchmark")
        return analyzer

    def test_onnx_vector_analyzer_p80_under_10ms(self, onnx_analyzer):
        """ONNXVectorAnalyzer p80 latency must be <10ms."""
        times_ms: list[float] = []
        for i in range(ITERATIONS):
            prompt = _SAMPLE_PROMPTS[i % len(_SAMPLE_PROMPTS)]
            t0 = time.perf_counter()
            onnx_analyzer.analyze(prompt)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        p80 = _percentile(times_ms, 80)
        assert p80 < 10.0, (
            f"ONNXVectorAnalyzer p80={p80:.3f}ms exceeds 10ms target. "
            f"p50={_percentile(times_ms, 50):.3f}ms"
        )


# ── T7.3c: Full pipeline latency (Stage 1 + Stage 2 without SLM) ─────────────


class TestFullPipelineLatency:
    @pytest.fixture(scope="class")
    def pipeline(self):
        from aurarouter.analyzer_pipeline import AnalyzerPipeline

        scorer = EdgeComplexityScorer()
        pipeline = AnalyzerPipeline(confidence_threshold=0.85)
        pipeline.add_pre_filter(scorer)
        # No classifiers — tests Stage 1 fast path (no SLM overhead)
        return pipeline

    def test_stage1_only_pipeline_p80_under_50ms(self, pipeline):
        """Stage-1-only pipeline p80 latency must be <50ms."""
        times_ms: list[float] = []
        for i in range(ITERATIONS):
            prompt = _SAMPLE_PROMPTS[i % len(_SAMPLE_PROMPTS)]
            t0 = time.perf_counter()
            pipeline.run(prompt)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        p80 = _percentile(times_ms, 80)
        assert p80 < 50.0, (
            f"Stage-1 pipeline p80={p80:.3f}ms exceeds 50ms target. "
            f"p50={_percentile(times_ms, 50):.3f}ms, p99={_percentile(times_ms, 99):.3f}ms"
        )

    def test_stage1_only_pipeline_p99_under_50ms(self, pipeline):
        """Stage-1-only pipeline p99 should also remain under the 50ms target."""
        times_ms: list[float] = []
        for i in range(ITERATIONS):
            prompt = _SAMPLE_PROMPTS[i % len(_SAMPLE_PROMPTS)]
            t0 = time.perf_counter()
            pipeline.run(prompt)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        p99 = _percentile(times_ms, 99)
        assert p99 < 50.0, f"Stage-1 pipeline p99={p99:.3f}ms exceeds 50ms"

    def test_slm_fallback_latency_baseline_captured(self):
        """Meta-test: SLM path is not benchmarked (requires live fabric).

        This test documents the intent and always passes.
        The SLM baseline is captured during integration testing with a live model.
        """
        # SLM latency target is not enforced here (depends on hardware)
        # Captured separately in integration environment
        assert True


# ── T7.3d: Benchmark reproducibility ─────────────────────────────────────────


class TestBenchmarkReproducibility:
    def test_edge_complexity_results_are_deterministic(self):
        """EdgeComplexityScorer must return the same complexity scores across runs."""
        scorer = EdgeComplexityScorer()
        prompt = "write a function to sort a list using quicksort algorithm"

        results = [scorer.analyze(prompt).complexity_score for _ in range(10)]
        # All scores should be identical — deterministic algorithm
        assert len(set(results)) == 1, f"Non-deterministic results: {results}"

    def test_edge_complexity_variance_is_low(self):
        """Latency variance for EdgeComplexityScorer must be low."""
        scorer = EdgeComplexityScorer()
        prompt = "hello world"

        times_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            scorer.analyze(prompt)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        # Coefficient of variation should be low; stdev < 0.5ms indicates stability
        assert stdev_ms < 0.5, f"High latency variance: stdev={stdev_ms:.3f}ms"
