"""Tests for the Pluggable Analyzer Pipeline (TG1).

Tests the PromptAnalyzer protocol, AnalysisResult, RoutingContext,
AnalyzerPipeline two-stage execution, SLMIntentAnalyzer adapter,
AnalyzerRegistry, and pipeline configuration.

TG1 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.analyzer_pipeline import AnalyzerPipeline
from aurarouter.analyzer_protocol import AnalysisResult, PromptAnalyzer, RoutingContext
from aurarouter.analyzers import AnalyzerRegistry, create_default_analyzer
from aurarouter.config import ConfigLoader
from aurarouter.intent_registry import IntentDefinition, IntentRegistry


# ---------------------------------------------------------------------------
# Helpers — Fake analyzers that implement PromptAnalyzer structurally
# ---------------------------------------------------------------------------


class _FakePreFilter:
    """A Stage 1 pre-filter that always returns a fixed complexity score."""

    analyzer_id = "fake-prefilter"
    priority = 100

    def __init__(self, complexity: int = 4) -> None:
        self._complexity = complexity
        self.calls: list[str] = []

    def supports(self, prompt: str) -> bool:
        return True

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult:
        self.calls.append(prompt)
        return AnalysisResult(
            intent="DIRECT",
            confidence=1.0,
            complexity_score=self._complexity,
            analyzer_id=self.analyzer_id,
        )


class _FakeClassifier:
    """A Stage 2 classifier with configurable confidence and intent."""

    def __init__(
        self,
        analyzer_id: str = "fake-clf",
        priority: int = 50,
        confidence: float = 0.90,
        intent: str = "SIMPLE_CODE",
        supports_result: bool = True,
        complexity_score: int = 0,  # Sentinel — classifiers don't own complexity
    ) -> None:
        self._id = analyzer_id
        self._priority = priority
        self._confidence = confidence
        self._intent = intent
        self._supports_result = supports_result
        self._complexity_score = complexity_score
        self.calls: list[str] = []
        self.supports_calls: list[str] = []

    @property
    def analyzer_id(self) -> str:
        return self._id

    @property
    def priority(self) -> int:
        return self._priority

    def supports(self, prompt: str) -> bool:
        self.supports_calls.append(prompt)
        return self._supports_result

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult | None:
        self.calls.append(prompt)
        return AnalysisResult(
            intent=self._intent,
            confidence=self._confidence,
            complexity_score=self._complexity_score,
            analyzer_id=self._id,
        )


class _AbstainingClassifier:
    """A classifier that always abstains (returns None)."""

    analyzer_id = "abstainer"
    priority = 70

    def supports(self, prompt: str) -> bool:
        return True

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult | None:
        return None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestPromptAnalyzerProtocol:
    """Verify structural protocol conformance (runtime_checkable)."""

    def test_fake_prefilter_conforms_to_protocol(self) -> None:
        pf = _FakePreFilter()
        assert isinstance(pf, PromptAnalyzer)

    def test_fake_classifier_conforms_to_protocol(self) -> None:
        clf = _FakeClassifier()
        assert isinstance(clf, PromptAnalyzer)

    def test_abstaining_classifier_conforms_to_protocol(self) -> None:
        a = _AbstainingClassifier()
        assert isinstance(a, PromptAnalyzer)

    def test_bare_object_does_not_conform(self) -> None:
        assert not isinstance(object(), PromptAnalyzer)


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """AnalysisResult is a frozen dataclass."""

    def test_frozen(self) -> None:
        r = AnalysisResult(intent="DIRECT", confidence=0.9, complexity_score=2, analyzer_id="x")
        with pytest.raises((AttributeError, TypeError)):
            r.intent = "OTHER"  # type: ignore[misc]

    def test_defaults(self) -> None:
        r = AnalysisResult(intent="X", confidence=0.5, complexity_score=5, analyzer_id="y")
        assert r.reasoning == ""
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# RoutingContext
# ---------------------------------------------------------------------------


class TestRoutingContext:
    """RoutingContext is a frozen dataclass with the correct field set."""

    def test_required_fields(self) -> None:
        ctx = RoutingContext(
            strategy="vector",
            confidence_score=0.92,
            complexity_score=2,
            selected_route="coding",
            analyzer_chain=["edge-complexity", "onnx-vector"],
            intent="SIMPLE_CODE",
        )
        assert ctx.hard_routed is False
        assert ctx.simulated_cost_avoided == 0.0
        assert ctx.metadata == {}

    def test_all_fields_present(self) -> None:
        ctx = RoutingContext(
            strategy="slm",
            confidence_score=0.95,
            complexity_score=6,
            selected_route="reasoning",
            analyzer_chain=["edge-complexity", "slm-intent"],
            intent="COMPLEX_REASONING",
            hard_routed=True,
            simulated_cost_avoided=0.00041,
            metadata={"extra": "data"},
        )
        assert ctx.hard_routed is True
        assert ctx.simulated_cost_avoided == pytest.approx(0.00041)
        assert ctx.metadata["extra"] == "data"

    def test_frozen(self) -> None:
        ctx = RoutingContext(
            strategy="s", confidence_score=0.5, complexity_score=1,
            selected_route="coding", analyzer_chain=[], intent="DIRECT",
        )
        with pytest.raises((AttributeError, TypeError)):
            ctx.hard_routed = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AnalyzerPipeline — Stage 1 execution
# ---------------------------------------------------------------------------


class TestPipelinePreFilterStage:
    """Verify Stage 1 (pre-filter) behavior."""

    def test_pipeline_pre_filter_always_runs(self) -> None:
        """Pre-filter must run even when classifier short-circuits at high confidence."""
        pf = _FakePreFilter(complexity=5)
        clf = _FakeClassifier(confidence=0.99)  # Ultra-high confidence

        pipeline = AnalyzerPipeline(pre_filters=[pf], classifiers=[clf])
        result = pipeline.run("test prompt")

        assert pf.calls, "Pre-filter was not called"
        assert clf.calls, "Classifier was not called even in Stage 2"

    def test_complexity_score_from_prefilter_survives_intent_shortcircuit(self) -> None:
        """When ONNX exits at 0.92 confidence, complexity comes from EdgeComplexityScorer."""
        pf = _FakePreFilter(complexity=3)   # Stage 1 value: 3
        clf = _FakeClassifier(confidence=0.92, complexity_score=8)  # Would set 8 if not merged

        pipeline = AnalyzerPipeline(pre_filters=[pf], classifiers=[clf], confidence_threshold=0.85)
        result = pipeline.run("hello world")

        # complexity_score must be 3 (from pre-filter), not 8 (from classifier)
        assert result.complexity_score == 3

    def test_prefilter_runs_even_when_supports_false_would_skip_classifier(self) -> None:
        """Pre-filter always runs; supports() gate is irrelevant for pre-filters."""
        pf = _FakePreFilter(complexity=2)
        clf = _FakeClassifier(supports_result=False)  # classifier is gated by supports()

        pipeline = AnalyzerPipeline(pre_filters=[pf], classifiers=[clf])
        result = pipeline.run("test")

        assert pf.calls  # pre-filter ran
        assert clf.supports_calls  # supports() was checked
        assert not clf.calls  # but analyze() was not called (unsupported)


# ---------------------------------------------------------------------------
# AnalyzerPipeline — Stage 2 execution
# ---------------------------------------------------------------------------


class TestPipelineClassifierStage:
    """Verify Stage 2 (classifier) behavior."""

    def test_pipeline_classifiers_short_circuit_on_confidence(self) -> None:
        """When first classifier >= threshold, second classifier does NOT run."""
        high_clf = _FakeClassifier(analyzer_id="high", priority=100, confidence=0.95)
        low_clf = _FakeClassifier(analyzer_id="low", priority=0, confidence=0.60)

        pipeline = AnalyzerPipeline(classifiers=[high_clf, low_clf], confidence_threshold=0.85)
        pipeline.run("test")

        assert high_clf.calls
        assert not low_clf.calls, "Second classifier should have been skipped"

    def test_pipeline_falls_through_to_slm(self) -> None:
        """If high-priority classifier has low confidence, low-priority SLM runs."""
        onnx_clf = _FakeClassifier(analyzer_id="onnx", priority=100, confidence=0.60)  # Below threshold
        slm_clf = _FakeClassifier(analyzer_id="slm", priority=0, confidence=0.95)

        pipeline = AnalyzerPipeline(classifiers=[onnx_clf, slm_clf], confidence_threshold=0.85)
        result = pipeline.run("complex task")

        assert onnx_clf.calls
        assert slm_clf.calls
        assert result.analyzer_id == "slm"

    def test_pipeline_returns_fallback_with_measured_complexity(self) -> None:
        """All classifiers abstain → DIRECT intent but complexity from pre-filter."""
        pf = _FakePreFilter(complexity=6)

        pipeline = AnalyzerPipeline(pre_filters=[pf], classifiers=[_AbstainingClassifier()])
        result = pipeline.run("ambiguous task")

        assert result.intent == "DIRECT"           # Fallback intent
        assert result.complexity_score == 6         # From pre-filter Stage 1
        assert result.analyzer_id == "pipeline-fallback"

    def test_supports_false_skips_classifier(self) -> None:
        """supports() returning False prevents analyze() from being called."""
        blocked = _FakeClassifier(analyzer_id="blocked", supports_result=False)

        pipeline = AnalyzerPipeline(classifiers=[blocked])
        pipeline.run("test")

        assert blocked.supports_calls  # supports() was polled
        assert not blocked.calls       # analyze() was not called

    def test_best_partial_returned_when_no_shortcircuit(self) -> None:
        """When no classifier meets threshold, return best (highest confidence) partial."""
        clf_a = _FakeClassifier(analyzer_id="a", priority=100, confidence=0.70)
        clf_b = _FakeClassifier(analyzer_id="b", priority=0, confidence=0.60)

        pipeline = AnalyzerPipeline(classifiers=[clf_a, clf_b], confidence_threshold=0.85)
        result = pipeline.run("moderate task")

        assert result.analyzer_id == "a"  # a had higher confidence (0.70 > 0.60)


# ---------------------------------------------------------------------------
# AnalyzerPipeline — routing context
# ---------------------------------------------------------------------------


class TestPipelineRoutingContext:
    """Verify RoutingContext construction."""

    def test_routing_context_construction(self) -> None:
        pf = _FakePreFilter(complexity=2)
        clf = _FakeClassifier(intent="SIMPLE_CODE", confidence=0.92)

        pipeline = AnalyzerPipeline(pre_filters=[pf], classifiers=[clf], confidence_threshold=0.85)
        result = pipeline.run("write hello world")
        ctx = pipeline.build_routing_context(result, selected_route="coding")

        assert ctx.intent == "SIMPLE_CODE"
        assert ctx.selected_route == "coding"
        assert ctx.complexity_score == 2
        assert isinstance(ctx.analyzer_chain, list)
        assert ctx.hard_routed is False
        assert ctx.simulated_cost_avoided == 0.0

    def test_routing_context_analyzer_chain_ordered(self) -> None:
        """analyzer_chain must list pre-filters before classifiers."""
        pf = _FakePreFilter()
        clf = _FakeClassifier()

        pipeline = AnalyzerPipeline(pre_filters=[pf], classifiers=[clf])
        pipeline.run("test")
        ctx = pipeline.build_routing_context(
            AnalysisResult(intent="X", confidence=0.9, complexity_score=1, analyzer_id="x"),
            selected_route="coding",
        )
        # pre-filter ID should appear before classifier ID
        chain = ctx.analyzer_chain
        assert "fake-prefilter" in chain
        assert "fake-clf" in chain
        assert chain.index("fake-prefilter") < chain.index("fake-clf")


# ---------------------------------------------------------------------------
# SLMIntentAnalyzer adapter
# ---------------------------------------------------------------------------


class TestSLMAdapterMapsTriage:
    """Verify TriageResult → AnalysisResult mapping in SLMIntentAnalyzer."""

    def _make_registry(self) -> IntentRegistry:
        r = IntentRegistry()
        return r

    def test_slm_adapter_maps_triage_result(self) -> None:
        from aurarouter.analyzers.slm_analyzer import SLMIntentAnalyzer
        from aurarouter.routing import TriageResult

        mock_fabric = MagicMock()
        registry = self._make_registry()

        slm = SLMIntentAnalyzer(fabric=mock_fabric, intent_registry=registry)
        assert slm.analyzer_id == "slm-intent"
        assert slm.priority == 0
        assert slm.supports("any prompt")

        with patch("aurarouter.routing.analyze_intent") as mock_ai:
            mock_ai.return_value = TriageResult(intent="SIMPLE_CODE", complexity=3)
            result = slm.analyze("write code")

        assert result is not None
        assert result.intent == "SIMPLE_CODE"
        assert result.confidence == 0.95
        assert result.complexity_score == 0  # Sentinel — pre-filter owns complexity
        assert result.analyzer_id == "slm-intent"
        assert "triage_complexity" in result.metadata
        assert result.metadata["triage_complexity"] == 3

    def test_slm_adapter_returns_none_on_exception(self) -> None:
        from aurarouter.analyzers.slm_analyzer import SLMIntentAnalyzer

        mock_fabric = MagicMock()
        registry = self._make_registry()
        slm = SLMIntentAnalyzer(fabric=mock_fabric, intent_registry=registry)

        with patch("aurarouter.routing.analyze_intent", side_effect=RuntimeError("boom")):
            result = slm.analyze("test")

        assert result is None


# ---------------------------------------------------------------------------
# AnalyzerRegistry
# ---------------------------------------------------------------------------


class TestAnalyzerRegistry:
    """Verify AnalyzerRegistry discovery and pipeline construction."""

    def _make_config_with_catalog(self, catalog: dict) -> ConfigLoader:
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"catalog": catalog}
        return cfg

    def test_analyzer_registry_skips_unknown_kinds(self, caplog) -> None:
        cfg = self._make_config_with_catalog({
            "weird-analyzer": {
                "kind": "analyzer",
                "analyzer_kind": "alien_brain",
                "display_name": "Alien",
            }
        })
        registry = AnalyzerRegistry(cfg)
        with caplog.at_level(logging.WARNING):
            discovered = registry.discover()
        assert discovered == []
        assert any("alien_brain" in r.message for r in caplog.records)

    def test_analyzer_registry_skips_moe_ranking_silently(self, caplog) -> None:
        """moe_ranking is a remote kind — no warning expected."""
        cfg = self._make_config_with_catalog({
            "xlm": {
                "kind": "analyzer",
                "analyzer_kind": "moe_ranking",
                "display_name": "XLM",
            }
        })
        registry = AnalyzerRegistry(cfg)
        with caplog.at_level(logging.WARNING):
            discovered = registry.discover()
        assert discovered == []
        # No warning about moe_ranking
        assert not any("moe_ranking" in r.message for r in caplog.records)

    def test_analyzer_registry_discovers_edge_complexity(self) -> None:
        cfg = self._make_config_with_catalog({
            "edge-complexity": {
                "kind": "analyzer",
                "analyzer_kind": "edge_complexity",
                "display_name": "Edge Complexity",
            }
        })
        registry = AnalyzerRegistry(cfg)
        discovered = registry.discover()
        assert len(discovered) == 1
        assert discovered[0].analyzer_id == "edge-complexity"

    def test_analyzer_registry_builds_two_stage_pipeline(self) -> None:
        """EdgeComplexityScorer in pre-filters, SLM in classifiers."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        registry = AnalyzerRegistry(cfg)

        mock_fabric = MagicMock()
        mock_registry = IntentRegistry()

        pipeline = registry.build_pipeline(
            fabric=mock_fabric,
            intent_registry=mock_registry,
            confidence_threshold=0.85,
        )

        # EdgeComplexityScorer should be in pre-filters
        pre_filter_ids = [a.analyzer_id for a in pipeline.pre_filters]
        assert "edge-complexity" in pre_filter_ids

        # SLM should be in classifiers
        classifier_ids = [a.analyzer_id for a in pipeline.classifiers]
        assert "slm-intent" in classifier_ids

    def test_analyzer_registry_manual_register(self) -> None:
        cfg = ConfigLoader(allow_missing=True)
        registry = AnalyzerRegistry(cfg)
        fake = _FakeClassifier(analyzer_id="my-analyzer")
        registry.register(fake)
        assert registry.get("my-analyzer") is fake

    def test_analyzer_registry_builds_pipeline_without_fabric(self) -> None:
        """Without fabric, SLM is excluded (it needs fabric to call models)."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        registry = AnalyzerRegistry(cfg)

        pipeline = registry.build_pipeline(fabric=None, intent_registry=None)

        # SLM should NOT be in classifiers (no fabric)
        classifier_ids = [a.analyzer_id for a in pipeline.classifiers]
        assert "slm-intent" not in classifier_ids


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Verify ConfigLoader.get_pipeline_config() and get_complexity_scorer_config()."""

    def test_get_pipeline_config_defaults(self) -> None:
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        pc = cfg.get_pipeline_config()
        assert pc["enabled"] is False
        assert pc["confidence_threshold"] == 0.85

    def test_get_pipeline_config_from_yaml(self) -> None:
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {
                "analyzer_pipeline": {
                    "enabled": True,
                    "confidence_threshold": 0.90,
                }
            }
        }
        pc = cfg.get_pipeline_config()
        assert pc["enabled"] is True
        assert pc["confidence_threshold"] == 0.90

    def test_get_complexity_scorer_config_defaults(self) -> None:
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        cc = cfg.get_complexity_scorer_config()
        assert cc["simple_ceiling"] == 3
        assert cc["complex_floor"] == 7

    def test_get_complexity_scorer_config_override(self) -> None:
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {
                "analyzer_pipeline": {
                    "edge_complexity": {
                        "simple_ceiling": 2,
                        "complex_floor": 8,
                    }
                }
            }
        }
        cc = cfg.get_complexity_scorer_config()
        assert cc["simple_ceiling"] == 2
        assert cc["complex_floor"] == 8


# ---------------------------------------------------------------------------
# create_default_analyzer backwards compat
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_create_default_analyzer_still_available(self) -> None:
        from aurarouter.analyzers import create_default_analyzer
        artifact = create_default_analyzer()
        assert artifact.artifact_id == "aurarouter-default"
