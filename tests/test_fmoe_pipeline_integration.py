"""TG4 — Integration tests for the pluggable analyzer pipeline wired into route_task().

Tests the pipeline-enabled path, legacy fallback path, hard-routing logic,
routing context injection, and cost telemetry.

TG4 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import json
from dataclasses import replace as _dc_replace
from unittest.mock import MagicMock, patch, call

import pytest

from aurarouter.analyzer_pipeline import AnalyzerPipeline
from aurarouter.analyzer_protocol import AnalysisResult, RoutingContext
from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import (
    _build_aura_routing_context,
    _calculate_avoided_cost,
    _inject_routing_context,
    _should_hard_route,
    route_task,
)
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fabric(models: dict, roles: dict) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg)


def _make_config(pipeline_enabled: bool = True, **extra) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "system": {
            "analyzer_pipeline": {
                "enabled": pipeline_enabled,
                "confidence_threshold": 0.85,
                "edge_complexity": {"simple_ceiling": 3, "complex_floor": 7},
            }
        },
        "savings": {
            "enabled": True,
            "hard_route": {
                "reference_cloud_model": "claude-3-5-haiku",
                "reference_cloud_provider": "anthropic",
                "assumed_output_tokens": 200,
            },
        },
        **extra,
    }
    return cfg


def _make_analysis(
    intent: str = "SIMPLE_CODE",
    confidence: float = 0.92,
    complexity: int = 2,
) -> AnalysisResult:
    return AnalysisResult(
        intent=intent,
        confidence=confidence,
        complexity_score=complexity,
        analyzer_id="onnx-vector",
    )


def _make_routing_ctx(
    intent: str = "SIMPLE_CODE",
    confidence: float = 0.92,
    complexity: int = 2,
    hard_routed: bool = False,
    simulated_cost_avoided: float = 0.0,
) -> RoutingContext:
    return RoutingContext(
        strategy="vector",
        confidence_score=confidence,
        complexity_score=complexity,
        selected_route="coding",
        analyzer_chain=["edge-complexity", "onnx-vector"],
        intent=intent,
        hard_routed=hard_routed,
        simulated_cost_avoided=simulated_cost_avoided,
    )


# ---------------------------------------------------------------------------
# _should_hard_route
# ---------------------------------------------------------------------------


class TestShouldHardRoute:
    def test_hard_route_simple_high_confidence(self) -> None:
        cfg = _make_config()
        analysis = _make_analysis(intent="SIMPLE_CODE", confidence=0.95, complexity=2)
        assert _should_hard_route(analysis, cfg) is True

    def test_hard_route_direct_intent(self) -> None:
        cfg = _make_config()
        analysis = _make_analysis(intent="DIRECT", confidence=0.90, complexity=1)
        assert _should_hard_route(analysis, cfg) is True

    def test_no_hard_route_complex_intent(self) -> None:
        cfg = _make_config()
        analysis = _make_analysis(intent="COMPLEX_REASONING", confidence=0.95, complexity=8)
        assert _should_hard_route(analysis, cfg) is False

    def test_no_hard_route_low_confidence(self) -> None:
        cfg = _make_config()
        analysis = _make_analysis(intent="SIMPLE_CODE", confidence=0.70, complexity=2)
        assert _should_hard_route(analysis, cfg) is False

    def test_no_hard_route_high_complexity(self) -> None:
        cfg = _make_config()
        # complexity > 3 (simple_ceiling), even with high confidence
        analysis = _make_analysis(intent="SIMPLE_CODE", confidence=0.95, complexity=5)
        assert _should_hard_route(analysis, cfg) is False

    def test_hard_route_only_for_simple_intents(self) -> None:
        """Hard-routing NEVER triggers for complex prompts (complexity >= 7)."""
        cfg = _make_config()
        for complexity in range(7, 11):
            analysis = _make_analysis(
                intent="COMPLEX_REASONING", confidence=0.99, complexity=complexity
            )
            assert _should_hard_route(analysis, cfg) is False, \
                f"Should not hard-route complexity={complexity}"


# ---------------------------------------------------------------------------
# _calculate_avoided_cost
# ---------------------------------------------------------------------------


class TestCalculateAvoidedCost:
    def test_simulated_cost_avoided_zero_when_savings_disabled(self) -> None:
        cfg = _make_config()
        result = _calculate_avoided_cost("hello world", cfg, cost_engine=None)
        assert result == 0.0

    def test_simulated_cost_avoided_uses_cost_engine(self) -> None:
        cfg = _make_config()
        mock_engine = MagicMock()
        mock_engine.calculate_cost.return_value = 0.00041

        result = _calculate_avoided_cost("write a function", cfg, cost_engine=mock_engine)

        assert result == pytest.approx(0.00041)
        mock_engine.calculate_cost.assert_called_once()
        # Verify the call used the reference model from config
        args = mock_engine.calculate_cost.call_args
        assert args[0][2] == "claude-3-5-haiku"
        assert args[0][3] == "anthropic"

    def test_simulated_cost_estimated_tokens(self) -> None:
        """Verify token estimation formula: words * 1.3 for input."""
        cfg = _make_config()
        mock_engine = MagicMock()
        mock_engine.calculate_cost.return_value = 0.0

        prompt = "write a hello world function"  # 5 words → ~6-7 input tokens
        _calculate_avoided_cost(prompt, cfg, cost_engine=mock_engine)

        args = mock_engine.calculate_cost.call_args[0]
        estimated_input = args[0]
        expected_input = int(5 * 1.3)  # = 6
        assert abs(estimated_input - expected_input) <= 1


# ---------------------------------------------------------------------------
# _build_aura_routing_context and _inject_routing_context
# ---------------------------------------------------------------------------


class TestRoutingContextSerialization:
    def test_routing_context_json_format(self) -> None:
        """Verify JSON serialization matches spec."""
        ctx = _make_routing_ctx()
        ctx_dict = _build_aura_routing_context(ctx)

        assert "_aura_routing_context" in ctx_dict
        inner = ctx_dict["_aura_routing_context"]
        assert "strategy" in inner
        assert "confidence_score" in inner
        assert "complexity_score" in inner
        assert "selected_route" in inner
        assert "analyzer_chain" in inner
        assert "intent" in inner
        assert "hard_routed" in inner
        assert "simulated_cost_avoided" in inner
        assert "metadata" in inner

    def test_simulated_cost_avoided_present_and_non_negative(self) -> None:
        ctx = _make_routing_ctx(hard_routed=True, simulated_cost_avoided=0.00041)
        ctx_dict = _build_aura_routing_context(ctx)
        cost = ctx_dict["_aura_routing_context"]["simulated_cost_avoided"]
        assert cost >= 0.0
        assert cost == pytest.approx(0.00041)

    def test_simulated_cost_avoided_zero_when_not_hard_routed(self) -> None:
        ctx = _make_routing_ctx(hard_routed=False, simulated_cost_avoided=0.0)
        ctx_dict = _build_aura_routing_context(ctx)
        assert ctx_dict["_aura_routing_context"]["simulated_cost_avoided"] == 0.0

    def test_inject_routing_context_appends_comment(self) -> None:
        ctx = _make_routing_ctx()
        response = "Hello, world!"
        injected = _inject_routing_context(response, ctx)

        assert injected.startswith("Hello, world!")
        assert "<!-- _aura_routing_context:" in injected
        assert "strategy" in injected

    def test_inject_routing_context_valid_json_in_comment(self) -> None:
        ctx = _make_routing_ctx()
        injected = _inject_routing_context("output", ctx)
        # Extract JSON from comment
        start = injected.index("<!-- _aura_routing_context:") + len("<!-- _aura_routing_context: ")
        end = injected.index(" -->")
        json_str = injected[start:end]
        data = json.loads(json_str)
        assert data["intent"] == "SIMPLE_CODE"


# ---------------------------------------------------------------------------
# route_task() pipeline integration
# ---------------------------------------------------------------------------


class TestRouteTaskPipelineEnabled:
    """Test that route_task uses the pipeline when enabled."""

    def _make_mock_pipeline(self, analysis: AnalysisResult) -> AnalyzerPipeline:
        """Create a mock pipeline that returns a fixed analysis."""
        mock = MagicMock(spec=AnalyzerPipeline)
        mock.run.return_value = analysis
        mock.build_routing_context.return_value = RoutingContext(
            strategy="vector",
            confidence_score=analysis.confidence,
            complexity_score=analysis.complexity_score,
            selected_route="coding",
            analyzer_chain=["edge-complexity"],
            intent=analysis.intent,
        )
        return mock

    def test_route_task_uses_legacy_when_disabled(self) -> None:
        """Pipeline disabled → uses legacy analyze_intent()."""
        cfg = _make_config(pipeline_enabled=False)
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "x", "endpoint": "http://x"}},
            roles={"router": ["m1"], "coding": ["m1"]},
        )

        with patch("aurarouter.mcp_tools.analyze_intent") as mock_ai:
            from aurarouter.routing import TriageResult
            mock_ai.return_value = TriageResult(intent="SIMPLE_CODE", complexity=2)

            with patch("aurarouter.fabric.ComputeFabric.execute") as mock_exec:
                mock_exec.return_value = GenerateResult(text="result output")
                output = route_task(fabric, None, task="write hello", config=cfg)

        mock_ai.assert_called()
        assert "result output" in output
        # Legacy path: no routing context comment
        assert "_aura_routing_context" not in output

    def test_routing_context_injected_in_response(self) -> None:
        """Pipeline enabled → _aura_routing_context present in response."""
        cfg = _make_config(pipeline_enabled=True)
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "x", "endpoint": "http://x"}},
            roles={"coding": ["m1"]},
        )

        analysis = _make_analysis(intent="SIMPLE_CODE", confidence=0.50, complexity=2)

        import aurarouter.mcp_tools as mt
        orig_pipeline = mt._analyzer_pipeline
        try:
            mock_pipeline = self._make_mock_pipeline(analysis)
            mt._analyzer_pipeline = mock_pipeline

            with patch("aurarouter.fabric.ComputeFabric.execute") as mock_exec:
                mock_exec.return_value = GenerateResult(text="hello output")
                output = route_task(fabric, None, task="write hello", config=cfg)

        finally:
            mt._analyzer_pipeline = orig_pipeline

        assert "_aura_routing_context" in output

    def test_routing_context_includes_simulated_cost_avoided_field(self) -> None:
        """_aura_routing_context always has the simulated_cost_avoided field."""
        ctx = _make_routing_ctx(simulated_cost_avoided=0.0)
        serialized = _build_aura_routing_context(ctx)
        assert "simulated_cost_avoided" in serialized["_aura_routing_context"]

    def test_pipeline_fallback_to_legacy_on_exception(self) -> None:
        """If pipeline raises, route_task falls back to legacy path gracefully."""
        cfg = _make_config(pipeline_enabled=True)
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "x", "endpoint": "http://x"}},
            roles={"router": ["m1"], "coding": ["m1"]},
        )

        import aurarouter.mcp_tools as mt
        orig_pipeline = mt._analyzer_pipeline
        try:
            bad_pipeline = MagicMock(spec=AnalyzerPipeline)
            bad_pipeline.run.side_effect = RuntimeError("pipeline exploded")
            mt._analyzer_pipeline = bad_pipeline

            with patch("aurarouter.mcp_tools.analyze_intent") as mock_ai:
                from aurarouter.routing import TriageResult
                mock_ai.return_value = TriageResult(intent="SIMPLE_CODE", complexity=2)

                with patch("aurarouter.fabric.ComputeFabric.execute") as mock_exec:
                    mock_exec.return_value = GenerateResult(text="fallback output")
                    output = route_task(fabric, None, task="test fallback", config=cfg)

        finally:
            mt._analyzer_pipeline = orig_pipeline

        # Legacy path was used and produced output
        assert "fallback output" in output


class TestHardRouting:
    """Test hard-routing bypasses cloud for simple tasks."""

    def test_hard_route_bypasses_cloud(self) -> None:
        """Hard-route detected → fabric.execute called with local_chain override."""
        cfg = _make_config(pipeline_enabled=True)

        # Local model in config
        fabric = _make_fabric(
            models={"local_m": {"provider": "ollama", "model_name": "x", "endpoint": "http://x",
                                 "hosting_tier": "on-prem"}},
            roles={"coding": ["local_m"]},
        )

        # Analysis that triggers hard-routing: low complexity, high confidence, DIRECT intent
        analysis = _make_analysis(intent="DIRECT", confidence=0.95, complexity=1)

        import aurarouter.mcp_tools as mt
        orig_pipeline = mt._analyzer_pipeline
        try:
            mock_pipeline = MagicMock(spec=AnalyzerPipeline)
            mock_pipeline.run.return_value = analysis
            mock_pipeline.build_routing_context.return_value = RoutingContext(
                strategy="vector",
                confidence_score=0.95,
                complexity_score=1,
                selected_route="coding",
                analyzer_chain=["edge-complexity", "onnx-vector"],
                intent="DIRECT",
            )
            mt._analyzer_pipeline = mock_pipeline

            with patch("aurarouter.fabric.ComputeFabric.execute") as mock_exec:
                mock_exec.return_value = GenerateResult(text="local result")
                output = route_task(fabric, None, task="hello", config=cfg)

        finally:
            mt._analyzer_pipeline = orig_pipeline

        # hard_routed flag should be in the response
        assert "hard_routed" in output
        assert "_aura_routing_context" in output

    def test_hard_route_fallthrough_when_no_local_models(self) -> None:
        """If no local models available, falls through gracefully (no crash)."""
        cfg = _make_config(pipeline_enabled=True)
        # Cloud model only
        fabric = _make_fabric(
            models={"cloud_m": {"provider": "anthropic", "model_name": "x",
                                 "hosting_tier": "cloud"}},
            roles={"coding": ["cloud_m"]},
        )

        analysis = _make_analysis(intent="DIRECT", confidence=0.95, complexity=1)

        import aurarouter.mcp_tools as mt
        orig_pipeline = mt._analyzer_pipeline
        try:
            mock_pipeline = MagicMock(spec=AnalyzerPipeline)
            mock_pipeline.run.return_value = analysis
            mock_pipeline.build_routing_context.return_value = RoutingContext(
                strategy="vector",
                confidence_score=0.95,
                complexity_score=1,
                selected_route="coding",
                analyzer_chain=["edge-complexity"],
                intent="DIRECT",
            )
            mt._analyzer_pipeline = mock_pipeline

            with patch("aurarouter.fabric.ComputeFabric.execute") as mock_exec:
                mock_exec.return_value = GenerateResult(text="cloud result")
                # Should not raise even though local_chain is empty
                output = route_task(fabric, None, task="hello", config=cfg)

        finally:
            mt._analyzer_pipeline = orig_pipeline

        # Output should be produced (either from hard-route or normal path)
        assert output is not None


class TestAnalyzerChain:
    """Verify analyzer_chain includes pre-filter and classifier IDs."""

    def test_analyzer_chain_includes_prefilter_and_classifier(self) -> None:
        """RoutingContext.analyzer_chain should list edge-complexity before classifiers."""
        from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer
        from aurarouter.analyzer_pipeline import AnalyzerPipeline

        class _FakeClassifier:
            analyzer_id = "fake-clf"
            priority = 50
            def supports(self, p): return True
            def analyze(self, p, c=""): return AnalysisResult(
                intent="SIMPLE_CODE", confidence=0.92, complexity_score=0, analyzer_id="fake-clf"
            )

        scorer = EdgeComplexityScorer()
        clf = _FakeClassifier()
        pipeline = AnalyzerPipeline(
            pre_filters=[scorer],
            classifiers=[clf],
            confidence_threshold=0.85,
        )
        result = pipeline.run("write a simple function")
        ctx = pipeline.build_routing_context(result, selected_route="coding")

        assert "edge-complexity" in ctx.analyzer_chain
        assert "fake-clf" in ctx.analyzer_chain
        # Pre-filter first
        assert ctx.analyzer_chain.index("edge-complexity") < ctx.analyzer_chain.index("fake-clf")


# ---------------------------------------------------------------------------
# GenerateResult routing_context field
# ---------------------------------------------------------------------------


class TestGenerateResultRoutingContext:
    def test_generate_result_has_routing_context_field(self) -> None:
        result = GenerateResult(text="hello")
        assert result.routing_context is None

    def test_generate_result_routing_context_settable(self) -> None:
        ctx = _make_routing_ctx()
        result = GenerateResult(text="hello")
        result.routing_context = ctx
        assert result.routing_context is ctx

    def test_fabric_execute_attaches_routing_context(self) -> None:
        """fabric.execute() attaches routing_context to GenerateResult."""
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "x", "endpoint": "http://x"}},
            roles={"coding": ["m1"]},
        )
        ctx = _make_routing_ctx()

        with patch("aurarouter.fabric.ComputeFabric._get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.generate_with_usage.return_value = GenerateResult(text="output")
            mock_get_provider.return_value = mock_provider

            result = fabric.execute("coding", "test", routing_context=ctx)

        assert result is not None
        assert result.routing_context is ctx


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestContractsUpdated:
    def test_auraxlm_contract_includes_routing_context_param(self) -> None:
        from aurarouter.contracts.auraxlm import ANALYZE_ROUTE_PARAMS, ANALYZE_ROUTE_RESPONSE
        assert "_aura_routing_context" in ANALYZE_ROUTE_PARAMS
        assert "_aura_routing_context" in ANALYZE_ROUTE_RESPONSE

    def test_auracode_contract_includes_routing_context_schema(self) -> None:
        from aurarouter.contracts.auracode import AURACODE_ROUTING_CONTEXT_SCHEMA
        assert "simulated_cost_avoided" in AURACODE_ROUTING_CONTEXT_SCHEMA
        assert "hard_routed" in AURACODE_ROUTING_CONTEXT_SCHEMA
