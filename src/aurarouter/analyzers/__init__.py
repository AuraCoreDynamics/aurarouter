"""Analyzer plugin registry for AuraRouter.

Discovers and manages PromptAnalyzer implementations from the catalog.
Also provides backwards-compatible re-export of create_default_analyzer().

TG1 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aurarouter.analyzer_protocol import AnalysisResult, PromptAnalyzer
from aurarouter.catalog_model import ArtifactKind, CatalogArtifact

if TYPE_CHECKING:
    from aurarouter.analyzer_pipeline import AnalyzerPipeline
    from aurarouter.config import ConfigLoader
    from aurarouter.fabric import ComputeFabric
    from aurarouter.intent_registry import IntentRegistry

logger = logging.getLogger("AuraRouter.AnalyzerRegistry")


# ---------------------------------------------------------------------------
# Backwards-compatible factory (was the only export from the old analyzers.py)
# ---------------------------------------------------------------------------

def create_default_analyzer() -> CatalogArtifact:
    """Built-in analyzer wrapping existing intent -> triage -> execute logic."""
    return CatalogArtifact(
        artifact_id="aurarouter-default",
        kind=ArtifactKind.ANALYZER,
        display_name="AuraRouter Default",
        description="Intent classification with complexity-based triage routing",
        provider="aurarouter",
        version="1.0",
        capabilities=["code", "reasoning", "review", "planning"],
        spec={
            "analyzer_kind": "intent_triage",
            "role_bindings": {
                "simple_code": "coding",
                "complex_reasoning": "reasoning",
                "review": "reviewer",
            },
        },
    )


# ---------------------------------------------------------------------------
# AnalyzerRegistry
# ---------------------------------------------------------------------------

#: Maps analyzer_kind catalog key → implementation class name (resolved lazily).
_KIND_TO_CLASS: dict[str, str] = {
    "slm_intent": "SLMIntentAnalyzer",
    "onnx_vector": "ONNXVectorAnalyzer",
    "edge_complexity": "EdgeComplexityScorer",
    # "moe_ranking" is handled remotely by broker — not instantiated locally
}


class AnalyzerRegistry:
    """Discovers and manages PromptAnalyzer plugins.

    Reads the artifact catalog for kind=analyzer entries and instantiates
    known analyzer_kind → class mappings.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self._config = config
        self._analyzers: dict[str, PromptAnalyzer] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        *,
        fabric: ComputeFabric | None = None,
        intent_registry: IntentRegistry | None = None,
    ) -> list[PromptAnalyzer]:
        """Scan catalog for kind=analyzer artifacts and instantiate known kinds.

        Unknown analyzer_kind values are logged as warnings and skipped.

        Returns the list of newly discovered analyzers.
        """
        discovered: list[PromptAnalyzer] = []
        artifact_ids = self._config.catalog_list(kind=ArtifactKind.ANALYZER.value)

        for artifact_id in artifact_ids:
            data = self._config.catalog_get(artifact_id)
            if not data:
                continue
            analyzer_kind = data.get("analyzer_kind", "")
            if not analyzer_kind:
                continue
            if analyzer_kind not in _KIND_TO_CLASS:
                if analyzer_kind != "moe_ranking":  # Suppress expected remote kind
                    logger.warning(
                        "Unknown analyzer_kind '%s' for artifact '%s' — skipping",
                        analyzer_kind, artifact_id,
                    )
                continue

            instance = self._instantiate(
                artifact_id=artifact_id,
                analyzer_kind=analyzer_kind,
                spec=data,
                fabric=fabric,
                intent_registry=intent_registry,
            )
            if instance is not None:
                self._analyzers[artifact_id] = instance
                discovered.append(instance)
                logger.info("Discovered analyzer: %s (%s)", artifact_id, analyzer_kind)

        return discovered

    # ------------------------------------------------------------------
    # Manual registration
    # ------------------------------------------------------------------

    def register(self, analyzer: PromptAnalyzer) -> None:
        """Manually register an analyzer instance."""
        self._analyzers[analyzer.analyzer_id] = analyzer
        logger.debug("Registered analyzer: %s", analyzer.analyzer_id)

    def get(self, analyzer_id: str) -> PromptAnalyzer | None:
        """Look up a registered analyzer by ID."""
        return self._analyzers.get(analyzer_id)

    def all(self) -> list[PromptAnalyzer]:
        """Return all registered analyzers."""
        return list(self._analyzers.values())

    # ------------------------------------------------------------------
    # Pipeline factory
    # ------------------------------------------------------------------

    def build_pipeline(
        self,
        fabric: ComputeFabric | None = None,
        intent_registry: IntentRegistry | None = None,
        confidence_threshold: float = 0.85,
    ) -> AnalyzerPipeline:
        """Construct a fully-wired two-stage AnalyzerPipeline.

        Stage 1 (pre-filters): EdgeComplexityScorer and any other mandatory
          analyzers that must always run.
        Stage 2 (classifiers): ONNXVectorAnalyzer (if available), then
          SLMIntentAnalyzer as fallback.  Short-circuit at confidence_threshold.

        Always includes SLMIntentAnalyzer as final classifier if fabric is
        available — this is the accuracy guarantee.
        """
        from aurarouter.analyzer_pipeline import AnalyzerPipeline

        # Run discovery to populate self._analyzers
        self.discover(fabric=fabric, intent_registry=intent_registry)

        pipeline_cfg = self._config.get_pipeline_config()
        threshold = pipeline_cfg.get("confidence_threshold", confidence_threshold)

        pipeline = AnalyzerPipeline(confidence_threshold=threshold)

        # ── Stage 1: EdgeComplexityScorer as pre-filter ──────────────
        edge_scorer = self._analyzers.get("edge-complexity")
        if edge_scorer is not None:
            pipeline.add_pre_filter(edge_scorer)
        else:
            # Create inline if not in catalog
            edge_scorer = self._make_edge_complexity_scorer()
            if edge_scorer is not None:
                pipeline.add_pre_filter(edge_scorer)

        # ── Stage 2: ONNX vector classifier (if available) ───────────
        onnx_clf = self._analyzers.get("onnx-vector")
        if onnx_clf is not None:
            pipeline.add_classifier(onnx_clf)

        # ── Stage 2: SLM fallback (always, if fabric available) ──────
        if fabric is not None and intent_registry is not None:
            slm = self._analyzers.get("slm-intent")
            if slm is None:
                from aurarouter.analyzers.slm_analyzer import SLMIntentAnalyzer
                slm = SLMIntentAnalyzer(fabric=fabric, intent_registry=intent_registry)
            pipeline.add_classifier(slm)

        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _instantiate(
        self,
        *,
        artifact_id: str,
        analyzer_kind: str,
        spec: dict,
        fabric: ComputeFabric | None,
        intent_registry: IntentRegistry | None,
    ) -> PromptAnalyzer | None:
        """Instantiate an analyzer from catalog spec."""
        try:
            if analyzer_kind == "slm_intent":
                if fabric is None or intent_registry is None:
                    logger.debug(
                        "Skipping '%s': requires fabric and intent_registry", artifact_id
                    )
                    return None
                from aurarouter.analyzers.slm_analyzer import SLMIntentAnalyzer
                return SLMIntentAnalyzer(fabric=fabric, intent_registry=intent_registry)

            elif analyzer_kind == "onnx_vector":
                from aurarouter.analyzers.onnx_vector import ONNXVectorAnalyzer
                if intent_registry is None:
                    logger.debug(
                        "Skipping '%s': requires intent_registry", artifact_id
                    )
                    return None
                kwargs: dict = {}
                if "onnx_similarity_threshold" in spec:
                    kwargs["similarity_threshold"] = float(spec["onnx_similarity_threshold"])
                if "onnx_margin_threshold" in spec:
                    kwargs["margin_threshold"] = float(spec["onnx_margin_threshold"])
                if "onnx_max_sequence_length" in spec:
                    kwargs["max_sequence_length"] = int(spec["onnx_max_sequence_length"])
                if "onnx_model_path_override" in spec:
                    kwargs["model_path"] = spec["onnx_model_path_override"]
                if "onnx_tokenizer_path_override" in spec:
                    kwargs["tokenizer_path"] = spec["onnx_tokenizer_path_override"]
                return ONNXVectorAnalyzer(intent_registry=intent_registry, **kwargs)

            elif analyzer_kind == "edge_complexity":
                from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer
                cfg = self._config.get_complexity_scorer_config()
                return EdgeComplexityScorer(
                    simple_ceiling=cfg.get("simple_ceiling", 3),
                    complex_floor=cfg.get("complex_floor", 7),
                )

        except Exception as exc:
            logger.warning(
                "Failed to instantiate analyzer '%s' (%s): %s",
                artifact_id, analyzer_kind, exc, exc_info=True,
            )
        return None

    def _make_edge_complexity_scorer(self) -> PromptAnalyzer | None:
        """Create an EdgeComplexityScorer with config defaults."""
        try:
            from aurarouter.analyzers.edge_complexity import EdgeComplexityScorer
            cfg = self._config.get_complexity_scorer_config()
            return EdgeComplexityScorer(
                simple_ceiling=cfg.get("simple_ceiling", 3),
                complex_floor=cfg.get("complex_floor", 7),
            )
        except Exception as exc:
            logger.debug("Could not create EdgeComplexityScorer: %s", exc)
            return None
