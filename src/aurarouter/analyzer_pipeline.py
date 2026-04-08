"""Two-stage Analyzer Pipeline for AuraRouter.

Orchestrates prompt analysis through two distinct stages:
  - Stage 1 (Pre-filter): Always runs all registered pre-filters unconditionally.
    Contributes partial state, notably complexity_score.
  - Stage 2 (Classifier): Runs intent classifiers in priority order and short-circuits
    when confidence exceeds the threshold.

The separation guarantees that complexity is always measured — even when the ONNX
classifier fires immediately at high confidence.

TG1 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from aurarouter.analyzer_protocol import AnalysisResult, PromptAnalyzer, RoutingContext

logger = logging.getLogger("AuraRouter.AnalyzerPipeline")

_FALLBACK_INTENT = "DIRECT"
_FALLBACK_COMPLEXITY = 1


class AnalyzerPipeline:
    """Two-stage pipeline: mandatory pre-filters then confidence-gated classifiers.

    Stage 1 — Pre-filter stage (always runs all):
      Analyzers registered as pre-filters run unconditionally regardless of any
      confidence result.  They contribute partial state (notably: complexity_score)
      that is stamped onto every downstream result.  Pre-filters never trigger
      the short-circuit — they are interceptors, not classifiers.

    Stage 2 — Classifier stage (short-circuits on confidence):
      Analyzers that classify intent (ONNX vector, SLM fallback) run in priority
      order and exit as soon as one returns confidence >= threshold.  The
      complexity_score stamped by Stage 1 is merged into the winning result.

    Key guarantee: complexity_score in the final AnalysisResult always comes from
    the pre-filter stage and is never overwritten by Stage 2.
    """

    def __init__(
        self,
        pre_filters: list[PromptAnalyzer] | None = None,
        classifiers: list[PromptAnalyzer] | None = None,
        confidence_threshold: float = 0.85,
        fallback_intent: str = _FALLBACK_INTENT,
        fallback_complexity: int = _FALLBACK_COMPLEXITY,
    ) -> None:
        self._pre_filters: list[PromptAnalyzer] = list(pre_filters or [])
        self._classifiers: list[PromptAnalyzer] = list(classifiers or [])
        self._confidence_threshold = confidence_threshold
        self._fallback_intent = fallback_intent
        self._fallback_complexity = fallback_complexity
        # Track the analyzer IDs that ran in the last pipeline execution
        self._last_chain: list[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_pre_filter(self, analyzer: PromptAnalyzer) -> None:
        """Register a mandatory pre-filter (always runs, never short-circuits)."""
        self._pre_filters.append(analyzer)
        logger.debug("Registered pre-filter: %s (priority=%d)", analyzer.analyzer_id, analyzer.priority)

    def add_classifier(self, analyzer: PromptAnalyzer) -> None:
        """Register a classifier (runs in priority order, short-circuits on confidence)."""
        self._classifiers.append(analyzer)
        logger.debug("Registered classifier: %s (priority=%d)", analyzer.analyzer_id, analyzer.priority)

    def remove_analyzer(self, analyzer_id: str) -> bool:
        """Remove from either stage by ID.  Returns True if found."""
        for collection in (self._pre_filters, self._classifiers):
            for i, a in enumerate(collection):
                if a.analyzer_id == analyzer_id:
                    collection.pop(i)
                    logger.debug("Removed analyzer: %s", analyzer_id)
                    return True
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, prompt: str, context: str = "") -> AnalysisResult:
        """Execute both stages and return a merged AnalysisResult.

        Stage 1 — Pre-filters (all run unconditionally):
          1. For each pre_filter (sorted by priority descending):
             a. Call supports(prompt) — if False, skip
             b. Call analyze(prompt, context)
             c. Accumulate non-None result into pre_filter_state
             d. Take the highest-priority non-None result's complexity_score
                as the authoritative complexity for this pipeline run.
          After Stage 1: complexity_score is fixed regardless of Stage 2 outcome.

        Stage 2 — Classifiers (short-circuit on confidence):
          1. For each classifier (sorted by priority descending):
             a. Call supports(prompt) — skip if False
             b. Call analyze(prompt, context)
             c. If result is not None and confidence >= threshold:
                → Merge complexity_score from Stage 1 into result
                → Return merged AnalysisResult immediately
             d. Track best partial result (highest confidence)
          After all classifiers: return best partial result merged with Stage 1
          complexity, or fallback AnalysisResult if all abstained.
        """
        chain: list[str] = []

        # ── Stage 1: Pre-filters ─────────────────────────────────────
        stage1_complexity: int = self._fallback_complexity
        sorted_prefilters = sorted(self._pre_filters, key=lambda a: a.priority, reverse=True)

        for pf in sorted_prefilters:
            try:
                result = pf.analyze(prompt, context)
                chain.append(pf.analyzer_id)
                if result is not None:
                    # First (highest-priority) non-None pre-filter result wins complexity.
                    # Since sorted descending, the first non-None is the highest priority.
                    if stage1_complexity == self._fallback_complexity:
                        stage1_complexity = result.complexity_score
            except Exception as exc:
                logger.warning("Pre-filter %s raised: %s", pf.analyzer_id, exc, exc_info=True)

        # ── Stage 2: Classifiers ─────────────────────────────────────
        sorted_classifiers = sorted(self._classifiers, key=lambda a: a.priority, reverse=True)
        best_partial: AnalysisResult | None = None

        for clf in sorted_classifiers:
            if not clf.supports(prompt):
                continue
            try:
                result = clf.analyze(prompt, context)
                chain.append(clf.analyzer_id)
                if result is None:
                    continue
                # Track best partial (highest confidence among non-None results)
                if best_partial is None or result.confidence > best_partial.confidence:
                    best_partial = result
                # Short-circuit if confidence threshold is met
                if result.confidence >= self._confidence_threshold:
                    merged = self._merge_complexity(result, stage1_complexity)
                    self._last_chain = list(chain)
                    logger.info(
                        "Pipeline short-circuit: analyzer=%s intent=%s confidence=%.3f complexity=%d",
                        result.analyzer_id, result.intent, result.confidence, merged.complexity_score,
                    )
                    return merged
            except Exception as exc:
                logger.warning("Classifier %s raised: %s", clf.analyzer_id, exc, exc_info=True)

        self._last_chain = list(chain)

        # ── Return best partial (merged with Stage 1 complexity) ─────
        if best_partial is not None:
            merged = self._merge_complexity(best_partial, stage1_complexity)
            logger.info(
                "Pipeline completed (no short-circuit): best=%s confidence=%.3f complexity=%d",
                merged.analyzer_id, merged.confidence, merged.complexity_score,
            )
            return merged

        # ── Fallback: all classifiers abstained ─────────────────────
        logger.info(
            "Pipeline fallback: all classifiers abstained. complexity=%d", stage1_complexity
        )
        return AnalysisResult(
            intent=self._fallback_intent,
            confidence=0.0,
            complexity_score=stage1_complexity,
            analyzer_id="pipeline-fallback",
            reasoning="All classifiers abstained",
        )

    def build_routing_context(
        self, result: AnalysisResult, selected_route: str
    ) -> RoutingContext:
        """Construct the standardized RoutingContext from a pipeline result.

        Populates analyzer_chain from the last pipeline run.
        simulated_cost_avoided defaults to 0.0 — set by the hard-routing logic in TG4.
        """
        # Determine strategy name from the winning analyzer's ID
        strategy_map = {
            "onnx-vector": "vector",
            "edge-complexity": "complexity",
            "slm-intent": "slm",
            "pipeline-fallback": "fallback",
        }
        strategy = strategy_map.get(result.analyzer_id, "slm")

        return RoutingContext(
            strategy=strategy,
            confidence_score=result.confidence,
            complexity_score=result.complexity_score,
            selected_route=selected_route,
            analyzer_chain=list(self._last_chain),
            intent=result.intent,
            hard_routed=False,
            simulated_cost_avoided=0.0,
            metadata=dict(result.metadata),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_complexity(result: AnalysisResult, stage1_complexity: int) -> AnalysisResult:
        """Return a new AnalysisResult with complexity_score replaced by Stage 1 value.

        If the result already has a non-zero complexity (from a pre-filter acting
        as a classifier), keep the Stage 1 value — it is always authoritative.
        """
        if result.complexity_score == stage1_complexity:
            return result
        return AnalysisResult(
            intent=result.intent,
            confidence=result.confidence,
            complexity_score=stage1_complexity,
            analyzer_id=result.analyzer_id,
            reasoning=result.reasoning,
            metadata=result.metadata,
        )

    @property
    def pre_filters(self) -> list[PromptAnalyzer]:
        """Read-only view of registered pre-filters."""
        return list(self._pre_filters)

    @property
    def classifiers(self) -> list[PromptAnalyzer]:
        """Read-only view of registered classifiers."""
        return list(self._classifiers)
