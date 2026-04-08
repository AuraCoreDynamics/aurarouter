"""SLM-based intent analyzer — wraps the existing analyze_intent() function.

This is the high-accuracy, high-latency fallback classifier.  It runs as a
Stage 2 classifier (lowest priority) only when faster analyzers abstain or
have low confidence.

TG1 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aurarouter.analyzer_protocol import AnalysisResult

if TYPE_CHECKING:
    from aurarouter.fabric import ComputeFabric
    from aurarouter.intent_registry import IntentRegistry

logger = logging.getLogger("AuraRouter.SLMIntentAnalyzer")

_ANALYZER_ID = "slm-intent"
_PRIORITY = 0  # Lowest priority — runs last as fallback
_SLM_CONFIDENCE = 0.95  # SLM classification is high-confidence by convention


class SLMIntentAnalyzer:
    """Wraps the existing SLM-based intent classification as a PromptAnalyzer.

    This is the smart fallback — high accuracy, high latency (~200–800ms).
    Always registered as a Stage 2 *classifier*, never a Stage 1 pre-filter.
    """

    def __init__(
        self,
        fabric: ComputeFabric,
        intent_registry: IntentRegistry,
    ) -> None:
        self._fabric = fabric
        self._intent_registry = intent_registry

    # ── PromptAnalyzer protocol ──────────────────────────────────────

    @property
    def analyzer_id(self) -> str:
        return _ANALYZER_ID

    @property
    def priority(self) -> int:
        return _PRIORITY

    def supports(self, prompt: str) -> bool:
        """SLM can handle any prompt."""
        return True

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult | None:
        """Delegate to existing analyze_intent() and map TriageResult → AnalysisResult.

        complexity_score is set to 0 as a sentinel — Stage 2 classifiers never
        own the complexity measurement.  The AnalyzerPipeline replaces it with
        the Stage 1 pre-filter value before returning the merged result.
        """
        from aurarouter.routing import analyze_intent

        try:
            triage = analyze_intent(
                self._fabric,
                prompt,
                intent_registry=self._intent_registry,
            )
        except Exception as exc:
            logger.warning("SLMIntentAnalyzer.analyze failed: %s", exc, exc_info=True)
            return None

        return AnalysisResult(
            intent=triage.intent,
            confidence=_SLM_CONFIDENCE,
            complexity_score=0,  # Sentinel: replaced by Stage 1 pre-filter value
            analyzer_id=_ANALYZER_ID,
            reasoning=f"SLM classification: {triage.intent} (complexity from triage: {triage.complexity})",
            metadata={"triage_complexity": triage.complexity},
        )
