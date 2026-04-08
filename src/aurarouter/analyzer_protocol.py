"""Pluggable Analyzer Protocol for AuraRouter.

Defines the core plugin interface (PromptAnalyzer), result types (AnalysisResult,
RoutingContext), and the structural Protocol that all analyzer implementations must
conform to.  Uses typing.Protocol for structural subtyping — consistent with
ProviderProtocol in providers/protocol.py.

TG1 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AnalysisResult:
    """Result of a single analyzer's evaluation of a prompt."""

    intent: str
    """Resolved intent name (e.g., "SIMPLE_CODE")."""

    confidence: float
    """0.0–1.0 confidence in the classification."""

    complexity_score: int
    """1–10 complexity rating.  Stage 2 classifiers set this to 0 as a sentinel;
    the pipeline replaces it with the Stage 1 pre-filter value before returning.
    Stage 1 pre-filters always return the real measured score."""

    analyzer_id: str
    """Which analyzer produced this result."""

    reasoning: str = ""
    """Optional human-readable explanation."""

    metadata: dict = field(default_factory=dict)
    """Analyzer-specific data.  May carry vector distances, token patterns, etc."""


@dataclass(frozen=True)
class RoutingContext:
    """Standardized routing metadata block injected into payload state.

    Compatible with OpenAI extra_body and OpenTelemetry span attributes.
    Uses snake_case keys throughout (enforced in C# via [JsonPropertyName]).
    """

    strategy: str
    """Classification strategy used: "vector", "complexity", "slm", "federated"."""

    confidence_score: float
    """Final pipeline confidence (0.0–1.0)."""

    complexity_score: int
    """Final complexity rating (1–10)."""

    selected_route: str
    """Role name (e.g., "coding", "reasoning")."""

    analyzer_chain: list
    """Ordered list of analyzer IDs that ran (pre-filter first, then classifiers)."""

    intent: str
    """Final resolved intent name."""

    hard_routed: bool = False
    """Whether cloud was bypassed (local-only execution)."""

    simulated_cost_avoided: float = 0.0
    """Estimated USD cost avoided by hard-routing to local instead of cloud.
    0.0 if not hard-routed or savings subsystem disabled."""

    metadata: dict = field(default_factory=dict)
    """Additional pipeline-level metadata."""


@runtime_checkable
class PromptAnalyzer(Protocol):
    """Plugin interface for prompt analysis.  Structural subtyping — no ABC.

    Implementations do NOT need to inherit from this class.  They only need
    to provide the matching attributes and methods.

    Stage assignment:
      - Stage 1 (pre-filter): priority >= 50, always runs, never short-circuits.
        Contributes complexity_score to shared pipeline state.
      - Stage 2 (classifier): priority < 50 (or registered as classifier),
        runs in priority order, short-circuits when confidence >= threshold.
    """

    @property
    def analyzer_id(self) -> str:
        """Unique identifier for this analyzer."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority.  Higher runs first.  Default: 0."""
        ...

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult | None:
        """Classify a prompt.

        Return None to abstain (pass to next analyzer in Stage 2).
        Stage 1 pre-filters must never return None — complexity is always measurable.
        """
        ...

    def supports(self, prompt: str) -> bool:
        """Fast pre-check: can this analyzer handle this prompt type?

        Return False to skip without calling analyze().
        Stage 1 pre-filters should always return True.
        """
        ...
