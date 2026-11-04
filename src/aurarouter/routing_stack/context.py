"""Data structures and trace models for the Two-Stage RoutingPolicyStack.

AuraRouter v2.1 §4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def complexity_to_band(complexity: float | int) -> int:
    """Discretize complexity score [1..10] into 3 bandit bands: 0 (<=3), 1 (4..7), 2 (>=8)."""
    if complexity <= 3:
        return 0
    if complexity <= 7:
        return 1
    return 2


@dataclass(frozen=True, slots=True)
class CandidateEvalContext:
    """Immutable per-request evaluation context passed through Stage 1 & Stage 2."""

    role: str
    intent: str = "DIRECT"
    complexity_score: int = 5
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 512
    required_capabilities: frozenset[str] = frozenset()
    effective_data_categories: frozenset[str] = frozenset()
    sovereignty_verdict_open: bool = True
    session_id: str | None = None
    last_model_id: str | None = None
    last_routed_at: float | None = None  # Unix epoch seconds (time.time())
    bypass_sovereignty: bool = False
    failover_projection_enabled: bool = True
    deterministic: bool = False
    deterministic_seed: int | None = None
    max_vram_gb: float | None = None
    quant: str = "q4_k_m"


@dataclass(frozen=True, slots=True)
class FilterVerdict:
    """Verdict emitted by a Stage 1 non-compensatory EligibilityFilter."""

    filter_name: str
    model_id: str
    passed: bool
    reason: str = ""
    is_availability_gate: bool = False
    recovery_seconds: float = 0.0
    is_daily_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class FactorScore:
    """Individual factor breakdown for a candidate model in Stage 2."""

    factor_name: str
    value: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Composite Stage 2 utility score for an eligible candidate model."""

    model_id: str
    utility: float
    quality_sample: float
    headroom_mult: float
    health_mult: float
    cache_mult: float
    cost_penalty: float
    raw_cost_usd: float
    original_index: int
    audit_mult: float = 1.0
    factors: tuple[FactorScore, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "utility": round(self.utility, 5),
            "quality_sample": round(self.quality_sample, 5),
            "headroom_mult": round(self.headroom_mult, 5),
            "health_mult": round(self.health_mult, 5),
            "cache_mult": round(self.cache_mult, 5),
            "audit_mult": round(self.audit_mult, 5),
            "cost_penalty": round(self.cost_penalty, 5),
            "raw_cost_usd": round(self.raw_cost_usd, 6),
            "original_index": self.original_index,
            "factors": [
                {"name": f.factor_name, "value": round(f.value, 5), "detail": f.detail}
                for f in self.factors
            ],
        }


@dataclass
class RoutingDecisionTrace:
    """Structured explanation of a Two-Stage RoutingPolicyStack decision."""

    role: str
    mode: str
    input_chain: list[str]
    ranked_chain: list[str]
    pruned: dict[str, str] = field(default_factory=dict)
    scores: list[CandidateScore] = field(default_factory=list)
    last_resort_probe: str | None = None
    elapsed_us: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "mode": self.mode,
            "input_chain": list(self.input_chain),
            "ranked_chain": list(self.ranked_chain),
            "pruned": dict(self.pruned),
            "scores": [s.to_dict() for s in self.scores],
            "last_resort_probe": self.last_resort_probe,
            "elapsed_us": round(self.elapsed_us, 2),
        }

