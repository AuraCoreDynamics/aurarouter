"""Smart triage routing — complexity-based role selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.savings.feedback_store import FeedbackStore

logger = get_logger("AuraRouter.Triage")


@dataclass
class TriageRule:
    """A single complexity-threshold rule that maps to a preferred role."""

    max_complexity: int
    preferred_role: str
    description: str = ""


class TriageRouter:
    """Select a role chain based on complexity score and ordered rules.

    Rules are evaluated in order; the first rule whose ``max_complexity``
    is >= the score wins.  If no rule matches, *default_role* is returned.
    """

    def __init__(
        self,
        rules: list[TriageRule] | None = None,
        default_role: str = "coding",
    ):
        self.rules: list[TriageRule] = rules or []
        self.default_role = default_role

    def select_role(self, complexity_score: int) -> str:
        """Return the preferred role for *complexity_score*."""
        for rule in self.rules:
            if complexity_score <= rule.max_complexity:
                logger.info(
                    "Triage: complexity %d <= %d → role '%s' (%s)",
                    complexity_score,
                    rule.max_complexity,
                    rule.preferred_role,
                    rule.description,
                )
                return rule.preferred_role

        logger.info(
            "Triage: complexity %d matched no rule → default '%s'",
            complexity_score,
            self.default_role,
        )
        return self.default_role

    def update_from_feedback(
        self,
        feedback_store: FeedbackStore,
        blend_factor: float = 0.3,
    ) -> None:
        """Update threshold weights using exponential moving average of observed success rates.

        *blend_factor* controls how much observed data influences the thresholds.
        ``0.0`` = ignore feedback (no change), ``1.0`` = fully data-driven.

        For each rule, the method queries model success rates within the
        complexity band that the rule covers.  If the observed success rate
        for the band's preferred role models is low, the rule's
        ``max_complexity`` is shifted down (making the band narrower) so
        that harder tasks fall through to more capable roles.  Conversely,
        high success rates widen the band.

        The update uses EMA: ``new = (1 - blend) * old + blend * observed``.
        """
        if blend_factor <= 0.0:
            return

        blend_factor = min(blend_factor, 1.0)

        stats = feedback_store.model_stats(window_days=7)
        if not stats:
            return

        # Build a lookup: model_id -> success_rate
        model_rates: dict[str, float] = {
            s["model_id"]: s["success_rate"] for s in stats
        }

        # For each rule, compute observed success rate in its complexity band
        prev_max = 0
        for rule in self.rules:
            band_min = prev_max
            band_max = rule.max_complexity

            # Query success rate for models used in this band
            rate = feedback_store.success_rate(
                model_id="",  # We'll compute from model_stats instead
                complexity_min=band_min,
                complexity_max=band_max,
                window_days=7,
            )

            # If no data from band-specific query, try per-model rates
            if rate == 0.0:
                # Check if any model we know about has stats
                for mid, mr in model_rates.items():
                    band_rate = feedback_store.success_rate(
                        model_id=mid,
                        complexity_min=band_min,
                        complexity_max=band_max,
                        window_days=7,
                    )
                    if band_rate > 0:
                        rate = band_rate
                        break

            if rate > 0.0:
                # EMA: shift threshold based on observed success rate
                # High success rate (> 0.8) → widen band (increase max_complexity)
                # Low success rate (< 0.5) → narrow band (decrease max_complexity)
                # Neutral at 0.65
                shift = (rate - 0.65) * 5.0  # Scale: ±0.35 → ±1.75
                observed_threshold = rule.max_complexity + shift
                # Clamp to reasonable range [1, 10]
                observed_threshold = max(1, min(10, observed_threshold))
                # EMA blend
                new_threshold = (1.0 - blend_factor) * rule.max_complexity + blend_factor * observed_threshold
                rule.max_complexity = int(round(new_threshold))

                logger.info(
                    "Feedback update: rule '%s' threshold %d → %d (rate=%.2f, blend=%.2f)",
                    rule.preferred_role,
                    band_max,
                    rule.max_complexity,
                    rate,
                    blend_factor,
                )

            prev_max = rule.max_complexity

    @classmethod
    def from_config(cls, config: dict) -> TriageRouter:
        """Build a ``TriageRouter`` from the ``savings.triage`` config dict.

        Expected shape::

            {
                "enabled": true,
                "rules": [
                    {"max_complexity": 3, "preferred_role": "coding_lite", "description": "..."},
                    ...
                ],
                "default_role": "coding"
            }
        """
        rules = [
            TriageRule(
                max_complexity=r["max_complexity"],
                preferred_role=r["preferred_role"],
                description=r.get("description", ""),
            )
            for r in config.get("rules", [])
        ]
        return cls(
            rules=rules,
            default_role=config.get("default_role", "coding"),
        )
