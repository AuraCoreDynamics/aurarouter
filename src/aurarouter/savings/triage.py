"""Smart triage routing — complexity-based role selection."""

from __future__ import annotations

from dataclasses import dataclass

from aurarouter._logging import get_logger

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
