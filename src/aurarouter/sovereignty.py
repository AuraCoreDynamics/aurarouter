"""Sovereignty enforcement gate for AuraRouter.

Evaluates prompts for sovereignty-sensitive content (PII, classification
markers, custom patterns) and forces routing to local-only models when
triggered.  Sits *before* model selection in the ComputeFabric execute
loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from aurarouter.config import ConfigLoader
from aurarouter.savings.privacy import PrivacyAuditor, PrivacyPattern

logger = logging.getLogger(__name__)


class SovereigntyVerdict(str, Enum):
    """Result of sovereignty evaluation."""

    OPEN = "open"  # no restrictions, any model is fine
    SOVEREIGN = "sovereign"  # route to local models only
    BLOCKED = "blocked"  # reject entirely (reserved for future use)


@dataclass(frozen=True)
class SovereigntyResult:
    """Complete sovereignty evaluation result."""

    verdict: SovereigntyVerdict
    reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


class SovereigntyGate:
    """Evaluate prompts and enforce local-only routing when needed.

    Uses the existing :class:`PrivacyAuditor` patterns plus optional
    additional ``sovereignty_patterns`` from config.  When PII or
    sovereignty markers are found, the chain is filtered to local-only
    models.
    """

    def __init__(
        self,
        config: ConfigLoader,
        privacy_auditor: PrivacyAuditor | None = None,
    ) -> None:
        self._config = config
        self._privacy_auditor = privacy_auditor or PrivacyAuditor()
        self._extra_patterns = self._load_extra_patterns()

    def is_enabled(self) -> bool:
        """Check if sovereignty enforcement is enabled in config."""
        system = self._config.config.get("system", {})
        return system.get("sovereignty_enforcement", False)

    def evaluate(self, prompt: str) -> SovereigntyResult:
        """Evaluate a prompt for sovereignty-sensitive content.

        Returns OPEN if nothing sensitive is found, or SOVEREIGN if any
        privacy/sovereignty pattern matches.
        """
        if not self.is_enabled():
            return SovereigntyResult(verdict=SovereigntyVerdict.OPEN)

        matched: list[str] = []

        # Check via PrivacyAuditor (runs all built-in + custom patterns).
        # We pass a dummy cloud-tier model so the auditor actually runs.
        event = self._privacy_auditor.audit(
            prompt,
            model_id="sovereignty-check",
            provider="cloud-dummy",
            hosting_tier="cloud",
        )
        if event and event.matches:
            matched.extend(m.pattern_name for m in event.matches)

        # Check extra sovereignty-specific patterns.
        for pat, compiled in self._extra_patterns:
            if compiled.search(prompt):
                matched.append(pat.name)

        if matched:
            unique = list(dict.fromkeys(matched))  # dedupe, preserve order
            return SovereigntyResult(
                verdict=SovereigntyVerdict.SOVEREIGN,
                reason=f"Sovereignty trigger: {', '.join(unique)}",
                matched_patterns=unique,
            )

        return SovereigntyResult(verdict=SovereigntyVerdict.OPEN)

    def enforce(
        self,
        chain: list[str],
        config: ConfigLoader,
        result: SovereigntyResult,
    ) -> list[str]:
        """Filter a model chain based on a sovereignty result.

        Returns only local (non-cloud) models when verdict is SOVEREIGN.
        Returns the original chain for OPEN.
        Raises ``SovereigntyViolationError`` for BLOCKED.
        """
        if result.verdict == SovereigntyVerdict.OPEN:
            return chain

        if result.verdict == SovereigntyVerdict.BLOCKED:
            raise SovereigntyViolationError(result.reason)

        # SOVEREIGN: filter to local models only.
        from aurarouter.savings.pricing import is_cloud_tier

        local: list[str] = []
        for model_id in chain:
            model_cfg = config.get_model_config(model_id)
            if not model_cfg:
                continue
            hosting_tier = model_cfg.get("hosting_tier")
            provider = model_cfg.get("provider", "")
            if not is_cloud_tier(hosting_tier, provider):
                local.append(model_id)

        if not local:
            logger.warning(
                "Sovereignty gate filtered to 0 local models. "
                "All chain models are cloud-hosted."
            )

        logger.info(
            "Sovereignty gate: %s → filtered chain from %d to %d models.",
            result.verdict.value,
            len(chain),
            len(local),
        )
        return local

    def _load_extra_patterns(self) -> list[tuple[PrivacyPattern, "re.Pattern"]]:
        """Load additional sovereignty patterns from config.

        Config format::

            system:
              sovereignty_patterns:
                - name: "FOUO Marker"
                  pattern: "(?i)\\bfor\\s+official\\s+use\\s+only\\b"
                  severity: "high"
                  description: "FOUO marking detected"
        """
        import re

        raw = (
            self._config.config
            .get("system", {})
            .get("sovereignty_patterns", [])
        )
        patterns = []
        for entry in raw:
            try:
                pat = PrivacyPattern(
                    name=entry["name"],
                    pattern=entry["pattern"],
                    severity=entry.get("severity", "high"),
                    description=entry.get("description", ""),
                )
                patterns.append((pat, re.compile(pat.pattern)))
            except (KeyError, re.error) as exc:
                logger.warning("Skipping invalid sovereignty pattern: %s", exc)
        return patterns


class SovereigntyViolationError(Exception):
    """Raised when a prompt is BLOCKED by the sovereignty gate."""
