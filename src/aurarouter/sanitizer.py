"""Response sanitizer for sovereignty enforcement.

Post-processes model responses to strip sovereignty-classified content
that may have leaked into model output. Uses regex patterns from the
sovereignty gate configuration.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader

logger = get_logger("AuraRouter.Sanitizer")


@dataclass
class SanitizationResult:
    """Result of sanitizing a response."""

    text: str
    was_sanitized: bool = False
    patterns_matched: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


class ResponseSanitizer:
    """Strips sovereignty-classified content from model responses.

    Uses the same patterns as SovereigntyGate to detect PII/sovereign
    content in LLM outputs and redact it before returning to callers.
    """

    # Built-in patterns for common PII types
    BUILTIN_PATTERNS: list[tuple[str, str]] = [
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        ("phone_us", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ]

    def __init__(self, config: ConfigLoader | None = None):
        self._patterns: list[tuple[str, re.Pattern]] = []
        self._load_patterns(config)

    def _load_patterns(self, config: ConfigLoader | None) -> None:
        """Load built-in + custom sovereignty patterns."""
        for name, pattern in self.BUILTIN_PATTERNS:
            try:
                self._patterns.append((name, re.compile(pattern)))
            except re.error:
                logger.warning("Invalid built-in pattern '%s': %s", name, pattern)

        if config is not None:
            custom = config.config.get("system", {}).get("sovereignty_patterns", [])
            for entry in custom:
                name = entry.get("name", "custom")
                pattern = entry.get("pattern", "")
                if pattern:
                    try:
                        self._patterns.append((name, re.compile(pattern)))
                    except re.error:
                        logger.warning("Invalid custom pattern '%s': %s", name, pattern)

    def sanitize(self, text: str) -> SanitizationResult:
        """Sanitize a response by redacting matched patterns.

        Returns:
            SanitizationResult with sanitized text, matched pattern names,
            and headers to add to the response.
        """
        if not text:
            return SanitizationResult(text=text)

        matched = []
        sanitized = text

        for name, pattern in self._patterns:
            if pattern.search(sanitized):
                matched.append(name)
                sanitized = pattern.sub(f"[REDACTED:{name}]", sanitized)

        was_sanitized = len(matched) > 0
        headers = {}
        if was_sanitized:
            headers["X-Sovereignty-Sanitized"] = "true"
            logger.info(
                "Sanitized response: %d patterns matched (%s)",
                len(matched), ", ".join(matched),
            )

        result = SanitizationResult(
            text=sanitized,
            was_sanitized=was_sanitized,
            patterns_matched=matched,
            headers=headers,
        )

        # Log structured sovereignty audit event
        self._log_sovereignty_decision(result)

        return result

    def _log_sovereignty_decision(self, result: SanitizationResult) -> None:
        """Log a structured sovereignty audit event matching the unified schema."""
        import json
        from datetime import datetime, timezone

        decision = "sanitize" if result.was_sanitized else "allow"
        audit_event = {
            "event_type": "sovereignty_decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": "aurarouter",
            "decision": decision,
            "resource_type": "response",
            "resource_id": "",
            "sovereignty_class": "sovereign" if result.was_sanitized else "open",
            "destination": "local",
            "reason": (
                f"Patterns matched: {', '.join(result.patterns_matched)}"
                if result.was_sanitized
                else "No sovereign content detected"
            ),
        }
        logger.debug("sovereignty_audit: %s", json.dumps(audit_event))
