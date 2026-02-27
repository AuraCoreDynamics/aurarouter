"""Privacy audit engine for detecting sensitive data in cloud-bound prompts."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DB_PATH = Path.home() / ".auracore" / "aurarouter" / "usage.db"

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS privacy_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    model_id        TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    match_count     INTEGER NOT NULL,
    severities      TEXT    NOT NULL,
    pattern_names   TEXT    NOT NULL,
    prompt_length   INTEGER NOT NULL,
    recommendation  TEXT    NOT NULL
)
"""


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class PrivacyPattern:
    """A single regex pattern for detecting sensitive data."""

    name: str
    pattern: str
    severity: str  # "low", "medium", "high"
    description: str


@dataclass
class PrivacyMatch:
    """A single match found in a prompt."""

    pattern_name: str
    severity: str
    matched_text: str  # redacted: first 4 chars + "***"
    position: int  # character offset


@dataclass
class PrivacyEvent:
    """An audit event recording all matches found in a single prompt."""

    timestamp: str  # ISO 8601
    model_id: str
    provider: str
    matches: list[PrivacyMatch]
    prompt_length: int
    recommendation: str


# ── Built-in patterns ────────────────────────────────────────────────

_BUILTIN_PATTERNS: list[PrivacyPattern] = [
    PrivacyPattern(
        name="Email Address",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        severity="medium",
        description="Email address detected in prompt.",
    ),
    PrivacyPattern(
        name="API Key",
        pattern=r'(?i)(?:api[_-]?key|token|secret|password)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}["\']?',
        severity="high",
        description="Possible API key or secret detected in prompt.",
    ),
    PrivacyPattern(
        name="AWS Access Key",
        pattern=r"AKIA[0-9A-Z]{16}",
        severity="high",
        description="AWS access key ID detected in prompt.",
    ),
    PrivacyPattern(
        name="SSN",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        severity="high",
        description="Social Security Number detected in prompt.",
    ),
    PrivacyPattern(
        name="Credit Card",
        pattern=r"\b(?:\d{4}[- ]?){3}\d{4}\b",
        severity="high",
        description="Possible credit card number detected in prompt.",
    ),
    PrivacyPattern(
        name="Confidential Marker",
        pattern=r"(?i)\b(?:confidential|classified|top\s+secret|internal\s+only|proprietary)\b",
        severity="medium",
        description="Confidentiality marker detected in prompt.",
    ),
    PrivacyPattern(
        name="Private IP Address",
        pattern=r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b",
        severity="low",
        description="Private/internal IP address detected in prompt.",
    ),
]


# ── Auditor ──────────────────────────────────────────────────────────


def _redact(text: str) -> str:
    """Return first 4 characters + '***'."""
    if len(text) <= 4:
        return text + "***"
    return text[:4] + "***"


class PrivacyAuditor:
    """Scans prompts for sensitive data before they leave for cloud providers."""

    def __init__(
        self, custom_patterns: Optional[list[PrivacyPattern]] = None
    ) -> None:
        all_patterns = list(_BUILTIN_PATTERNS)
        if custom_patterns:
            all_patterns.extend(custom_patterns)
        self._patterns: list[tuple[PrivacyPattern, re.Pattern[str]]] = [
            (p, re.compile(p.pattern)) for p in all_patterns
        ]

    @staticmethod
    def is_cloud_provider(provider: str) -> bool:
        """Return ``True`` for cloud providers (``google``, ``claude``).

        .. deprecated::
            Use ``aurarouter.savings.pricing.is_cloud_tier()`` instead,
            which resolves via the model's ``hosting_tier`` config field.
        """
        from aurarouter.savings.pricing import is_cloud_tier

        return is_cloud_tier(None, provider)

    def audit(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        hosting_tier: str | None = None,
    ) -> PrivacyEvent | None:
        """Run all patterns against *prompt*.

        Returns a ``PrivacyEvent`` if any matches are found, else ``None``.
        Only audits cloud-bound prompts. Uses *hosting_tier* (if provided)
        or falls back to provider-name classification.
        """
        from aurarouter.savings.pricing import is_cloud_tier

        if not is_cloud_tier(hosting_tier, provider):
            return None

        matches: list[PrivacyMatch] = []
        for pat, compiled in self._patterns:
            for m in compiled.finditer(prompt):
                matches.append(
                    PrivacyMatch(
                        pattern_name=pat.name,
                        severity=pat.severity,
                        matched_text=_redact(m.group()),
                        position=m.start(),
                    )
                )

        if not matches:
            return None

        return PrivacyEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            provider=provider,
            matches=matches,
            prompt_length=len(prompt),
            recommendation="Consider routing to a local model",
        )


# ── Persistence ──────────────────────────────────────────────────────


class PrivacyStore:
    """Thread-safe SQLite store for privacy audit events."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event: PrivacyEvent) -> None:
        """Insert a single privacy event."""
        severities = [m.severity for m in event.matches]
        pattern_names = [m.pattern_name for m in event.matches]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO privacy_events "
                    "(timestamp, model_id, provider, match_count, "
                    "severities, pattern_names, prompt_length, recommendation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.timestamp,
                        event.model_id,
                        event.provider,
                        len(event.matches),
                        json.dumps(severities),
                        json.dumps(pattern_names),
                        event.prompt_length,
                        event.recommendation,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def query(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> list[dict]:
        """Return matching events with optional filters."""
        clauses: list[str] = []
        params: list[object] = []

        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, timestamp, model_id, provider, match_count, "
            "severities, pattern_names, prompt_length, recommendation "
            f"FROM privacy_events{where} ORDER BY timestamp"
        )

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        _SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
        min_rank = _SEVERITY_RANK.get(min_severity or "", -1)

        results: list[dict] = []
        for r in rows:
            severities = json.loads(r[5])
            max_sev = max(
                (_SEVERITY_RANK.get(s, 0) for s in severities), default=0
            )
            if max_sev < min_rank:
                continue
            results.append(
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "model_id": r[2],
                    "provider": r[3],
                    "match_count": r[4],
                    "severities": severities,
                    "pattern_names": json.loads(r[6]),
                    "prompt_length": r[7],
                    "recommendation": r[8],
                }
            )
        return results

    def summary(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        """Aggregate summary of privacy events.

        Returns ``{"total_events": int, "by_severity": {...},
        "by_pattern": {...}}``.
        """
        events = self.query(start=start, end=end)
        by_severity: dict[str, int] = {}
        by_pattern: dict[str, int] = {}
        for ev in events:
            for sev in ev["severities"]:
                by_severity[sev] = by_severity.get(sev, 0) + 1
            for name in ev["pattern_names"]:
                by_pattern[name] = by_pattern.get(name, 0) + 1

        return {
            "total_events": len(events),
            "by_severity": by_severity,
            "by_pattern": by_pattern,
        }
