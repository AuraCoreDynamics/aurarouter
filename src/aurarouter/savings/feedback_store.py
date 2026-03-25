"""SQLite-backed store for routing decision outcomes."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.FeedbackStore")

_DEFAULT_DB_PATH = Path.home() / ".auracore" / "aurarouter" / "feedback.db"

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    complexity REAL NOT NULL,
    model_id TEXT NOT NULL,
    success INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0
)
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model_id, timestamp)
"""


class FeedbackStore:
    """SQLite-backed store for routing decision outcomes."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()

    def close(self) -> None:
        """Close the persistent database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        role: str,
        complexity: float,
        model_id: str,
        success: bool,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a routing outcome."""
        timestamp = datetime.now(timezone.utc).isoformat()
        latency_ms = latency * 1000.0
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO feedback "
                "(timestamp, role, complexity, model_id, success, "
                "latency_ms, input_tokens, output_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    role,
                    complexity,
                    model_id,
                    int(success),
                    latency_ms,
                    input_tokens,
                    output_tokens,
                ),
            )
            conn.commit()

    def success_rate(
        self,
        model_id: str,
        complexity_min: float = 0,
        complexity_max: float = 10,
        window_days: int = 7,
    ) -> float:
        """Return success rate for a model within a complexity band.

        Returns 0.0 if no records match.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT COUNT(*), SUM(success) FROM feedback "
                "WHERE model_id = ? AND complexity >= ? AND complexity <= ? "
                "AND timestamp >= ?",
                (model_id, complexity_min, complexity_max, cutoff),
            ).fetchone()

        total, successes = row
        if not total:
            return 0.0
        return (successes or 0) / total

    def model_stats(self, window_days: int = 7) -> list[dict]:
        """Return aggregate stats per model: success_rate, avg_latency, call_count."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT model_id, COUNT(*), SUM(success), AVG(latency_ms) "
                "FROM feedback WHERE timestamp >= ? GROUP BY model_id",
                (cutoff,),
            ).fetchall()

        results = []
        for model_id, count, successes, avg_latency in rows:
            results.append({
                "model_id": model_id,
                "call_count": count,
                "success_rate": (successes or 0) / count if count else 0.0,
                "avg_latency_ms": avg_latency or 0.0,
            })
        return results
