"""SQLite-backed usage persistence for AuraRouter token tracking."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from aurarouter.savings.models import UsageRecord

_DEFAULT_DB_PATH = Path.home() / ".auracore" / "aurarouter" / "usage.db"

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    model_id    TEXT    NOT NULL,
    provider    TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    intent      TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    elapsed_s   REAL    NOT NULL,
    success     INTEGER NOT NULL,
    is_cloud    INTEGER NOT NULL
)
"""


class UsageStore:
    """Thread-safe, connection-per-call SQLite store for usage records."""

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

    def record(self, usage: UsageRecord) -> None:
        """Insert a single usage record."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO usage "
                    "(timestamp, model_id, provider, role, intent, "
                    "input_tokens, output_tokens, elapsed_s, success, is_cloud) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        usage.timestamp,
                        usage.model_id,
                        usage.provider,
                        usage.role,
                        usage.intent,
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.elapsed_s,
                        int(usage.success),
                        int(usage.is_cloud),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def query(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        role: Optional[str] = None,
    ) -> list[UsageRecord]:
        """Return matching records with optional filters."""
        clauses: list[str] = []
        params: list[object] = []

        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if role is not None:
            clauses.append("role = ?")
            params.append(role)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT timestamp, model_id, provider, role, intent, input_tokens, output_tokens, elapsed_s, success, is_cloud FROM usage{where} ORDER BY timestamp"

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        return [
            UsageRecord(
                timestamp=r[0],
                model_id=r[1],
                provider=r[2],
                role=r[3],
                intent=r[4],
                input_tokens=r[5],
                output_tokens=r[6],
                elapsed_s=r[7],
                success=bool(r[8]),
                is_cloud=bool(r[9]),
            )
            for r in rows
        ]

    def aggregate_tokens(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        group_by: str = "model_id",
    ) -> list[dict]:
        """SUM(input_tokens), SUM(output_tokens) grouped by *group_by* column."""
        allowed = {"model_id", "provider", "role", "intent"}
        if group_by not in allowed:
            raise ValueError(f"group_by must be one of {allowed}")

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
            f"SELECT {group_by}, SUM(input_tokens), SUM(output_tokens) "
            f"FROM usage{where} GROUP BY {group_by}"
        )

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        return [
            {
                group_by: r[0],
                "input_tokens": r[1],
                "output_tokens": r[2],
                "total_tokens": r[1] + r[2],
            }
            for r in rows
        ]

    def total_tokens(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        """Return total input, output, and combined token counts."""
        clauses: list[str] = []
        params: list[object] = []
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM usage{where}"

        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
        finally:
            conn.close()

        inp, out = row  # type: ignore[misc]
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    def purge_before(self, timestamp: str) -> int:
        """Delete records older than *timestamp*. Return number deleted."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM usage WHERE timestamp < ?", (timestamp,)
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
