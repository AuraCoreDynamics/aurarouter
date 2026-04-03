"""SQLite-backed session persistence. Follows the UsageStore pattern."""

from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from aurarouter.sessions.models import Session


class SessionStore:
    """SQLite storage for sessions with cross-process locking (Task 6.1).

    Pattern: one thread lock + one file lock, connection-per-call.
    """

    DEFAULT_PATH = Path.home() / ".auracore" / "aurarouter" / "sessions.db"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DEFAULT_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.db_path.with_suffix(".db.lock")
        self._thread_lock = threading.Lock()
        self._init_schema()

    def _file_lock(self):
        """Cross-process file lock context manager."""
        import os
        import time
        from contextlib import contextmanager

        @contextmanager
        def _lock():
            while True:
                try:
                    fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    try:
                        yield
                    finally:
                        os.close(fd)
                        os.remove(self.lock_path)
                    break
                except FileExistsError:
                    # Check if lock is stale (older than 10s)
                    try:
                        if time.time() - os.path.getmtime(self.lock_path) > 10:
                            os.remove(self.lock_path)
                            continue
                    except OSError:
                        pass
                    time.sleep(0.05)
        return _lock()

    def _init_schema(self) -> None:
        with self._thread_lock, self._file_lock():
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def save(self, session: Session) -> None:
        """Save or update a session."""
        with self._thread_lock, self._file_lock():
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO sessions
                       (session_id, data, created_at, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        session.session_id,
                        json.dumps(session.to_dict()),
                        session.created_at,
                        session.updated_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID. Returns None if not found."""
        with self._thread_lock, self._file_lock():
            conn = sqlite3.connect(str(self.db_path))
            try:
                row = conn.execute(
                    "SELECT data FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                return Session.from_dict(json.loads(row[0]))
            finally:
                conn.close()

    def list_sessions(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """List sessions (metadata only: session_id, created_at, updated_at)."""
        with self._thread_lock, self._file_lock():
            conn = sqlite3.connect(str(self.db_path))
            try:
                rows = conn.execute(
                    """SELECT session_id, created_at, updated_at
                       FROM sessions
                       ORDER BY updated_at DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()
                return [
                    {
                        "session_id": r[0],
                        "created_at": r[1],
                        "updated_at": r[2],
                    }
                    for r in rows
                ]
            finally:
                conn.close()

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was deleted."""
        with self._thread_lock, self._file_lock():
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def purge_before(self, timestamp: str) -> int:
        """Delete sessions last updated before timestamp. Returns count deleted."""
        with self._thread_lock, self._file_lock():
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE updated_at < ?",
                    (timestamp,),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
