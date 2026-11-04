"""TraceLogger for appending execution traces as JSONL."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.TraceLogger")


class TraceLogger:
    """Appends routing and execution traces to a session-specific JSONL file.

    File appends are serialized with a lock (preventing interleaved JSONL lines
    or Windows sharing violations across concurrent requests sharing a session)
    and optionally offloaded asynchronously via an :class:`EventReporter` so
    hot-path routing never blocks on disk I/O.
    """

    DEFAULT_TRACE_DIR = Path.home() / ".auracore" / "aurarouter" / "traces"

    def __init__(
        self,
        trace_dir: Path | None = None,
        event_reporter: Any | None = None,
    ) -> None:
        self.trace_dir = trace_dir or self.DEFAULT_TRACE_DIR
        self._event_reporter = event_reporter
        self._lock = threading.Lock()
        self._drain_cond = threading.Condition(self._lock)
        self._pending_writes = 0
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create trace directory %s: %s", self.trace_dir, exc)

    def _get_trace_file(self, session_id: str) -> Path:
        return self.trace_dir / f"{session_id}.jsonl"

    def _write_line(self, trace_file: Path, serialized_line: str) -> None:
        with self._lock:
            try:
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(serialized_line)
            except Exception as exc:
                logger.error("Failed to write to trace file %s: %s", trace_file, exc)

    def log_stage(self, session_id: str, stage_type: str, payload: dict[str, Any]) -> None:
        """Append a single execution stage to the session's trace file.

        Args:
            session_id: The ID of the session.
            stage_type: Type of the stage (e.g. 'intent_classification', 'circuit_breaker').
            payload: JSON serializable dictionary containing the trace data.
        """
        if not session_id:
            return

        trace_file = self._get_trace_file(session_id)
        try:
            serialized = json.dumps({"stage_type": stage_type, "payload": payload}) + "\n"
        except Exception as exc:
            logger.error("Failed to serialize trace stage %s: %s", stage_type, exc)
            return

        reporter = self._event_reporter
        if reporter is not None and hasattr(reporter, "submit"):
            with self._lock:
                self._pending_writes += 1

            def _async_write() -> None:
                try:
                    self._write_line(trace_file, serialized)
                finally:
                    with self._drain_cond:
                        self._pending_writes = max(0, self._pending_writes - 1)
                        if self._pending_writes == 0:
                            self._drain_cond.notify_all()

            try:
                reporter.submit(_async_write)
                return
            except Exception:
                with self._drain_cond:
                    self._pending_writes = max(0, self._pending_writes - 1)
                    if self._pending_writes == 0:
                        self._drain_cond.notify_all()

        self._write_line(trace_file, serialized)

    def flush(self, timeout: float = 2.0) -> None:
        """Wait for any queued asynchronous trace writes to hit disk."""
        with self._drain_cond:
            if self._pending_writes > 0:
                self._drain_cond.wait_for(lambda: self._pending_writes == 0, timeout=timeout)

    def read_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Read and parse all trace records for a session."""
        if not session_id:
            return []

        self.flush()
        trace_file = self._get_trace_file(session_id)

        records: list[dict[str, Any]] = []
        with self._lock:
            if not trace_file.exists():
                return []
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception as exc:
                logger.error("Failed to read trace file %s: %s", trace_file, exc)

        return records
