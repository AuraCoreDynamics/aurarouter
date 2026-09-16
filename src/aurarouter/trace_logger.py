"""TraceLogger for appending execution traces as JSONL."""

import json
import logging
from pathlib import Path
from typing import Any

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.TraceLogger")


class TraceLogger:
    """Appends routing and execution traces to a session-specific JSONL file."""

    DEFAULT_TRACE_DIR = Path.home() / ".auracore" / "aurarouter" / "traces"

    def __init__(self, trace_dir: Path | None = None):
        self.trace_dir = trace_dir or self.DEFAULT_TRACE_DIR
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create trace directory %s: %s", self.trace_dir, exc)

    def _get_trace_file(self, session_id: str) -> Path:
        return self.trace_dir / f"{session_id}.jsonl"

    def log_stage(self, session_id: str, stage_type: str, payload: dict[str, Any]) -> None:
        """Append a single execution stage to the session's trace file.
        
        Args:
            session_id: The ID of the session.
            stage_type: Type of the stage (e.g. 'intent_classification', 'circuit_breaker', 'mcp_tool_granted').
            payload: JSON serializable dictionary containing the trace data.
        """
        if not session_id:
            return

        trace_file = self._get_trace_file(session_id)
        
        record = {
            "stage_type": stage_type,
            "payload": payload
        }
        
        try:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Failed to write to trace file %s: %s", trace_file, exc)

    def read_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Read and parse all trace records for a session."""
        if not session_id:
            return []

        trace_file = self._get_trace_file(session_id)
        if not trace_file.exists():
            return []

        records = []
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            logger.error("Failed to read trace file %s: %s", trace_file, exc)
            
        return records
