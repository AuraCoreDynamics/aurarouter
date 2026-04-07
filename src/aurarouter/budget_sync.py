"""Global budget synchronization protocol for AuraRouter.

T9.2 — Cross-project spend reconciliation via BudgetSyncMessage.

Usage:
    Each AuraCore service (AuraRouter, AuraXLM, AuraGrid) periodically
    reports its local token and cost spend via the report_budget_sync MCP
    tool.  AuraRouter stores the last message per source (last-write-wins)
    and exposes a merged view through get_global_budget.

    No persistent storage — the store is in-memory and eventually consistent.
    It resets on AuraRouter restart.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.BudgetSync")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BudgetSyncMessage(BaseModel):
    """Cross-project budget report. Immutable wire model."""

    model_config = ConfigDict(frozen=True)

    source: Literal["aurarouter", "auraxlm", "auragrid"]
    period_start: datetime
    period_end: datetime
    token_spend: dict  # {"input": int, "output": int}
    inference_cost_usd: float = 0.0
    compute_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class BudgetSyncStore:
    """Thread-safe in-memory budget sync store.

    Keyed by source — last BudgetSyncMessage per source wins.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: dict[str, BudgetSyncMessage] = {}

    def record(self, msg: BudgetSyncMessage) -> None:
        """Store or overwrite the message for msg.source."""
        with self._lock:
            self._messages[msg.source] = msg
        logger.info(
            "BudgetSync from %s: input=%d output=%d inference=$%.4f compute=$%.4f",
            msg.source,
            msg.token_spend.get("input", 0),
            msg.token_spend.get("output", 0),
            msg.inference_cost_usd,
            msg.compute_cost_usd,
        )

    def get_all(self) -> list[BudgetSyncMessage]:
        """Return a snapshot of all stored messages."""
        with self._lock:
            return list(self._messages.values())

    def reset(self) -> None:
        """Clear all stored messages."""
        with self._lock:
            self._messages.clear()


# ---------------------------------------------------------------------------
# MCP-callable helpers
# ---------------------------------------------------------------------------


def report_budget_sync_fn(store: BudgetSyncStore, payload_json: str) -> str:
    """Accept and store a BudgetSyncMessage payload.

    Args:
        store: The shared BudgetSyncStore instance.
        payload_json: JSON string conforming to BudgetSyncMessage.

    Returns:
        JSON string: {"ok": true, "source": "..."} or {"error": "..."}.
    """
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    try:
        msg = BudgetSyncMessage.model_validate(data)
    except Exception as exc:
        return json.dumps({"error": f"Validation failed: {exc}"})

    store.record(msg)
    return json.dumps({"ok": True, "source": msg.source})


def get_global_budget_fn(store: BudgetSyncStore) -> str:
    """Return a merged view across all reported budget sync messages.

    Response:
    {
        "period_start": "...",
        "period_end": "...",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_inference_cost_usd": 0.0,
        "total_compute_cost_usd": 0.0,
        "sources_reported": ["aurarouter", "auraxlm"]
    }
    """
    messages = store.get_all()
    if not messages:
        return json.dumps(
            {
                "period_start": None,
                "period_end": None,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_inference_cost_usd": 0.0,
                "total_compute_cost_usd": 0.0,
                "sources_reported": [],
            }
        )

    min_start = min(m.period_start for m in messages)
    max_end = max(m.period_end for m in messages)
    total_input = sum(m.token_spend.get("input", 0) for m in messages)
    total_output = sum(m.token_spend.get("output", 0) for m in messages)
    total_inference = sum(m.inference_cost_usd for m in messages)
    total_compute = sum(m.compute_cost_usd for m in messages)

    return json.dumps(
        {
            "period_start": min_start.isoformat(),
            "period_end": max_end.isoformat(),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_inference_cost_usd": round(total_inference, 6),
            "total_compute_cost_usd": round(total_compute, 6),
            "sources_reported": [m.source for m in messages],
        },
        indent=2,
    )
