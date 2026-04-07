"""Three-phase service discovery handshake for AuraRouter.

T9.1 — RegistrationStore manages per-service handshake state.

Protocol:
  1. ANNOUNCE  — service POSTs capabilities via aurarouter.catalog.register
                 with handshake_version=1.  AuraRouter returns catalog_id ACK.
  2. READY     — service POSTs catalog_id to /api/registration/ready once
                 it has confirmed the ACK.  AuraRouter marks service operational.
  3. Gate      — xlm_gate_check() returns an AuraError payload when XLM-
                 dependent tools are called before AuraXLM is operational.

Backward compatibility:
  When handshake_version is absent (0), AuraRouter returns the old-style
  {"success": true} response and no state is stored in RegistrationStore.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from aurarouter._logging import get_logger
from aurarouter.errors import AuraError

logger = get_logger("AuraRouter.Registration")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class _ServiceRecord:
    name: str
    catalog_id: str
    status: str = "pending"                              # "pending" | "operational"
    accepted_capabilities: list[str] = field(default_factory=list)
    registered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class RegistrationStore:
    """Thread-safe in-memory registry for service discovery handshake state.

    Last-write-wins per service name — re-announcing overwrites the previous
    record (e.g., on service restart) with a fresh catalog_id.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, _ServiceRecord] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def announce(self, name: str, capabilities: list[str]) -> tuple[str, list[str]]:
        """Register (or re-register) a service.

        Returns:
            (catalog_id, accepted_capabilities)
        """
        catalog_id = str(uuid4())
        with self._lock:
            self._records[name] = _ServiceRecord(
                name=name,
                catalog_id=catalog_id,
                accepted_capabilities=list(capabilities),
            )
        logger.info(
            "Registration ANNOUNCE: service=%s catalog_id=%s", name, catalog_id
        )
        return catalog_id, list(capabilities)

    def mark_ready(self, catalog_id: str) -> bool:
        """Mark service as operational by catalog_id.

        Returns:
            True if the catalog_id was found; False otherwise.
        """
        with self._lock:
            for record in self._records.values():
                if record.catalog_id == catalog_id:
                    record.status = "operational"
                    logger.info(
                        "Registration READY: service=%s catalog_id=%s",
                        record.name,
                        catalog_id,
                    )
                    return True
        logger.warning(
            "Registration READY: unknown catalog_id=%s", catalog_id
        )
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_operational(self, name: str) -> bool:
        """Return True if the named service has completed the READY phase."""
        with self._lock:
            rec = self._records.get(name)
            return rec is not None and rec.status == "operational"

    def get_status(self) -> list[dict]:
        """Return a snapshot of all service registration entries."""
        with self._lock:
            return [
                {
                    "name": r.name,
                    "status": r.status,
                    "catalog_id": r.catalog_id,
                }
                for r in self._records.values()
            ]


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------


def xlm_gate_check(store: RegistrationStore) -> str | None:
    """Return an AuraError JSON string if AuraXLM has not completed READY.

    Returns None when XLM is operational (gate passes).
    """
    if not store.is_operational("auraxlm"):
        err = AuraError(
            error_code=1005,
            category="infrastructure",
            message="AuraXLM not yet operational",
            source_project="aurarouter",
        )
        return err.model_dump_json()
    return None


# ---------------------------------------------------------------------------
# MCP-callable helpers
# ---------------------------------------------------------------------------


def registration_ready_fn(store: RegistrationStore, catalog_id: str) -> str:
    """Handle a READY notification by catalog_id.

    Args:
        store: The shared RegistrationStore instance.
        catalog_id: The catalog_id received in the ANNOUNCE ACK.

    Returns:
        JSON string: {"ok": true, "catalog_id": "..."} or {"error": "..."}.
    """
    if not catalog_id:
        return json.dumps({"error": "catalog_id is required"})
    found = store.mark_ready(catalog_id)
    if found:
        return json.dumps({"ok": True, "catalog_id": catalog_id})
    return json.dumps({"error": f"Unknown catalog_id: {catalog_id}"})


def discovery_status_fn(store: RegistrationStore) -> str:
    """Return all service registration statuses.

    Returns:
        JSON: {"services": [{"name": "auraxlm", "status": "operational|pending|unregistered",
               "catalog_id": "..."}, ...]}
    """
    return json.dumps({"services": store.get_status()}, indent=2)
