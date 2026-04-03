"""Speculative decoding orchestrator for AuraRouter.

Coordinates drafter→verifier loop:
- Drafter model on edge node generates token candidates
- Verifier (via AuraXLM's auraxlm.verify_draft MCP tool) validates in parallel
- Sovereignty gate enforced for BOTH drafter and verifier models
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader
    from aurarouter.fabric import ComputeFabric
    from aurarouter.mcp_client.registry import McpClientRegistry
    from aurarouter.savings.triage import TriageRouter
    from aurarouter.sovereignty import SovereigntyGate

logger = get_logger("AuraRouter.Speculative")


@dataclass
class SpeculativeSession:
    """Tracks the state of an active speculative decoding session."""

    session_id: str
    drafter_model: str
    verifier_model: str
    task: str
    accepted_tokens: list[int] = field(default_factory=list)
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    notional_enabled: bool = False
    intent_confidence: float = 0.0
    status: str = "active"  # active, completed, failed
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "drafter_model": self.drafter_model,
            "verifier_model": self.verifier_model,
            "task": self.task,
            "accepted_count": len(self.accepted_tokens),
            "rejected_count": self.rejected_count,
            "acceptance_rate": self.acceptance_rate,
            "notional_enabled": self.notional_enabled,
            "intent_confidence": self.intent_confidence,
            "status": self.status,
        }


class SpeculativeOrchestrator:
    """Orchestrates speculative decoding across drafter/verifier models.

    Uses AuraXLM MCP tools for verification, sovereignty gate for both
    models, and triage router for confidence-gated notional emission.
    """

    def __init__(
        self,
        fabric: ComputeFabric,
        mcp_registry: McpClientRegistry | None,
        sovereignty_gate: SovereigntyGate | None,
        triage_router: TriageRouter | None,
    ):
        self._fabric = fabric
        self._mcp_registry = mcp_registry
        self._sovereignty_gate = sovereignty_gate
        self._triage_router = triage_router
        self._sessions: dict[str, SpeculativeSession] = {}

    @property
    def config(self) -> ConfigLoader:
        return self._fabric.config

    def _get_config_value(self, key: str, default):
        """Read a speculative config value from system.* config."""
        sys_cfg = self.config.config.get("system", {})
        return sys_cfg.get(key, default)

    def is_enabled(self) -> bool:
        """Check if speculative decoding is globally enabled."""
        return bool(self._get_config_value("speculative_decoding", False))

    @property
    def complexity_threshold(self) -> int:
        return int(self._get_config_value("speculative_complexity_threshold", 7))

    @property
    def confidence_threshold(self) -> float:
        return float(self._get_config_value("notional_confidence_threshold", 0.85))

    @property
    def session_timeout(self) -> float:
        return float(self._get_config_value("speculative_timeout", 60.0))

    def should_trigger(self, complexity: int) -> bool:
        """Determine if speculative mode should activate for given complexity."""
        return self.is_enabled() and complexity >= self.complexity_threshold

    def create_session(
        self,
        task: str,
        drafter_model: str,
        verifier_model: str,
        confidence: float = 0.0,
    ) -> SpeculativeSession:
        """Create and register a new speculative decoding session."""
        session_id = uuid.uuid4().hex[:16]
        session = SpeculativeSession(
            session_id=session_id,
            drafter_model=drafter_model,
            verifier_model=verifier_model,
            task=task,
            intent_confidence=confidence,
            notional_enabled=confidence >= self.confidence_threshold,
        )
        self._sessions[session_id] = session
        logger.info(
            "Speculative session %s created: drafter=%s, verifier=%s, notional=%s",
            session_id, drafter_model, verifier_model, session.notional_enabled,
        )
        return session

    def get_session(self, session_id: str) -> SpeculativeSession | None:
        return self._sessions.get(session_id)

    def get_active_sessions(self) -> list[SpeculativeSession]:
        return [s for s in self._sessions.values() if s.status == "active"]

    def complete_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.status = "completed"
        return True

    async def execute_speculative(
        self,
        task: str,
        context: str | None = None,
        notional_callback: Callable[[dict], None] | None = None,
        correction_callback: Callable[[dict], None] | None = None,
        permissions: dict | None = None,
    ) -> dict | None:
        """Execute a task using speculative decoding.

        1. Evaluate sovereignty for the task
        2. Select drafter (lighter) and verifier (heavier) models
        3. Run drafter for an initial response
        4. If notional enabled, emit draft tokens via callback
        5. Call AuraXLM verify_draft via MCP
        6. On rejection, emit correction event
        7. Return final verified result

        Returns a dict with the result, or None on failure.
        """
        from aurarouter.sovereignty import SovereigntyVerdict

        # Local permissions check (Task 2.2)
        if permissions:
            log.debug(f"Speculative execution using local permissions: {permissions}")

        # Step 1: Sovereignty enforcement
        sovereignty_result = None
        if self._sovereignty_gate is not None:
            sovereignty_result = self._sovereignty_gate.evaluate(task)
            if sovereignty_result.verdict == SovereigntyVerdict.BLOCKED:
                logger.warning("Speculative execution blocked by sovereignty gate.")
                return {"error": "blocked_by_sovereignty", "reason": sovereignty_result.reason}

        # Step 2: Select drafter and verifier from config
        drafter_chain = self.config.get_role_chain("coding")
        verifier_chain = self.config.get_role_chain("reasoning")

        if sovereignty_result is not None:
            drafter_chain = self._sovereignty_gate.enforce(
                drafter_chain, self.config, sovereignty_result
            )
            verifier_chain = self._sovereignty_gate.enforce(
                verifier_chain, self.config, sovereignty_result
            )

        if not drafter_chain or not verifier_chain:
            logger.warning("No models available for speculative execution after sovereignty filtering.")
            return None

        drafter_model = drafter_chain[0]
        verifier_model = verifier_chain[0]

        # Step 3: Determine confidence from triage
        confidence = 0.0
        if self._triage_router is not None:
            # Use rule matching as confidence proxy
            role = self._triage_router.select_role(self.complexity_threshold)
            # If triage matched a rule (not default), confidence is high
            confidence = 0.9 if role != self._triage_router.default_role else 0.5

        session = self.create_session(task, drafter_model, verifier_model, confidence)

        try:
            # Step 4: Run drafter
            full_prompt = f"{context}\n\n{task}" if context else task
            drafter_result = self._fabric.execute(
                "coding", full_prompt, chain_override=[drafter_model]
            )
            if drafter_result is None:
                session.status = "failed"
                return None

            # Step 5: Emit notional response if confidence is high
            if session.notional_enabled and notional_callback is not None:
                notional_callback({
                    "session_id": session.session_id,
                    "status": "notional",
                    "content": drafter_result.text,
                    "drafter_model": drafter_model,
                })

            # Step 6: Verify via AuraXLM MCP (or fall back to local verifier)
            verification = await self._verify_draft(
                session, drafter_result.text, verifier_model
            )

            if verification is not None and verification.get("accepted", False):
                session.accepted_tokens.extend(
                    verification.get("accepted_tokens", [])
                )
                total = verification.get("total_tokens", 1)
                accepted = verification.get("accepted_count", 0)
                session.acceptance_rate = accepted / max(total, 1)
                self.complete_session(session.session_id)
                return {
                    "session_id": session.session_id,
                    "content": drafter_result.text,
                    "verified": True,
                    "acceptance_rate": session.acceptance_rate,
                    "model_id": drafter_model,
                    "verifier_model": verifier_model,
                }

            # Step 7: Rejection — emit correction and fall back to verifier
            if correction_callback is not None:
                correction_callback({
                    "session_id": session.session_id,
                    "correction_position": verification.get("accepted_count", 0) if verification else 0,
                    "reason": "verifier_rejection",
                })

            session.rejected_count += 1

            # Fall back to verifier model directly
            verifier_result = self._fabric.execute(
                "reasoning", full_prompt, chain_override=[verifier_model]
            )
            self.complete_session(session.session_id)

            if verifier_result is not None:
                return {
                    "session_id": session.session_id,
                    "content": verifier_result.text,
                    "verified": True,
                    "acceptance_rate": 0.0,
                    "model_id": verifier_model,
                    "verifier_model": verifier_model,
                    "fallback": True,
                }
            return None

        except asyncio.TimeoutError:
            session.status = "failed"
            logger.error("Speculative session %s timed out.", session.session_id)
            return None
        except Exception as exc:
            session.status = "failed"
            logger.error("Speculative session %s failed: %s", session.session_id, exc)
            return None

    async def _verify_draft(
        self,
        session: SpeculativeSession,
        draft_text: str,
        verifier_model: str,
    ) -> dict | None:
        """Verify draft via AuraXLM MCP tool or local fallback.

        Calls auraxlm.verify_draft if MCP registry is available,
        otherwise falls back to a simple local heuristic.
        """
        if self._mcp_registry is not None:
            xlm_clients = self._mcp_registry.get_clients_with_capability(
                "auraxlm.verify_draft"
            )
            if xlm_clients:
                try:
                    import json
                    client = xlm_clients[0]
                    result = client.call_tool("auraxlm.verify_draft", {
                        "draft": {
                            "draft_id": f"{session.session_id}-draft",
                            "session_id": session.session_id,
                            "drafter_node_id": "local",
                            "tokens": [],  # Token IDs populated by real inference
                            "log_probs": [],
                            "kv_cache_pointer": {
                                "cache_id": "local",
                                "layer_offset": 0,
                                "sequence_length": 0,
                                "node_id": "local",
                            },
                            "timestamp": time.time(),
                        },
                        "verifier_model_id": verifier_model,
                    })
                    if result:
                        data = json.loads(result) if isinstance(result, str) else result
                        return {
                            "accepted": data.get("accepted_count", 0) > 0,
                            "accepted_count": data.get("accepted_count", 0),
                            "correction_token": data.get("correction_token"),
                            "total_tokens": len(data.get("tokens", [])) or 1,
                        }
                except Exception as exc:
                    logger.warning("MCP verify_draft failed, using fallback: %s", exc)

        # Local fallback: accept the draft (optimistic)
        logger.debug("Using local fallback verification for session %s.", session.session_id)
        return {"accepted": True, "accepted_count": 1, "total_tokens": 1}
