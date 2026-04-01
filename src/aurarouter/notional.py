"""Notional response streaming and correction protocol.

Implements confidence-gated emission of draft tokens before verification
completes. On verifier rejection, emits CorrectionEvent to rewind the
client UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Notional")


@dataclass(frozen=True)
class NotionalResponse:
    """A non-final response emitted before verification completes.

    Marked ``status: notional`` so clients know this may be revised.
    """

    session_id: str
    content: str
    drafter_model: str
    status: str = "notional"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "content": self.content,
            "drafter_model": self.drafter_model,
            "status": self.status,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CorrectionEvent:
    """Emitted when the verifier rejects draft tokens.

    Contains the position of the first rejected token and the
    verifier's correction tokens (ground truth).
    """

    session_id: str
    correction_position: int
    correction_tokens: list[int] = field(default_factory=list)
    reason: str = "verifier_rejection"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "correction_position": self.correction_position,
            "correction_tokens": self.correction_tokens,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class NotionalEmitter:
    """Manages confidence-gated notional response emission.

    Only emits draft tokens to the client when the triage confidence
    score exceeds the configured threshold (default 0.85).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        on_notional: Callable[[NotionalResponse], None] | None = None,
        on_correction: Callable[[CorrectionEvent], None] | None = None,
    ):
        self._confidence_threshold = confidence_threshold
        self._on_notional = on_notional
        self._on_correction = on_correction
        self._emitted: list[NotionalResponse] = []
        self._corrections: list[CorrectionEvent] = []

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def emitted_count(self) -> int:
        return len(self._emitted)

    @property
    def correction_count(self) -> int:
        return len(self._corrections)

    def should_emit(self, confidence: float) -> bool:
        """Return True if confidence exceeds the threshold."""
        return confidence >= self._confidence_threshold

    def emit_notional(
        self,
        session_id: str,
        content: str,
        drafter_model: str,
        confidence: float,
    ) -> NotionalResponse | None:
        """Emit a notional response if confidence is sufficient.

        Returns the NotionalResponse if emitted, None if gated.
        """
        if not self.should_emit(confidence):
            logger.debug(
                "Notional emission gated: confidence %.3f < threshold %.3f",
                confidence, self._confidence_threshold,
            )
            return None

        response = NotionalResponse(
            session_id=session_id,
            content=content,
            drafter_model=drafter_model,
        )
        self._emitted.append(response)

        if self._on_notional is not None:
            self._on_notional(response)

        logger.info(
            "Notional response emitted for session %s (confidence=%.3f).",
            session_id, confidence,
        )
        return response

    def emit_correction(
        self,
        session_id: str,
        correction_position: int,
        correction_tokens: list[int] | None = None,
        reason: str = "verifier_rejection",
    ) -> CorrectionEvent:
        """Emit a correction event for a rejected draft."""
        event = CorrectionEvent(
            session_id=session_id,
            correction_position=correction_position,
            correction_tokens=correction_tokens or [],
            reason=reason,
        )
        self._corrections.append(event)

        if self._on_correction is not None:
            self._on_correction(event)

        logger.info(
            "Correction event emitted for session %s at position %d.",
            session_id, correction_position,
        )
        return event
