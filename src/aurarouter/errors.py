"""Unified AuraError schema — wire-format error model shared across all AuraCore projects.

Error codes:
  1xxx = AuraRouter  2xxx = AuraXLM  3xxx = AuraGrid  4xxx = AuraCode

This file is intentionally duplicated across Python projects (no shared package) to
preserve release-cycle independence. Keep in sync via TG8 serialization fixture tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AuraError(BaseModel):
    """Canonical error schema for wire-boundary error responses."""

    model_config = ConfigDict(frozen=True)

    error_code: int
    category: Literal[
        "auth", "resource", "routing", "governance",
        "infrastructure", "validation", "internal",
    ]
    message: str
    detail: str | None = None
    source_project: str | None = None
    timestamp: datetime | None = None

    @classmethod
    def from_exception(
        cls,
        ex: Exception,
        error_code: int,
        category: str,
        source_project: str | None = None,
    ) -> "AuraError":
        return cls(
            error_code=error_code,
            category=category,  # type: ignore[arg-type]
            message=str(ex),
            detail=type(ex).__name__,
            source_project=source_project,
            timestamp=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Typed Provider Exceptions (AuraRouter v2.0 §3.4 / v2.1 §3)
# ---------------------------------------------------------------------------

class ProviderException(Exception):
    """Base exception for provider generation failures."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        model_id: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model_id = model_id


class RateLimitExceededError(ProviderException):
    """Raised when a provider rejects a request due to rate limit saturation (HTTP 429)."""

    def __init__(
        self,
        message: str,
        snapshot: Any = None,
        *,
        provider: str = "",
        model_id: str = "",
        retry_after_s: float | None = None,
    ):
        super().__init__(message, provider=provider, model_id=model_id)
        if snapshot is None and retry_after_s is not None:
            try:
                from aurarouter.ratelimit.models import RateLimitSnapshot
                snapshot = RateLimitSnapshot(retry_after_s=retry_after_s)
            except Exception:
                snapshot = None
        self.snapshot = snapshot
        self.retry_after_s = (
            retry_after_s
            if retry_after_s is not None
            else getattr(snapshot, "retry_after_s", None)
        )


class ContextLengthExceededError(ProviderException):
    """Raised when prompt + generation exceeds target model context window limit."""

    def __init__(
        self,
        message: str,
        max_tokens: int | None = None,
        *,
        provider: str = "",
        model_id: str = "",
    ):
        super().__init__(message, provider=provider, model_id=model_id)
        self.max_tokens = max_tokens


class ProviderTransientError(ProviderException):
    """Raised when provider encounters a transient infrastructure failure (5xx/timeouts)."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        provider: str = "",
        model_id: str = "",
    ):
        super().__init__(message, provider=provider, model_id=model_id)
        self.status_code = status_code
