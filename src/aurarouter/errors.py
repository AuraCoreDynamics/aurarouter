"""Unified AuraError schema — wire-format error model shared across all AuraCore projects.

Error codes:
  1xxx = AuraRouter  2xxx = AuraXLM  3xxx = AuraGrid  4xxx = AuraCode

This file is intentionally duplicated across Python projects (no shared package) to
preserve release-cycle independence. Keep in sync via TG8 serialization fixture tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

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
