"""Data models for token usage tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerateResult:
    """Result from a provider generate call, including token usage.

    Backwards-compatible with plain ``str`` in string contexts via
    ``__str__`` — code that does ``f"{result}"`` or ``str(result)``
    gets the text content.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""
    provider: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass
class UsageRecord:
    """Single row in the usage ledger."""

    timestamp: str  # ISO 8601
    model_id: str
    provider: str
    role: str
    intent: str
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    success: bool
    is_cloud: bool
