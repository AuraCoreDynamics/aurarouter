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
    context_limit: int = 0
    gist: str | None = None

    def __str__(self) -> str:
        return self.text

    @property
    def usage(self) -> dict:
        """Return usage stats as a dict matching AURAROUTER_SPEC §3 GenerateResult."""
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "remaining": max(0, self.context_limit - self.input_tokens - self.output_tokens)
                         if self.context_limit > 0 else 0,
            "limit": self.context_limit,
        }


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
