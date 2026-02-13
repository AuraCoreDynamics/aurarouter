"""Shared formatting utilities for the AuraRouter GUI."""

from __future__ import annotations


def format_tokens(n: int) -> str:
    """Format a token count for display.

    * < 1_000_000  → comma-separated (``1,234``)
    * >= 1_000_000 → millions with two decimals (``1.50M``)
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n:,}"


def format_cost(amount: float) -> str:
    """Format a dollar amount (``$0.04``, ``$12.50``)."""
    return f"${amount:,.2f}"


def format_duration(seconds: float) -> str:
    """Format elapsed time.

    * < 60 → ``0.5s``
    * >= 60 → ``1m 5s``
    """
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds != int(seconds) else f"{int(seconds)}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs}s"
