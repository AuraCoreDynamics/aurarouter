"""Rate limiting, GCRA lease management, and headroom tracking subsystem.

AuraRouter v2.0 §3
"""
from __future__ import annotations

from aurarouter.ratelimit.gcra import TatCell
from aurarouter.ratelimit.models import (
    RateLimitQuota,
    RateLimitSnapshot,
    RateState,
    RateUtilization,
    TokenLease,
)
from aurarouter.ratelimit.parsers import (
    format_ietf_ratelimit_headers,
    parse_duration_string,
    parse_ietf_ratelimit_header,
    parse_rate_limit_headers,
    parse_retry_after,
    parse_rfc3339_reset,
)
from aurarouter.ratelimit.tracker import RateLimitTracker

__all__ = [
    "RateState",
    "RateLimitSnapshot",
    "RateLimitQuota",
    "TokenLease",
    "RateUtilization",
    "TatCell",
    "RateLimitTracker",
    "parse_rate_limit_headers",
    "parse_duration_string",
    "parse_rfc3339_reset",
    "parse_retry_after",
    "parse_ietf_ratelimit_header",
    "format_ietf_ratelimit_headers",
]

