"""Data models for rate limiting, capacity tracking, and token leases.

AuraRouter v2.0 §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid
import time


class RateState(str, Enum):
    """Operational rate-limit saturation state for a candidate model."""
    NORMAL = "normal"                  # < warning_threshold (e.g. < 80%)
    NEARING_SATURATION = "nearing"     # >= warning_threshold and < 100%
    EXHAUSTED = "exhausted"            # >= 100% or active 429 backoff
    DAILY_EXHAUSTED = "daily_exhausted"  # RPD / daily quota depleted (long cooldown)


@dataclass(frozen=True)
class RateLimitSnapshot:
    """Normalized provider capacity & rate-limit telemetry."""
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpm_reset_s: float | None = None          # Monotonic delta seconds until reset

    itpm_limit: int | None = None             # Input TPM (or unified TPM if unsplit)
    itpm_remaining: int | None = None
    itpm_reset_s: float | None = None

    otpm_limit: int | None = None             # Output TPM (Anthropic splits ITPM/OTPM)
    otpm_remaining: int | None = None
    otpm_reset_s: float | None = None

    retry_after_s: float | None = None        # Explicit backoff directive
    is_daily_quota: bool = False              # True when QuotaFailure scopes to PerDay

    # Local GPU / Inference Server saturation signals (0.0..1.0)
    slot_utilization: float | None = None     # Active slots / total slots
    kv_cache_utilization: float | None = None # KV cache blocks used / total blocks
    requests_waiting: int | None = None       # Pending queue depth

    def to_dict(self) -> dict:
        """Serialize non-None fields for wire / MCP transport."""
        out: dict = {}
        for k in self.__dataclass_fields__:
            val = getattr(self, k)
            if val is not None and (k != "is_daily_quota" or val is True):
                out[k] = val
        return out

    @classmethod
    def from_dict(cls, data: dict | None) -> "RateLimitSnapshot | None":
        """Deserialize from wire / MCP dict."""
        if not data or not isinstance(data, dict):
            return None
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass(frozen=True)
class RateLimitQuota:
    """Static rate limit quotas and threshold configuration for a provider or model."""
    rpm: int = 0
    itpm: int = 0
    otpm: int = 0
    warning_threshold: float = 0.80
    header_sync_ttl_seconds: float = 60.0
    replica_skew_damping: float = 0.25


@dataclass
class TokenLease:
    """Active pre-flight capacity lease guarding an in-flight model attempt."""
    model_id: str
    provider: str
    reserved_requests: int = 1
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    rpm_admitted: bool = True
    itpm_admitted: bool = False
    otpm_admitted: bool = False
    rpm_tau: float = 0.0
    itpm_tau: float = 0.0
    otpm_tau: float = 0.0
    lease_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.monotonic)
    settled: bool = False

    @property
    def admitted(self) -> bool:
        return self.rpm_admitted

    @property
    def total_reserved_tokens(self) -> int:
        return self.reserved_input_tokens + self.reserved_output_tokens


@dataclass(frozen=True)
class RateUtilization:
    """Live snapshot of utilization, saturation state, and headroom."""
    model_id: str
    provider: str
    rpm_utilization: float
    itpm_utilization: float
    otpm_utilization: float
    composite_utilization: float
    state: RateState
    seconds_until_reset: float
    headroom_score: float
