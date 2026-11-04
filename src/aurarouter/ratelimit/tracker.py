"""Central rate limit, capacity lease, and headroom tracking coordinator.

AuraRouter v2.0 §3.1, §3.3
"""
from __future__ import annotations

import threading
import time
from typing import Any

from aurarouter._logging import get_logger
from aurarouter.ratelimit.gcra import TatCell
from aurarouter.ratelimit.models import (
    RateLimitQuota,
    RateLimitSnapshot,
    RateState,
    RateUtilization,
    TokenLease,
)
from aurarouter.ratelimit.parsers import parse_rate_limit_headers

logger = get_logger("AuraRouter.RateLimit")


class _ModelRateLimiter:
    """Internal container managing GCRA cells and metrics for a specific provider or model."""

    def __init__(self, key: str, quota: RateLimitQuota):
        self.key = key
        self.quota = quota
        self.rpm_cell = TatCell()
        self.itpm_cell = TatCell()
        self.otpm_cell = TatCell()
        self.seen_models: set[str] = set()

        # Vegas / Gradient2 TTFT estimator state
        self.baseline_ttft_s: float | None = None
        self.short_ewma_ttft_s: float | None = None

        # Local inference / GPU slot telemetry
        self.slot_utilization: float | None = None
        self.kv_cache_utilization: float | None = None
        self.requests_waiting: int | None = None

    def record_ttft(self, ttft_s: float) -> None:
        if ttft_s <= 0.0:
            return
        if self.baseline_ttft_s is None:
            self.baseline_ttft_s = ttft_s
            self.short_ewma_ttft_s = ttft_s
            return

        # Baseline adapts slowly (alpha = 0.05) to track minimum latency floor
        if ttft_s < self.baseline_ttft_s:
            self.baseline_ttft_s = ttft_s
        else:
            self.baseline_ttft_s = 0.95 * self.baseline_ttft_s + 0.05 * ttft_s

        # Short-term EWMA reacts quickly (alpha = 0.30) to queue spikes
        if self.short_ewma_ttft_s is None:
            self.short_ewma_ttft_s = ttft_s
        else:
            self.short_ewma_ttft_s = 0.70 * self.short_ewma_ttft_s + 0.30 * ttft_s

    def compute_vegas_utilization(self) -> float:
        """Derive virtual utilization from TTFT gradient."""
        if not self.baseline_ttft_s or not self.short_ewma_ttft_s:
            return 0.0
        if self.short_ewma_ttft_s <= 0.0:
            return 0.0

        gradient = self.baseline_ttft_s / self.short_ewma_ttft_s
        if gradient >= 0.85:
            return 0.0
        if gradient < 0.65:
            # Significant queuing delay detected; synthesize saturation backpressure
            scale = (0.65 - gradient) / 0.65
            return min(1.0, 0.80 + (scale * 0.20))
        return 0.0


class RateLimitTracker:
    """Thread-safe rate limit and headroom manager.

    Coordinates pre-flight token leases, multi-protocol header sync,
    asymmetric-safe GCRA execution, and headroom scoring.
    """

    def __init__(
        self,
        replica_count: int = 1,
        default_warning_threshold: float = 0.80,
        enable_vegas: bool = True,
    ):
        self._replica_count = max(1, replica_count)
        self._default_warning_threshold = default_warning_threshold
        self._enable_vegas = enable_vegas

        self._lock = threading.Lock()
        self._limiters: dict[str, _ModelRateLimiter] = {}
        self._leases: dict[str, TokenLease] = {}
        self._quotas: dict[str, RateLimitQuota] = {}

    def _get_key(self, model_id: str, provider: str) -> str:
        """Resolve limiter key without collapsing distinct unconfigured models into one bucket."""
        prov_l = (provider or "unknown").lower()
        if not model_id or model_id == "*":
            return prov_l

        model_key = f"{prov_l}::{model_id.lower()}"
        if model_key in self._quotas or model_key in self._limiters:
            return model_key
        # Only share a vendor-wide limiter across models when an explicit provider quota exists
        if prov_l in self._quotas:
            return prov_l
        return model_key

    def configure_quota(
        self,
        provider: str,
        quota: RateLimitQuota,
        model_id: str | None = None,
    ) -> None:
        """Register or update static quotas for a provider or model."""
        key = f"{provider}::{model_id}".lower() if model_id else provider.lower()
        with self._lock:
            self._quotas[key] = quota
            if key in self._limiters:
                self._limiters[key].quota = quota
            else:
                limiter = _ModelRateLimiter(key, quota)
                if model_id:
                    limiter.seen_models.add(model_id)
                new_map = dict(self._limiters)
                new_map[key] = limiter
                self._limiters = new_map

    def _get_or_create_limiter(self, model_id: str, provider: str) -> _ModelRateLimiter:
        key = self._get_key(model_id, provider)
        limiter = self._limiters.get(key)
        if limiter is None:
            quota = self._quotas.get(
                key,
                self._quotas.get(
                    (provider or "unknown").lower(),
                    RateLimitQuota(warning_threshold=self._default_warning_threshold),
                ),
            )
            limiter = _ModelRateLimiter(key, quota)
            new_map = dict(self._limiters)
            new_map[key] = limiter
            self._limiters = new_map
        if model_id and model_id != "*":
            limiter.seen_models.add(model_id)
        return limiter

    # ------------------------------------------------------------------
    # Lease Lifecycle
    # ------------------------------------------------------------------

    def reserve(
        self,
        model_id: str,
        provider: str,
        est_input_tokens: int = 0,
        est_output_tokens: int = 0,
    ) -> TokenLease:
        """Obtain a pre-flight TokenLease guarding in-flight dispatch."""
        with self._lock:
            limiter = self._get_or_create_limiter(model_id, provider)
            t_now = time.monotonic()
            quota = limiter.quota

            rpm_admitted, rpm_tau = limiter.rpm_cell.reserve_with_tau(
                cost=1,
                quota=quota.rpm,
                replica_count=self._replica_count,
                skew_damping=quota.replica_skew_damping,
                header_sync_ttl_s=quota.header_sync_ttl_seconds,
                t_now=t_now,
            )

            itpm_admitted = False
            itpm_tau = 0.0
            if est_input_tokens > 0:
                itpm_admitted, itpm_tau = limiter.itpm_cell.reserve_with_tau(
                    cost=est_input_tokens,
                    quota=quota.itpm,
                    replica_count=self._replica_count,
                    skew_damping=quota.replica_skew_damping,
                    header_sync_ttl_s=quota.header_sync_ttl_seconds,
                    t_now=t_now,
                )

            otpm_admitted = False
            otpm_tau = 0.0
            if est_output_tokens > 0 and quota.otpm > 0:
                otpm_admitted, otpm_tau = limiter.otpm_cell.reserve_with_tau(
                    cost=est_output_tokens,
                    quota=quota.otpm,
                    replica_count=self._replica_count,
                    skew_damping=quota.replica_skew_damping,
                    header_sync_ttl_s=quota.header_sync_ttl_seconds,
                    t_now=t_now,
                )

            lease = TokenLease(
                model_id=model_id,
                provider=provider,
                reserved_requests=1,
                reserved_input_tokens=est_input_tokens,
                reserved_output_tokens=est_output_tokens,
                rpm_admitted=rpm_admitted,
                itpm_admitted=itpm_admitted,
                otpm_admitted=otpm_admitted,
                rpm_tau=rpm_tau,
                itpm_tau=itpm_tau,
                otpm_tau=otpm_tau,
                created_at=t_now,
            )
            self._leases[lease.lease_id] = lease
            return lease

    def settle(
        self,
        lease: TokenLease,
        actual_input_tokens: int = 0,
        actual_output_tokens: int = 0,
        snapshot: RateLimitSnapshot | None = None,
        ttft_s: float | None = None,
    ) -> None:
        """Reconcile reservation upon receiving a successful 200 OK response."""
        with self._lock:
            if lease.settled:
                return
            lease.settled = True
            self._leases.pop(lease.lease_id, None)

            limiter = self._get_or_create_limiter(lease.model_id, lease.provider)
            t_now = time.monotonic()
            quota = limiter.quota

            # Reconcile RPM
            rpm_lim = snapshot.rpm_limit if snapshot else None
            rpm_rem = snapshot.rpm_remaining if snapshot else None
            rpm_reset = snapshot.rpm_reset_s if snapshot else None
            limiter.rpm_cell.settle(
                actual_cost=1,
                reserved_cost=lease.reserved_requests,
                quota=quota.rpm,
                limit=rpm_lim,
                remaining=rpm_rem,
                reset_s=rpm_reset,
                t_now=t_now,
                admitted=lease.rpm_admitted,
                tau_used=lease.rpm_tau,
                lease_created_at=lease.created_at,
            )

            # Reconcile ITPM
            itpm_lim = snapshot.itpm_limit if snapshot else None
            itpm_rem = snapshot.itpm_remaining if snapshot else None
            itpm_reset = snapshot.itpm_reset_s if snapshot else None
            limiter.itpm_cell.settle(
                actual_cost=actual_input_tokens,
                reserved_cost=lease.reserved_input_tokens,
                quota=quota.itpm,
                limit=itpm_lim,
                remaining=itpm_rem,
                reset_s=itpm_reset,
                t_now=t_now,
                admitted=lease.itpm_admitted,
                tau_used=lease.itpm_tau,
                lease_created_at=lease.created_at,
            )

            # Reconcile OTPM
            otpm_lim = snapshot.otpm_limit if snapshot else None
            otpm_rem = snapshot.otpm_remaining if snapshot else None
            otpm_reset = snapshot.otpm_reset_s if snapshot else None
            limiter.otpm_cell.settle(
                actual_cost=actual_output_tokens,
                reserved_cost=lease.reserved_output_tokens,
                quota=quota.otpm,
                limit=otpm_lim,
                remaining=otpm_rem,
                reset_s=otpm_reset,
                t_now=t_now,
                admitted=lease.otpm_admitted,
                tau_used=lease.otpm_tau,
                lease_created_at=lease.created_at,
            )

            # Sync local GPU slot metrics if present
            if snapshot:
                if snapshot.slot_utilization is not None:
                    limiter.slot_utilization = snapshot.slot_utilization
                if snapshot.kv_cache_utilization is not None:
                    limiter.kv_cache_utilization = snapshot.kv_cache_utilization
                if snapshot.requests_waiting is not None:
                    limiter.requests_waiting = snapshot.requests_waiting

            # Record TTFT for Vegas estimator
            if self._enable_vegas and ttft_s is not None:
                limiter.record_ttft(ttft_s)

    def settle_rate_limited(
        self,
        lease: TokenLease,
        snapshot: RateLimitSnapshot | None = None,
    ) -> None:
        """Trigger backoff cooldown upon receiving an HTTP 429 without tripping CircuitBreaker."""
        with self._lock:
            if lease.settled:
                return
            lease.settled = True
            self._leases.pop(lease.lease_id, None)

            limiter = self._get_or_create_limiter(lease.model_id, lease.provider)
            t_now = time.monotonic()

            retry_after = snapshot.retry_after_s if snapshot else None
            is_daily = snapshot.is_daily_quota if snapshot else False

            limiter.rpm_cell.settle_error(
                reserved_cost=lease.reserved_requests,
                retry_after_s=retry_after,
                is_daily=is_daily,
                t_now=t_now,
                admitted=lease.rpm_admitted,
            )
            limiter.itpm_cell.settle_error(
                reserved_cost=lease.reserved_input_tokens,
                retry_after_s=retry_after,
                is_daily=is_daily,
                t_now=t_now,
                admitted=lease.itpm_admitted,
            )
            limiter.otpm_cell.settle_error(
                reserved_cost=lease.reserved_output_tokens,
                retry_after_s=retry_after,
                is_daily=is_daily,
                t_now=t_now,
                admitted=lease.otpm_admitted,
            )

    def abort(self, lease: TokenLease) -> None:
        """Cancel an in-flight lease when the attempt fails before or during dispatch."""
        with self._lock:
            if lease.settled:
                return
            lease.settled = True
            self._leases.pop(lease.lease_id, None)

            limiter = self._get_or_create_limiter(lease.model_id, lease.provider)
            t_now = time.monotonic()
            quota = limiter.quota

            limiter.rpm_cell.abort(
                lease.reserved_requests,
                quota.rpm,
                t_now=t_now,
                admitted=lease.rpm_admitted,
                tau_used=lease.rpm_tau,
            )
            limiter.itpm_cell.abort(
                lease.reserved_input_tokens,
                quota.itpm,
                t_now=t_now,
                admitted=lease.itpm_admitted,
                tau_used=lease.itpm_tau,
            )
            limiter.otpm_cell.abort(
                lease.reserved_output_tokens,
                quota.otpm,
                t_now=t_now,
                admitted=lease.otpm_admitted,
                tau_used=lease.otpm_tau,
            )

    # ------------------------------------------------------------------
    # Telemetry and Header Ingestion
    # ------------------------------------------------------------------

    def sync_from_headers(
        self,
        provider: str,
        model_id: str,
        headers: dict[str, Any],
        now_utc: Any | None = None,
    ) -> RateLimitSnapshot:
        """Parse HTTP response headers and synchronize limiter state."""
        snapshot = parse_rate_limit_headers(headers, provider=provider, now_utc=now_utc)
        with self._lock:
            limiter = self._get_or_create_limiter(model_id, provider)
            t_now = time.monotonic()
            quota = limiter.quota

            if snapshot.rpm_limit is not None or snapshot.rpm_remaining is not None:
                limiter.rpm_cell.settle(
                    actual_cost=0,
                    reserved_cost=0,
                    quota=quota.rpm,
                    limit=snapshot.rpm_limit,
                    remaining=snapshot.rpm_remaining,
                    reset_s=snapshot.rpm_reset_s,
                    t_now=t_now,
                    admitted=False,
                )
            if snapshot.itpm_limit is not None or snapshot.itpm_remaining is not None:
                limiter.itpm_cell.settle(
                    actual_cost=0,
                    reserved_cost=0,
                    quota=quota.itpm,
                    limit=snapshot.itpm_limit,
                    remaining=snapshot.itpm_remaining,
                    reset_s=snapshot.itpm_reset_s,
                    t_now=t_now,
                    admitted=False,
                )
            if snapshot.otpm_limit is not None or snapshot.otpm_remaining is not None:
                limiter.otpm_cell.settle(
                    actual_cost=0,
                    reserved_cost=0,
                    quota=quota.otpm,
                    limit=snapshot.otpm_limit,
                    remaining=snapshot.otpm_remaining,
                    reset_s=snapshot.otpm_reset_s,
                    t_now=t_now,
                    admitted=False,
                )

            if snapshot.retry_after_s:
                limiter.rpm_cell.settle_error(0, snapshot.retry_after_s, snapshot.is_daily_quota, t_now, admitted=False)
                limiter.itpm_cell.settle_error(0, snapshot.retry_after_s, snapshot.is_daily_quota, t_now, admitted=False)
                limiter.otpm_cell.settle_error(0, snapshot.retry_after_s, snapshot.is_daily_quota, t_now, admitted=False)

        return snapshot

    def record_slot_telemetry(
        self,
        model_id: str,
        provider: str,
        slot_utilization: float | None = None,
        kv_cache_utilization: float | None = None,
        requests_waiting: int | None = None,
    ) -> None:
        """Record passive local GPU or inference server slot telemetry."""
        with self._lock:
            limiter = self._get_or_create_limiter(model_id, provider)
            if slot_utilization is not None:
                limiter.slot_utilization = slot_utilization
            if kv_cache_utilization is not None:
                limiter.kv_cache_utilization = kv_cache_utilization
            if requests_waiting is not None:
                limiter.requests_waiting = requests_waiting

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def _evaluate_limiter(
        self, limiter: _ModelRateLimiter, model_id: str, provider: str, t_now: float
    ) -> RateUtilization:
        quota = limiter.quota
        u_rpm = limiter.rpm_cell.get_utilization(t_now)
        u_itpm = limiter.itpm_cell.get_utilization(t_now)
        u_otpm = limiter.otpm_cell.get_utilization(t_now)
        u_slots = limiter.slot_utilization or 0.0
        u_kv = limiter.kv_cache_utilization or 0.0
        u_vegas = limiter.compute_vegas_utilization() if self._enable_vegas else 0.0

        u_composite = max(u_rpm, u_itpm, u_otpm, u_slots, u_kv, u_vegas)

        s_rpm = limiter.rpm_cell.get_state(
            quota.warning_threshold, quota.header_sync_ttl_seconds, t_now=t_now
        )
        s_itpm = limiter.itpm_cell.get_state(
            quota.warning_threshold, quota.header_sync_ttl_seconds, t_now=t_now
        )
        s_otpm = limiter.otpm_cell.get_state(
            quota.warning_threshold, quota.header_sync_ttl_seconds, t_now=t_now
        )

        if RateState.DAILY_EXHAUSTED in (s_rpm, s_itpm, s_otpm):
            state = RateState.DAILY_EXHAUSTED
        elif RateState.EXHAUSTED in (s_rpm, s_itpm, s_otpm) or u_composite >= 1.0:
            state = RateState.EXHAUSTED
        elif (
            RateState.NEARING_SATURATION in (s_rpm, s_itpm, s_otpm)
            or u_composite >= quota.warning_threshold
        ):
            state = RateState.NEARING_SATURATION
        else:
            state = RateState.NORMAL

        reset_s = max(
            limiter.rpm_cell.seconds_until_reset(t_now),
            limiter.itpm_cell.seconds_until_reset(t_now),
            limiter.otpm_cell.seconds_until_reset(t_now),
        )
        headroom = self._compute_headroom_score(u_composite, state, quota.warning_threshold)
        return RateUtilization(
            model_id=model_id,
            provider=provider,
            rpm_utilization=u_rpm,
            itpm_utilization=u_itpm,
            otpm_utilization=u_otpm,
            composite_utilization=u_composite,
            state=state,
            seconds_until_reset=reset_s,
            headroom_score=headroom,
        )

    def _peek_limiter(
        self,
        model_id: str,
        provider: str,
        fallback_provider: str = "",
    ) -> tuple[_ModelRateLimiter | None, str]:
        limiters = self._limiters
        if not limiters:
            return None, provider or fallback_provider
        prov_l = (provider or "").lower()
        mid_l = (model_id or "").lower()
        if prov_l:
            if mid_l:
                lim = limiters.get(f"{prov_l}::{mid_l}")
                if lim is not None:
                    return lim, provider
            lim = limiters.get(prov_l)
            if lim is not None:
                return lim, provider

        fb_l = (fallback_provider or "").lower()
        if fb_l and fb_l != prov_l:
            if mid_l:
                lim = limiters.get(f"{fb_l}::{mid_l}")
                if lim is not None:
                    return lim, fallback_provider
            lim = limiters.get(fb_l)
            if lim is not None:
                return lim, fallback_provider
        return None, provider or fallback_provider

    def peek_utilization(
        self,
        model_id: str,
        provider: str,
        fallback_provider: str = "",
    ) -> RateUtilization:
        """Pure lock-free read of utilization & headroom for hot-path RoutingPolicyStack."""
        limiter, eff_provider = self._peek_limiter(
            model_id, provider, fallback_provider=fallback_provider
        )
        if limiter is None:
            return RateUtilization(
                model_id=model_id,
                provider=eff_provider,
                rpm_utilization=0.0,
                itpm_utilization=0.0,
                otpm_utilization=0.0,
                composite_utilization=0.0,
                state=RateState.NORMAL,
                seconds_until_reset=0.0,
                headroom_score=1.0,
            )
        return self._evaluate_limiter(limiter, model_id, eff_provider, time.monotonic())

    def get_utilization(self, model_id: str, provider: str) -> RateUtilization:
        """Return the composite and dimensional utilization breakdown for a candidate model."""
        with self._lock:
            limiter = self._get_or_create_limiter(model_id, provider)
            return self._evaluate_limiter(limiter, model_id, provider, time.monotonic())

    def get_all_utilizations(self) -> dict[str, dict[str, Any]]:
        """Return a serializable status dictionary across all tracked limiters."""
        limiters = self._limiters
        t_now = time.monotonic()
        out: dict[str, dict[str, Any]] = {}
        for key, limiter in limiters.items():
            parts = key.split("::", 1)
            prov = parts[0]
            mid = parts[1] if len(parts) > 1 else "*"
            u = self._evaluate_limiter(limiter, mid, prov, t_now)
            out[key] = {
                "provider": prov,
                "model_id": mid,
                "models": sorted(limiter.seen_models),
                "state": u.state.value,
                "rpm_utilization": round(u.rpm_utilization, 4),
                "itpm_utilization": round(u.itpm_utilization, 4),
                "otpm_utilization": round(u.otpm_utilization, 4),
                "composite_utilization": round(u.composite_utilization, 4),
                "headroom_score": round(u.headroom_score, 4),
                "seconds_until_reset": round(u.seconds_until_reset, 2),
                "baseline_ttft_s": limiter.baseline_ttft_s,
                "short_ewma_ttft_s": limiter.short_ewma_ttft_s,
                "slot_utilization": limiter.slot_utilization,
                "kv_cache_utilization": limiter.kv_cache_utilization,
            }
        return out

    def get_state(self, model_id: str, provider: str) -> RateState:
        return self.get_utilization(model_id, provider).state

    def get_headroom_score(self, model_id: str, provider: str) -> float:
        return self.get_utilization(model_id, provider).headroom_score

    @staticmethod
    def _compute_headroom_score(u: float, state: RateState, warning_threshold: float) -> float:
        """Formulate smooth quadratic headroom decay factor (AuraRouter v2.0 §4.2)."""
        if state in (RateState.EXHAUSTED, RateState.DAILY_EXHAUSTED):
            return 0.0

        if u < warning_threshold:
            return 1.0

        # Smooth quadratic drop: Phi_headroom = 1.0 - 0.85 * ((u - theta) / (1.0 - theta))**2
        norm = (u - warning_threshold) / max(0.001, (1.0 - warning_threshold))
        score = 1.0 - 0.85 * (norm ** 2)
        return max(0.05, min(1.0, score))
