"""Generic Cell Rate Algorithm (GCRA) implementation with asymmetric-burst resilience.

AuraRouter v2.0 §3.3
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from aurarouter.ratelimit.models import RateState


@dataclass
class TatCell:
    """A single GCRA tracking cell representing one rate-limit dimension (RPM, ITPM, or OTPM).

    State is minimal: Theoretical Arrival Time (_tat), pending in-flight load,
    explicit cooldown expiration, upstream reset horizon, and timestamp of the
    last authoritative upstream sync.
    """
    _tat: float = 0.0
    _inflight_cost: int = 0
    _cooldown_until: float = 0.0
    _reset_at: float = 0.0
    _last_sync_ts: float = 0.0
    _is_daily_cooldown: bool = False

    def _effective_tat_and_reset(self, curr: float) -> tuple[float, float]:
        """Pure non-mutating view of (_tat, _reset_at) safe for lock-free hot-path reads."""
        tat = self._tat
        reset_at = self._reset_at
        if reset_at > 0.0 and curr >= reset_at:
            return min(tat, curr), 0.0
        return tat, reset_at

    def _expire_reset_horizon(self, curr: float) -> None:
        """Mutating horizon cleanup invoked ONLY on locked write paths."""
        if self._reset_at > 0.0 and curr >= self._reset_at:
            self._tat = min(self._tat, curr)
            self._reset_at = 0.0

    def is_in_cooldown(self, t_now: float | None = None) -> bool:
        curr = t_now if t_now is not None else time.monotonic()
        return curr < self._cooldown_until

    def seconds_until_reset(self, t_now: float | None = None) -> float:
        curr = t_now if t_now is not None else time.monotonic()
        eff_tat, eff_reset = self._effective_tat_and_reset(curr)
        if curr < self._cooldown_until:
            return max(0.0, self._cooldown_until - curr)
        if eff_reset > curr:
            return max(0.0, eff_reset - curr)
        return max(0.0, eff_tat - curr)

    def is_authoritative_regime(self, t_now: float, header_sync_ttl_s: float = 60.0) -> bool:
        """True if we have received an authoritative upstream sync within the TTL window."""
        last_sync = self._last_sync_ts
        if last_sync <= 0.0:
            return False
        return (t_now - last_sync) <= header_sync_ttl_s

    def compute_tau(
        self,
        quota: int,
        replica_count: int = 1,
        skew_damping: float = 0.25,
        window: float = 60.0,
        is_authoritative: bool = False,
    ) -> float:
        """Compute emission interval tau.

        Under the authoritative header regime, we use unscaled tau_global (W / Q)
        so a single node receiving an asymmetric burst can utilize up to 100% of global headroom.
        Under the headerless/cold-start regime, we use tau_elastic with skew damping.
        """
        if quota <= 0:
            return 0.0
        tau_global = window / float(quota)
        if is_authoritative or replica_count <= 1:
            return tau_global

        # Damped elastic emission interval for headerless multi-replica setup
        clamped_kappa = max(0.0, min(1.0, skew_damping))
        return tau_global * (1.0 + clamped_kappa * float(replica_count - 1))

    def reserve_with_tau(
        self,
        cost: int,
        quota: int,
        replica_count: int = 1,
        skew_damping: float = 0.25,
        header_sync_ttl_s: float = 60.0,
        window: float = 60.0,
        t_now: float | None = None,
    ) -> tuple[bool, float]:
        """Reserve capacity and return ``(admitted, tau_used)``."""
        curr = t_now if t_now is not None else time.monotonic()
        self._expire_reset_horizon(curr)
        if self.is_in_cooldown(curr):
            return False, 0.0

        if quota <= 0 or cost <= 0:
            if cost > 0:
                self._inflight_cost += cost
            return True, 0.0

        is_auth = self.is_authoritative_regime(curr, header_sync_ttl_s)
        tau = self.compute_tau(
            quota=quota,
            replica_count=replica_count,
            skew_damping=skew_damping,
            window=window,
            is_authoritative=is_auth,
        )

        tat_next = max(curr, self._tat) + (cost * tau)
        # Only hard-reject pre-flight if authoritative headers confirmed exhaustion
        if (tat_next - curr) > (window * 1.05) and self._tat >= (curr + window):
            if is_auth:
                return False, 0.0

        self._inflight_cost += cost
        self._tat = tat_next
        return True, tau

    def reserve(
        self,
        cost: int,
        quota: int,
        replica_count: int = 1,
        skew_damping: float = 0.25,
        header_sync_ttl_s: float = 60.0,
        window: float = 60.0,
        t_now: float | None = None,
    ) -> bool:
        """Attempt to reserve capacity for an in-flight request.

        Returns True if admitted, False if hard exhausted (e.g. active cooldown).
        """
        admitted, _ = self.reserve_with_tau(
            cost=cost,
            quota=quota,
            replica_count=replica_count,
            skew_damping=skew_damping,
            header_sync_ttl_s=header_sync_ttl_s,
            window=window,
            t_now=t_now,
        )
        return admitted

    def settle(
        self,
        actual_cost: int,
        reserved_cost: int,
        quota: int,
        limit: int | None = None,
        remaining: int | None = None,
        reset_s: float | None = None,
        window: float = 60.0,
        t_now: float | None = None,
        *,
        admitted: bool = True,
        tau_used: float | None = None,
        lease_created_at: float | None = None,
    ) -> None:
        """Reconcile reservation upon receiving response."""
        curr = t_now if t_now is not None else time.monotonic()
        self._expire_reset_horizon(curr)
        eff_reserved = reserved_cost if admitted else 0
        self._inflight_cost = max(0, self._inflight_cost - eff_reserved)

        effective_limit = limit if (limit is not None and limit > 0) else quota
        if effective_limit > 0 and remaining is not None:
            # Guard against out-of-order / delayed header time-travel:
            # If a newer request or 429 already synced after this lease was created,
            # ignore the older response's stale remaining header.
            if (
                lease_created_at is not None
                and self._last_sync_ts > 0.0
                and lease_created_at < self._last_sync_ts
            ):
                return

            # Authoritative upstream header sync!
            u_upstream = max(0.0, min(1.0, 1.0 - (float(remaining) / float(effective_limit))))
            tau_global = window / float(effective_limit)

            # Map true upstream utilization into GCRA window + local still-in-flight leases
            self._tat = curr + (u_upstream * window) + (self._inflight_cost * tau_global)
            self._last_sync_ts = curr

            if reset_s is not None and reset_s > 0.0:
                self._reset_at = curr + reset_s
            else:
                self._reset_at = 0.0

            if remaining <= 0 and reset_s is not None and reset_s > 0.0:
                # Zero units left until reset_s elapses -> exact hard cooldown
                self._cooldown_until = max(self._cooldown_until, curr + reset_s)
                if self._cooldown_until <= curr:
                    self._is_daily_cooldown = False
            elif self._cooldown_until <= curr:
                # Never wipe an active 429 cooldown from a sibling request!
                self._cooldown_until = 0.0
                self._is_daily_cooldown = False
        else:
            # Fallback local adjustment using exact tau_used from reserve() so tau_elastic never leaks
            if effective_limit > 0:
                tau_base = window / float(effective_limit)
                eff_tau = tau_used if (tau_used is not None and tau_used > 0.0) else tau_base
                if admitted:
                    delta = (actual_cost * eff_tau) - (eff_reserved * eff_tau)
                else:
                    delta = actual_cost * tau_base
                self._tat = max(curr, self._tat + delta)

    def settle_error(
        self,
        reserved_cost: int,
        retry_after_s: float | None = None,
        is_daily: bool = False,
        t_now: float | None = None,
        *,
        admitted: bool = True,
    ) -> None:
        """Handle 429 or explicit rate-limit backoff."""
        curr = t_now if t_now is not None else time.monotonic()
        if admitted and reserved_cost > 0:
            self._inflight_cost = max(0, self._inflight_cost - reserved_cost)

        backoff = retry_after_s if (retry_after_s is not None and retry_after_s > 0) else 5.0
        was_active_daily = curr < self._cooldown_until and self._is_daily_cooldown
        self._cooldown_until = max(self._cooldown_until, curr + backoff)
        self._is_daily_cooldown = bool(is_daily or was_active_daily)
        self._tat = max(self._tat, curr + backoff)
        self._last_sync_ts = curr

    def abort(
        self,
        reserved_cost: int,
        quota: int,
        window: float = 60.0,
        t_now: float | None = None,
        *,
        admitted: bool = True,
        tau_used: float | None = None,
    ) -> None:
        """Cancel an in-flight lease when request failed before execution or timed out."""
        if not admitted or reserved_cost <= 0:
            return
        curr = t_now if t_now is not None else time.monotonic()
        self._inflight_cost = max(0, self._inflight_cost - reserved_cost)
        eff_tau = (
            tau_used
            if (tau_used is not None and tau_used > 0.0)
            else (window / float(quota) if quota > 0 else 0.0)
        )
        if eff_tau > 0.0:
            self._tat = max(curr, self._tat - (reserved_cost * eff_tau))

    def get_utilization(self, t_now: float | None = None, window: float = 60.0) -> float:
        """Return projected utilization [0.0..1.0] without mutating cell state."""
        curr = t_now if t_now is not None else time.monotonic()
        eff_tat, _ = self._effective_tat_and_reset(curr)
        if self.is_in_cooldown(curr):
            return 1.0
        raw_u = (eff_tat - curr) / window
        return max(0.0, min(1.0, raw_u))

    def get_state(
        self,
        warning_threshold: float = 0.80,
        header_sync_ttl_s: float = 60.0,
        window: float = 60.0,
        t_now: float | None = None,
    ) -> RateState:
        """Derive RateState purely, respecting the No-False-Exhaustion invariant."""
        curr = t_now if t_now is not None else time.monotonic()
        if self.is_in_cooldown(curr):
            return RateState.DAILY_EXHAUSTED if self._is_daily_cooldown else RateState.EXHAUSTED

        u = self.get_utilization(curr, window=window)
        if u >= 1.0:
            if not self.is_authoritative_regime(curr, header_sync_ttl_s=header_sync_ttl_s):
                return RateState.NEARING_SATURATION
            return RateState.EXHAUSTED

        if u >= warning_threshold:
            return RateState.NEARING_SATURATION

        return RateState.NORMAL
