"""Lock-free Copy-On-Write Beta posterior cache for Thompson Sampling.

AuraRouter v2.1 §4.2.1

Invariants:
- Hot-path ``get_posterior()`` performs ZERO SQLite queries and ZERO lock
  acquisitions (~50 ns dict reference lookup).
- Post-flight ``record_outcome()`` updates a staging map under ``_write_lock``
  and atomically swaps ``self._snapshot`` so concurrent readers never block.
- Startup hydration runs asynchronously off the request thread, applies
  exponential decay, and replays any live ``record_outcome()`` deltas recorded
  before or during hydration so write-through state is never clobbered.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
import time
from typing import TYPE_CHECKING, Any

from aurarouter._logging import get_logger
from aurarouter.routing_stack.context import complexity_to_band

if TYPE_CHECKING:
    from aurarouter.savings.feedback_store import FeedbackStore

logger = get_logger("AuraRouter.PosteriorCache")

PosteriorKey = tuple[str, str, int]  # (model_id, role_or_intent, complexity_band)
EmpiricalCounts = tuple[float, float]  # (decayed_successes, decayed_failures)
OutcomeDelta = tuple[str, str, int, bool, float]  # (model_id, role, band, success, gamma)


class PosteriorSnapshotCache:
    """Lock-free Copy-On-Write (COW) in-memory cache of Beta(alpha, beta) counts."""

    def __init__(self, decay_factor: float = 0.995, window_days: int = 14) -> None:
        self._decay_factor = max(0.5, min(1.0, decay_factor))
        self._window_days = window_days
        # Immutable dict swapped via atomic reference assignment on CPython
        self._snapshot: dict[PosteriorKey, EmpiricalCounts] = {}
        self._staging: dict[PosteriorKey, EmpiricalCounts] = {}
        self._write_lock = threading.Lock()
        self._last_hydrated_at: float | None = None
        self._hydration_count: int = 0
        # Live outcomes recorded before initial hydration completes + active query buffers
        self._pre_hydration_deltas: deque[OutcomeDelta] = deque(maxlen=4096)
        self._inflight_hydration_buffers: list[list[OutcomeDelta]] = []

    @property
    def size(self) -> int:
        return len(self._snapshot)

    @property
    def last_hydrated_at(self) -> float | None:
        return self._last_hydrated_at

    # ------------------------------------------------------------------
    # Lock-Free Hot-Path Read (~50 ns, zero locks, zero SQLite I/O)
    # ------------------------------------------------------------------

    def get_posterior(
        self,
        model_id: str,
        role_or_intent: str,
        complexity_band: int,
        default_prior: tuple[float, float] = (2.0, 1.0),
    ) -> tuple[float, float]:
        """Return ``(alpha, beta)`` combining ``default_prior`` with cached counts."""
        snap = self._snapshot
        counts = snap.get((model_id, role_or_intent, complexity_band))
        if counts is None:
            counts = snap.get((model_id, "*", complexity_band))
        if counts is None:
            return default_prior
        return (default_prior[0] + counts[0], default_prior[1] + counts[1])

    # ------------------------------------------------------------------
    # Write-Through COW Post-Flight Update
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_delta_to_map(
        target: dict[PosteriorKey, EmpiricalCounts],
        delta: OutcomeDelta,
    ) -> None:
        model_id, role_or_intent, band, success, gamma = delta
        s_inc = 1.0 if success else 0.0
        f_inc = 0.0 if success else 1.0
        for key in ((model_id, role_or_intent, band), (model_id, "*", band)):
            prev_s, prev_f = target.get(key, (0.0, 0.0))
            target[key] = (prev_s * gamma + s_inc, prev_f * gamma + f_inc)

    def record_outcome(
        self,
        model_id: str,
        role_or_intent: str,
        complexity: float | int,
        success: bool,
        decay: float | None = None,
    ) -> None:
        """Update decayed counts in-memory and atomically swap the snapshot."""
        gamma = self._decay_factor if decay is None else decay
        band = complexity_to_band(complexity)
        delta: OutcomeDelta = (model_id, role_or_intent, band, bool(success), gamma)

        with self._write_lock:
            self._apply_delta_to_map(self._staging, delta)
            if self._hydration_count == 0 and not self._inflight_hydration_buffers:
                self._pre_hydration_deltas.append(delta)
            for buf in self._inflight_hydration_buffers:
                buf.append(delta)
            self._snapshot = dict(self._staging)

    # ------------------------------------------------------------------
    # Off-Thread Bulk SQLite Hydration
    # ------------------------------------------------------------------

    def hydrate_from_store(
        self,
        feedback_store: FeedbackStore | None,
        window_days: int | None = None,
        max_timestamp: str | None = None,
    ) -> int:
        """Bulk-load decayed (model, role, band) counts from FeedbackStore without clobbering live deltas."""
        if feedback_store is None:
            return 0
        days = window_days if window_days is not None else self._window_days

        inflight_buf: list[OutcomeDelta] = []
        with self._write_lock:
            pre_deltas = list(self._pre_hydration_deltas) if self._hydration_count == 0 else []
            self._pre_hydration_deltas.clear()
            self._inflight_hydration_buffers.append(inflight_buf)

        try:
            try:
                rows = feedback_store.bulk_band_counts(
                    window_days=days,
                    decay_factor=self._decay_factor,
                    max_timestamp=max_timestamp,
                )
            except TypeError:
                # Backward compatibility with custom/mocked FeedbackStore implementations
                rows = feedback_store.bulk_band_counts(window_days=days)
        except Exception:
            with self._write_lock:
                if inflight_buf in self._inflight_hydration_buffers:
                    self._inflight_hydration_buffers.remove(inflight_buf)
            logger.debug("PosteriorSnapshotCache hydration failed", exc_info=True)
            return 0

        max_eff_weight = (
            1.0 / max(1e-6, 1.0 - self._decay_factor)
            if self._decay_factor < 1.0
            else float("inf")
        )

        loaded: dict[PosteriorKey, EmpiricalCounts] = {}
        wildcard_acc: dict[PosteriorKey, list[float]] = {}
        has_explicit_wildcards = any(str(r) == "*" for _, r, _, _, _ in rows)

        for model_id, role, band, succ, fail in rows:
            s_val = float(succ)
            f_val = float(fail)
            total_w = s_val + f_val
            if total_w > max_eff_weight:
                scale = max_eff_weight / total_w
                s_val *= scale
                f_val *= scale

            key = (str(model_id), str(role), int(band))
            loaded[key] = (s_val, f_val)
            if not has_explicit_wildcards and str(role) != "*":
                w_key = (str(model_id), "*", int(band))
                acc = wildcard_acc.setdefault(w_key, [0.0, 0.0])
                acc[0] += s_val
                acc[1] += f_val

        if not has_explicit_wildcards:
            for w_key, (ws, wf) in wildcard_acc.items():
                total_w = ws + wf
                if total_w > max_eff_weight:
                    scale = max_eff_weight / total_w
                    ws *= scale
                    wf *= scale
                loaded[w_key] = (ws, wf)

        with self._write_lock:
            if inflight_buf in self._inflight_hydration_buffers:
                self._inflight_hydration_buffers.remove(inflight_buf)

            # Start from any staging keys absent in feedback_store, overlay decayed DB counts,
            # then replay pre-hydration & mid-query live deltas so nothing is lost.
            merged: dict[PosteriorKey, EmpiricalCounts] = {
                k: v for k, v in self._staging.items() if k not in loaded
            }
            merged.update(loaded)
            for delta in pre_deltas:
                self._apply_delta_to_map(merged, delta)
            for delta in inflight_buf:
                self._apply_delta_to_map(merged, delta)

            self._staging = merged
            self._snapshot = dict(merged)
            self._last_hydrated_at = time.time()
            self._hydration_count += 1

        return len([1 for _, r, _, _, _ in rows if str(r) != "*"]) or len(rows)

    def hydrate_async(
        self,
        feedback_store: FeedbackStore | None,
        event_reporter: Any | None = None,
        window_days: int | None = None,
    ) -> None:
        """Dispatch ``hydrate_from_store`` onto a background worker without blocking startup."""
        if feedback_store is None:
            return

        # Freeze upper timestamp bound at scheduling time so any post-startup async
        # FeedbackStore writes are accounted exclusively via _pre_hydration_deltas.
        scheduled_utc = datetime.now(timezone.utc).isoformat()

        def _job() -> None:
            self.hydrate_from_store(
                feedback_store,
                window_days=window_days,
                max_timestamp=scheduled_utc,
            )

        if event_reporter is not None and hasattr(event_reporter, "submit"):
            try:
                event_reporter.submit(_job)
                return
            except Exception:
                pass

        t = threading.Thread(target=_job, name="AuraRouter-PosteriorHydrate", daemon=True)
        t.start()

    def status_dict(self) -> dict[str, Any]:
        snap = self._snapshot
        return {
            "entries": len(snap),
            "decay_factor": self._decay_factor,
            "window_days": self._window_days,
            "hydration_count": self._hydration_count,
            "last_hydrated_at": self._last_hydrated_at,
        }
