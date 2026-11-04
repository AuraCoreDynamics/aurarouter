"""Two-Stage RoutingPolicyStack: Non-Compensatory Gates + Multiplicative Utility.

AuraRouter v2.1 §4
"""

from __future__ import annotations

import random
import threading
import time
from typing import TYPE_CHECKING, Any

from aurarouter.routing_stack.cache import PosteriorSnapshotCache
from aurarouter.routing_stack.context import (
    CandidateEvalContext,
    CandidateScore,
    FactorScore,
    FilterVerdict,
    RoutingDecisionTrace,
)
from aurarouter.routing_stack.factors import (
    CacheAffinityFactor,
    CapabilityBanditFactor,
    CircuitHealthFactor,
    ElasticCostPenalty,
    HeadroomFactor,
)
from aurarouter.routing_stack.filters import (
    AvailabilityFilter,
    BudgetCapFilter,
    CapabilityFilter,
    SovereigntyFilter,
)

if TYPE_CHECKING:
    from aurarouter.circuit_breaker import CircuitBreakerRegistry
    from aurarouter.config import ConfigLoader
    from aurarouter.ratelimit.tracker import RateLimitTracker
    from aurarouter.savings.budget import BudgetManager


class RoutingPolicyStack:
    """Fast, lock-free two-stage candidate filter & utility scorer."""

    def __init__(
        self,
        config: ConfigLoader,
        posterior_cache: PosteriorSnapshotCache,
        rate_tracker: RateLimitTracker | None = None,
        circuit_breakers: CircuitBreakerRegistry | None = None,
        budget_manager: BudgetManager | None = None,
        stack_config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._posterior_cache = posterior_cache
        self._rate_tracker = rate_tracker
        self._cb_registry = circuit_breakers
        self._budget_manager = budget_manager
        cfg = stack_config or {}

        self._mode: str = str(cfg.get("mode", "off")).lower()
        self._bandit_enabled: bool = bool(cfg.get("bandit_enabled", True))
        self._cache_affinity_enabled: bool = bool(cfg.get("cache_affinity_enabled", True))
        lambda_max = float(cfg.get("cost_lambda_max", 0.35))
        kappa = float(cfg.get("cost_elasticity_kappa", 0.32))

        # Stage 1 hard non-compensatory filters (evaluated in strict short-circuit order)
        self._hard_policy_filters = (
            SovereigntyFilter(config),
            CapabilityFilter(),
            BudgetCapFilter(budget_manager),
        )
        self._availability_filter = AvailabilityFilter(circuit_breakers, rate_tracker)

        # Stage 2 factors
        self._bandit = CapabilityBanditFactor(posterior_cache)
        self._headroom = HeadroomFactor(rate_tracker)
        self._health = CircuitHealthFactor(circuit_breakers)
        self._cache_affinity = CacheAffinityFactor(enabled=self._cache_affinity_enabled)
        self._cost_penalty = ElasticCostPenalty(lambda_max=lambda_max, kappa=kappa)

        # True per-thread RNG container so betavariate/gammavariate never race on gauss_next
        self._rng_local = threading.local()

    def _get_thread_rng(self) -> random.Random:
        rng = getattr(self._rng_local, "rng", None)
        if rng is None:
            rng = random.Random()
            self._rng_local.rng = rng
        return rng

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value.lower()

    @property
    def posterior_cache(self) -> PosteriorSnapshotCache:
        return self._posterior_cache

    def evaluate_and_rank(
        self,
        chain: list[str],
        ctx: CandidateEvalContext,
        mode_override: str | None = None,
    ) -> RoutingDecisionTrace:
        """Run Stage 1 hard filters + Stage 2 utility ranking in ~5-20 us."""
        t0 = time.perf_counter()
        active_mode = (mode_override or self._mode).lower()

        if not chain:
            return RoutingDecisionTrace(
                role=ctx.role,
                mode=active_mode,
                input_chain=[],
                ranked_chain=[],
                elapsed_us=(time.perf_counter() - t0) * 1_000_000.0,
            )

        models_map = self._config.config.get("models", {})
        pruned: dict[str, str] = {}
        eligible: list[tuple[int, str, dict]] = []
        availability_pruned: list[tuple[int, str, dict, FilterVerdict]] = []

        # --- Stage 1: Non-Compensatory Hard Eligibility Pipe ---
        for idx, model_id in enumerate(chain):
            model_cfg = models_map.get(model_id) or {}
            hard_failed = False
            for flt in self._hard_policy_filters:
                verdict = flt.evaluate(model_id, model_cfg, ctx)
                if not verdict.passed:
                    pruned[model_id] = f"{verdict.filter_name}: {verdict.reason}"
                    hard_failed = True
                    break
            if hard_failed:
                continue

            avail_verdict = self._availability_filter.evaluate(model_id, model_cfg, ctx)
            if not avail_verdict.passed:
                pruned[model_id] = f"{avail_verdict.filter_name}: {avail_verdict.reason}"
                availability_pruned.append((idx, model_id, model_cfg, avail_verdict))
                continue

            eligible.append((idx, model_id, model_cfg))

        # RT1 Last-Resort Probe Guarantee:
        # If >=1 candidate passed policy/capability/budget gates, and ALL were pruned
        # strictly by AvailabilityFilter, preserve the single candidate with shortest recovery.
        last_resort_probe: str | None = None
        if not eligible and availability_pruned:
            availability_pruned.sort(
                key=lambda item: (
                    1 if item[3].is_daily_exhausted else 0,
                    item[3].recovery_seconds,
                    item[0],
                )
            )
            best_idx, best_mid, best_cfg, _ = availability_pruned[0]
            eligible.append((best_idx, best_mid, best_cfg))
            last_resort_probe = best_mid

        if not eligible:
            return RoutingDecisionTrace(
                role=ctx.role,
                mode=active_mode,
                input_chain=list(chain),
                ranked_chain=[],
                pruned=pruned,
                scores=[],
                last_resort_probe=None,
                elapsed_us=(time.perf_counter() - t0) * 1_000_000.0,
            )

        # --- Stage 2: Multiplicative Utility & Complexity-Elastic Cost Scoring ---
        rng = (
            random.Random(ctx.deterministic_seed)
            if ctx.deterministic_seed is not None
            else self._get_thread_rng()
        )
        now_wall = time.time()

        raw_costs = [
            ElasticCostPenalty.estimate_raw_cost_usd(
                mid, mcfg, ctx.estimated_input_tokens, ctx.estimated_output_tokens
            )
            for _, mid, mcfg in eligible
        ]
        max_cost = max(raw_costs) if raw_costs else 0.0
        lam = self._cost_penalty.lambda_for_complexity(ctx.complexity_score)

        scored: list[CandidateScore] = []
        for (orig_idx, mid, mcfg), raw_cost in zip(eligible, raw_costs):
            if self._bandit_enabled:
                f_q = self._bandit.score(mid, mcfg, ctx, rng)
            else:
                # Neutral constant quality baseline when bandit disabled:
                # skips PosteriorSnapshotCache & tier priors so equal-utility candidates
                # preserve exact YAML chain order via original_index tie-breaking.
                f_q = FactorScore("capability_bandit", 1.0, "disabled")

            f_headroom = self._headroom.score(mid, mcfg)
            f_health = self._health.score(mid, mcfg)
            f_cache = self._cache_affinity.score(mid, mcfg, ctx, now_wall=now_wall)
            f_cost = self._cost_penalty.compute_penalty(
                raw_cost, max_cost, ctx.complexity_score, lam=lam
            )

            utility = (
                f_q.value * f_headroom.value * f_health.value * f_cache.value
                - f_cost.value
            )
            scored.append(
                CandidateScore(
                    model_id=mid,
                    utility=utility,
                    quality_sample=f_q.value,
                    headroom_mult=f_headroom.value,
                    health_mult=f_health.value,
                    cache_mult=f_cache.value,
                    cost_penalty=f_cost.value,
                    raw_cost_usd=raw_cost,
                    original_index=orig_idx,
                    factors=(f_q, f_headroom, f_health, f_cache, f_cost),
                )
            )

        # Sort descending by utility, breaking exact ties deterministically by original YAML chain index
        scored.sort(key=lambda s: (-round(s.utility, 9), s.original_index))
        ranked_chain = [s.model_id for s in scored]

        return RoutingDecisionTrace(
            role=ctx.role,
            mode=active_mode,
            input_chain=list(chain),
            ranked_chain=ranked_chain,
            pruned=pruned,
            scores=scored,
            last_resort_probe=last_resort_probe,
            elapsed_us=(time.perf_counter() - t0) * 1_000_000.0,
        )

    def status_dict(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "bandit_enabled": self._bandit_enabled,
            "cache_affinity_enabled": self._cache_affinity_enabled,
            "posterior_cache": self._posterior_cache.status_dict(),
        }
