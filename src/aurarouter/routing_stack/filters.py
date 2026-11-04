"""Stage 1 Non-Compensatory Hard Eligibility Filters.

AuraRouter v2.1 §4.1

A candidate failing ANY hard filter is pruned immediately — low cost or high
headroom can never compensate for a sovereignty violation, capability deficit,
exhausted budget, or depleted rate/circuit state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aurarouter.ratelimit.models import RateState
from aurarouter.routing_stack.context import CandidateEvalContext, FilterVerdict
from aurarouter.savings.pricing import is_cloud_tier, resolve_canonical_vendor

if TYPE_CHECKING:
    from aurarouter.circuit_breaker import CircuitBreakerRegistry
    from aurarouter.config import ConfigLoader
    from aurarouter.ratelimit.tracker import RateLimitTracker
    from aurarouter.savings.budget import BudgetManager


class EligibilityFilter(Protocol):
    """Protocol for Stage 1 hard candidate gates."""

    name: str

    def evaluate(
        self, model_id: str, model_cfg: dict, ctx: CandidateEvalContext
    ) -> FilterVerdict: ...


class SovereigntyFilter:
    """Enforces data-sovereignty category authorization on every candidate."""

    name = "SovereigntyFilter"

    def __init__(self, config: ConfigLoader) -> None:
        self._config = config

    def evaluate(
        self, model_id: str, model_cfg: dict, ctx: CandidateEvalContext
    ) -> FilterVerdict:
        if ctx.bypass_sovereignty or not ctx.effective_data_categories:
            return FilterVerdict(self.name, model_id, passed=True)

        if not model_cfg:
            return FilterVerdict(
                self.name, model_id, passed=False, reason="unconfigured_model"
            )

        allowed = set(self._config.get_model_allowed_data_categories(model_id))
        if ctx.effective_data_categories.issubset(allowed):
            return FilterVerdict(self.name, model_id, passed=True)

        missing = sorted(ctx.effective_data_categories - allowed)
        return FilterVerdict(
            self.name,
            model_id,
            passed=False,
            reason=f"unauthorized_data_categories:{','.join(missing)}",
        )


class CapabilityFilter:
    """Enforces max_complexity ceilings, required capability tags, and hard stateless context limits."""

    name = "CapabilityFilter"

    def evaluate(
        self, model_id: str, model_cfg: dict, ctx: CandidateEvalContext
    ) -> FilterVerdict:
        if not model_cfg:
            return FilterVerdict(
                self.name, model_id, passed=False, reason="unconfigured_model"
            )

        # 1. Complexity ceiling: prevents small models from taking high-complexity tasks
        max_complexity = model_cfg.get("max_complexity")
        if max_complexity is not None:
            try:
                if ctx.complexity_score > int(max_complexity):
                    return FilterVerdict(
                        self.name,
                        model_id,
                        passed=False,
                        reason=f"complexity {ctx.complexity_score} > max_complexity {max_complexity}",
                    )
            except (TypeError, ValueError):
                pass

        # 2. Required capabilities / tags
        if ctx.required_capabilities:
            caps = set(model_cfg.get("capabilities") or ()) | set(model_cfg.get("tags") or ())
            if not ctx.required_capabilities.issubset(caps):
                missing = sorted(ctx.required_capabilities - caps)
                return FilterVerdict(
                    self.name,
                    model_id,
                    passed=False,
                    reason=f"missing_capabilities:{','.join(missing)}",
                )

        # 3. Hard context fit check ONLY when failover projection is disabled
        if not ctx.failover_projection_enabled and ctx.estimated_input_tokens > 0:
            ctx_limit = int(model_cfg.get("context_limit") or model_cfg.get("parameters", {}).get("n_ctx", 0) or 0)
            if ctx_limit > 0 and ctx.estimated_input_tokens > ctx_limit:
                return FilterVerdict(
                    self.name,
                    model_id,
                    passed=False,
                    reason=f"input_tokens {ctx.estimated_input_tokens} > context_limit {ctx_limit}",
                )

        # 4. Hardware VRAM feasibility check via BenchWarmer for local hosting tiers
        max_vram = getattr(ctx, "max_vram_gb", None) or model_cfg.get("max_vram_gb")
        if max_vram is not None and float(max_vram) > 0:
            provider = str(model_cfg.get("provider", "")).lower()
            hosting_tier = model_cfg.get("hosting_tier")
            endpoint = model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")
            if not is_cloud_tier(hosting_tier, provider, endpoint=endpoint):
                quant = str(model_cfg.get("quant") or model_cfg.get("quantization") or getattr(ctx, "quant", "q4_k_m"))
                try:
                    from aurarouter.benchmarks import check_hardware_fit

                    fits, est_vram = check_hardware_fit(
                        model_id,
                        float(max_vram),
                        quant=quant,
                        model_name=model_cfg.get("model_name"),
                    )
                    if fits is False and est_vram is not None:
                        return FilterVerdict(
                            self.name,
                            model_id,
                            passed=False,
                            reason=f"vram_exceeded:{est_vram:.1f}GB>{float(max_vram):.1f}GB({quant})",
                        )
                except Exception:
                    pass

        return FilterVerdict(self.name, model_id, passed=True)


class BudgetCapFilter:
    """Prunes cloud candidates when daily or monthly spend limits are exceeded."""

    name = "BudgetCapFilter"

    def __init__(self, budget_manager: BudgetManager | None) -> None:
        self._budget_manager = budget_manager

    def evaluate(
        self, model_id: str, model_cfg: dict, ctx: CandidateEvalContext
    ) -> FilterVerdict:
        bm = self._budget_manager
        if bm is None:
            return FilterVerdict(self.name, model_id, passed=True)

        provider = str(model_cfg.get("provider", ""))
        hosting_tier = model_cfg.get("hosting_tier")
        endpoint = model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")
        try:
            allowed, reason = bm.peek_budget_allowed(
                provider, hosting_tier=hosting_tier, endpoint=endpoint
            )
        except TypeError:
            allowed, reason = bm.peek_budget_allowed(provider)
        if allowed:
            return FilterVerdict(self.name, model_id, passed=True)
        return FilterVerdict(self.name, model_id, passed=False, reason=reason or "budget_exceeded")


class AvailabilityFilter:
    """Prunes candidates with an open CircuitBreaker or EXHAUSTED/DAILY_EXHAUSTED rate state."""

    name = "AvailabilityFilter"

    def __init__(
        self,
        circuit_breakers: CircuitBreakerRegistry | None,
        rate_tracker: RateLimitTracker | None,
    ) -> None:
        self._cb_registry = circuit_breakers
        self._rate_tracker = rate_tracker

    def evaluate(
        self, model_id: str, model_cfg: dict, ctx: CandidateEvalContext
    ) -> FilterVerdict:
        provider = str(model_cfg.get("provider", ""))
        vendor = resolve_canonical_vendor(model_id, model_cfg)

        # 1. Lock-free CircuitBreaker inspection (checks both model_id and provider keys)
        if self._cb_registry is not None:
            cb = self._cb_registry.peek(model_id) or (
                self._cb_registry.peek(provider) if provider else None
            )
            if cb is not None:
                state, _ = cb.peek_state()
                if state == "open":
                    return FilterVerdict(
                        self.name,
                        model_id,
                        passed=False,
                        reason=f"circuit_open:{model_id}",
                        is_availability_gate=True,
                        recovery_seconds=30.0,
                    )

        # 2. Lock-free GCRA RateLimitTracker inspection
        if self._rate_tracker is not None and (vendor or provider):
            util = self._rate_tracker.peek_utilization(
                model_id, vendor or provider, fallback_provider=provider
            )
            if isinstance(model_cfg, dict):
                model_cfg["_peek_util"] = util
            if util.state == RateState.DAILY_EXHAUSTED:
                return FilterVerdict(
                    self.name,
                    model_id,
                    passed=False,
                    reason=f"rate_daily_exhausted (reset={util.seconds_until_reset:.0f}s)",
                    is_availability_gate=True,
                    recovery_seconds=max(3600.0, util.seconds_until_reset),
                    is_daily_exhausted=True,
                )
            if util.state == RateState.EXHAUSTED:
                return FilterVerdict(
                    self.name,
                    model_id,
                    passed=False,
                    reason=f"rate_exhausted (reset={util.seconds_until_reset:.1f}s)",
                    is_availability_gate=True,
                    recovery_seconds=max(0.5, util.seconds_until_reset),
                )

        return FilterVerdict(self.name, model_id, passed=True)
