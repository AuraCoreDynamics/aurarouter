"""Stage 2 Multiplicative Utility & Complexity-Elastic Cost Factors.

AuraRouter v2.1 §4.2

Utility equation:
    U(M | x) = Q_hat(M, x) * Phi_headroom(M) * Phi_health(M) * Phi_cache(M, s)
               - lambda(c) * Cost_tilde(M, x)
"""

from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING

from aurarouter.routing_stack.cache import PosteriorSnapshotCache
from aurarouter.routing_stack.context import (
    CandidateEvalContext,
    FactorScore,
    complexity_to_band,
)
from aurarouter.savings.pricing import (
    _BUILTIN_PRICES,
    is_cloud_tier,
    resolve_canonical_vendor,
)

if TYPE_CHECKING:
    from aurarouter.circuit_breaker import CircuitBreakerRegistry
    from aurarouter.ratelimit.tracker import RateLimitTracker


# Provider prefix/KV cache TTLs and stickiness bonuses (v2.1 §4.2)
_CACHE_TTL_BY_PROVIDER: dict[str, tuple[float, float]] = {
    "anthropic": (300.0, 0.18),
    "claude": (300.0, 0.18),
    "openai": (300.0, 0.17),
    "azure_openai": (300.0, 0.17),
    "mcp": (300.0, 0.16),
    "google": (300.0, 0.15),
    "gemini": (300.0, 0.15),
    "openapi": (300.0, 0.15),
    "llamacpp-server": (120.0, 0.20),
    "llamacpp": (120.0, 0.20),
    "ollama": (120.0, 0.18),
}


def resolve_cold_start_prior(model_id: str, model_cfg: dict, band: int) -> tuple[float, float]:
    """Informative cold-start Beta(alpha_0, beta_0) prior by model tier and complexity band."""
    key = f"_cold_prior_{band}"
    if isinstance(model_cfg, dict):
        cached = model_cfg.get(key)
        if cached is not None:
            return cached

    # 1. Try empirical benchmark oracle (BenchWarmer)
    prior: tuple[float, float] | None = None
    try:
        from aurarouter.benchmarks import get_benchmark_prior

        prior = get_benchmark_prior(model_id, band, model_cfg=model_cfg)
    except Exception:
        prior = None

    # 2. Fallback to keyword heuristics if unbenchmarked
    if prior is None:
        provider = str(model_cfg.get("provider", "")).lower()
        hosting_tier = model_cfg.get("hosting_tier")
        name_lower = f"{model_id} {model_cfg.get('model_name', '')}".lower()

        endpoint = model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")
        if is_cloud_tier(hosting_tier, provider, endpoint=endpoint):
            if any(k in name_lower for k in ("opus", "pro", "gpt-4", "gpt-5", "o1", "o3", "o4", "sonnet")):
                prior = (4.0, 1.0)  # Frontier cloud mean = 0.80
            else:
                prior = (3.2, 1.0)  # Fast cloud mean = 0.76
        elif band == 0:
            prior = (3.6, 1.0)      # Low complexity: local excels (mean ~0.78)
        elif band == 1:
            prior = (2.6, 1.2)      # Medium complexity (mean ~0.68)
        else:
            prior = (1.8, 1.8)      # High complexity (mean 0.50 until proven)

    if isinstance(model_cfg, dict):
        model_cfg[key] = prior
    return prior


class CapabilityBanditFactor:
    """Contextual Thompson Sampling over lock-free PosteriorSnapshotCache."""

    name = "capability_bandit"

    def __init__(self, cache: PosteriorSnapshotCache) -> None:
        self._cache = cache

    def score(
        self,
        model_id: str,
        model_cfg: dict,
        ctx: CandidateEvalContext,
        rng: random.Random,
    ) -> FactorScore:
        band = complexity_to_band(ctx.complexity_score)
        prior = resolve_cold_start_prior(model_id, model_cfg, band)
        alpha, beta = self._cache.get_posterior(model_id, ctx.role, band, default_prior=prior)

        if ctx.deterministic and ctx.deterministic_seed is None:
            q_hat = alpha / (alpha + beta)
            mode = "mean"
        else:
            q_hat = rng.betavariate(alpha, beta)
            mode = "thompson"

        return FactorScore(
            self.name,
            max(0.01, min(1.0, q_hat)),
            f"{mode} Beta({alpha:.1f},{beta:.1f}) band={band}",
        )


class HeadroomFactor:
    """Smooth quadratic rate-limit & GPU slot headroom multiplier in (0, 1]."""

    name = "headroom"

    def __init__(self, rate_tracker: RateLimitTracker | None) -> None:
        self._rate_tracker = rate_tracker

    def score(self, model_id: str, model_cfg: dict) -> FactorScore:
        if self._rate_tracker is None:
            return FactorScore(self.name, 1.0, "untracked")
        util = model_cfg.get("_peek_util") if isinstance(model_cfg, dict) else None
        if util is None:
            vendor = resolve_canonical_vendor(model_id, model_cfg)
            raw_provider = str(model_cfg.get("provider", ""))
            util = self._rate_tracker.peek_utilization(model_id, vendor, fallback_provider=raw_provider)
        val = max(0.05, min(1.0, util.headroom_score))
        return FactorScore(
            self.name,
            val,
            f"u={util.composite_utilization:.2f} state={util.state.value}",
        )


class CircuitHealthFactor:
    """Penalizes half-open recovery probes or recent consecutive transient errors."""

    name = "circuit_health"

    def __init__(self, circuit_breakers: CircuitBreakerRegistry | None) -> None:
        self._cb_registry = circuit_breakers

    def score(self, model_id: str, model_cfg: dict) -> FactorScore:
        if self._cb_registry is None:
            return FactorScore(self.name, 1.0, "untracked")
        provider = str(model_cfg.get("provider", ""))
        cb = self._cb_registry.peek(model_id) or (
            self._cb_registry.peek(provider) if provider else None
        )
        if cb is None:
            return FactorScore(self.name, 1.0, "closed(0)")
        state, failures = cb.peek_state()
        if state == "half_open":
            return FactorScore(self.name, 0.35, "half_open_probe")
        if failures > 0:
            mult = max(0.40, 1.0 - 0.12 * failures)
            return FactorScore(self.name, mult, f"closed(failures={failures})")
        return FactorScore(self.name, 1.0, "closed(0)")


class CacheAffinityFactor:
    """KV / prompt-cache stickiness bonus [1.0, 1.0 + gamma] within provider cache TTL."""

    name = "cache_affinity"

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def score(
        self,
        model_id: str,
        model_cfg: dict,
        ctx: CandidateEvalContext,
        now_wall: float | None = None,
    ) -> FactorScore:
        if (
            not self._enabled
            or not ctx.last_model_id
            or ctx.last_model_id != model_id
            or ctx.last_routed_at is None
        ):
            return FactorScore(self.name, 1.0, "cold")

        vendor = resolve_canonical_vendor(model_id, model_cfg)
        provider = str(model_cfg.get("provider", "")).lower()
        ttl, gamma = _CACHE_TTL_BY_PROVIDER.get(
            vendor, _CACHE_TTL_BY_PROVIDER.get(provider, (180.0, 0.15))
        )
        now_ts = now_wall if now_wall is not None else time.time()
        age = max(0.0, now_ts - ctx.last_routed_at)
        if age >= ttl:
            return FactorScore(self.name, 1.0, f"expired({age:.0f}s>={ttl:.0f}s)")

        bonus = 1.0 + gamma * (1.0 - (age / ttl))
        return FactorScore(self.name, bonus, f"warm(age={age:.0f}s/{ttl:.0f}s)")


class ElasticCostPenalty:
    """Complexity-elastic normalized cost penalty lambda(c) * Cost_tilde(M, x)."""

    name = "elastic_cost"

    def __init__(self, lambda_max: float = 0.35, kappa: float = 0.32) -> None:
        self._lambda_max = max(0.0, lambda_max)
        self._kappa = max(0.0, kappa)

    @staticmethod
    def estimate_raw_cost_usd(
        model_id: str, model_cfg: dict, est_in: int, est_out: int
    ) -> float:
        """Lock-free dollar cost estimate for a candidate model."""
        inp_rate = model_cfg.get("cost_per_1m_input")
        out_rate = model_cfg.get("cost_per_1m_output")
        if inp_rate is not None and out_rate is not None:
            in_tok = max(64, est_in)
            out_tok = max(64, est_out)
            return (in_tok * float(inp_rate) + out_tok * float(out_rate)) / 1_000_000.0

        provider = str(model_cfg.get("provider", ""))
        endpoint = model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")
        if not is_cloud_tier(model_cfg.get("hosting_tier"), provider, endpoint=endpoint):
            inp_rate = float(inp_rate or 0.0)
            out_rate = float(out_rate or 0.0)
        else:
            model_name = str(model_cfg.get("model_name", model_id))
            vendor = resolve_canonical_vendor(model_id, model_cfg)
            price = (
                _BUILTIN_PRICES.get(model_name)
                or _BUILTIN_PRICES.get(f"{vendor}:*")
                or _BUILTIN_PRICES.get(f"{provider}:*")
            )
            if price is not None:
                inp_rate = price.input_per_million if inp_rate is None else inp_rate
                out_rate = price.output_per_million if out_rate is None else out_rate
            else:
                inp_rate = float(inp_rate or 0.0)
                out_rate = float(out_rate or 0.0)

        in_tok = max(64, est_in)
        out_tok = max(64, est_out)
        return (in_tok * float(inp_rate) + out_tok * float(out_rate)) / 1_000_000.0

    def lambda_for_complexity(self, complexity_score: int) -> float:
        c = max(1, min(10, int(complexity_score)))
        return self._lambda_max * math.exp(-self._kappa * (c - 1))

    def compute_penalty(
        self,
        raw_cost: float,
        max_chain_cost: float,
        complexity_score: int,
        lam: float | None = None,
    ) -> FactorScore:
        if max_chain_cost <= 1e-9 or raw_cost <= 0.0:
            return FactorScore(self.name, 0.0, "free_or_zero")
        norm_cost = min(1.0, raw_cost / max_chain_cost)
        lam_val = lam if lam is not None else self.lambda_for_complexity(complexity_score)
        penalty = lam_val * norm_cost
        return FactorScore(
            self.name,
            penalty,
            f"lam={lam_val:.3f}*norm={norm_cost:.2f}(${raw_cost:.5f})",
        )


class ContaminationAuditFactor:
    """Stage 2 Multiplicative Factor dampening Goodhart overfit smells & benchmark contamination."""

    name = "contamination_audit"

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def score(self, model_id: str, model_cfg: dict) -> FactorScore:
        if not self._enabled:
            return FactorScore(self.name, 1.0, "disabled")

        try:
            from aurarouter.benchmarks import check_model_contamination

            model_name = model_cfg.get("model_name") if isinstance(model_cfg, dict) else None
            is_high_risk, score, flags = check_model_contamination(model_id, model_name=model_name)
            if is_high_risk:
                flag_str = ",".join(flags[:2]) if flags else "anomaly"
                return FactorScore(
                    self.name,
                    0.60,
                    f"high_contamination_risk:score={score:.1f};flags={flag_str}",
                )
            if score > 40.0:
                return FactorScore(
                    self.name,
                    0.85,
                    f"moderate_anomaly_score:{score:.1f}",
                )
            return FactorScore(self.name, 1.0, "clean")
        except Exception:
            return FactorScore(self.name, 1.0, "audit_unavailable")
