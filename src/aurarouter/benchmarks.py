"""Offline LLM Benchmark Oracle & Pareto Routing Integration.

Bridges AuraRouter with BenchWarmer (zero-runtime-dependency benchmark oracle).
Provides:
  1. Empirical cold-start Beta(alpha_0, beta_0) priors for Thompson Sampling (factors.py).
  2. Hardware VRAM feasibility & quantization gating for local nodes (filters.py).
  3. Goodhart smell & contamination anomaly detection / gating.
  4. Automatic catalog model enrichment with HF Open LLM v2 and LMSYS Arena metrics.
  5. Multi-candidate Pareto efficiency frontier calculations.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("AuraRouter.Benchmarks")

_ORACLE_LOCK = threading.Lock()
_ORACLE_INSTANCE: Any = None
_BENCHWARMER_AVAILABLE: bool | None = None

# Thread-safe in-memory resolution & prior caches to preserve <20 us hot-path
_PRIOR_CACHE: dict[tuple[str, int, bool], tuple[float, float] | None] = {}
_RECORD_CACHE: dict[str, Any] = {}
_AUDIT_CACHE: dict[str, tuple[bool, float, list[str]]] = {}
_HARDWARE_FIT_CACHE: dict[tuple[str, float, str], tuple[bool, float | None]] = {}


def is_benchwarmer_available() -> bool:
    """Return True if benchwarmer is importable and operational."""
    global _BENCHWARMER_AVAILABLE
    if _BENCHWARMER_AVAILABLE is not None:
        return _BENCHWARMER_AVAILABLE
    try:
        import benchwarmer  # noqa: F401

        _BENCHWARMER_AVAILABLE = True
    except ImportError:
        _BENCHWARMER_AVAILABLE = False
    return _BENCHWARMER_AVAILABLE


def get_benchwarmer(db_path: str | None = None) -> Any:
    """Return singleton BenchWarmer instance, or None if unavailable."""
    global _ORACLE_INSTANCE, _BENCHWARMER_AVAILABLE
    if _ORACLE_INSTANCE is not None:
        return _ORACLE_INSTANCE

    if not is_benchwarmer_available():
        return None

    with _ORACLE_LOCK:
        if _ORACLE_INSTANCE is None:
            try:
                from benchwarmer import BenchWarmer

                _ORACLE_INSTANCE = BenchWarmer(db_path=db_path)
                _BENCHWARMER_AVAILABLE = True
                logger.info("BenchWarmer oracle initialized successfully.")
            except Exception as exc:
                logger.warning("Failed to initialize BenchWarmer oracle: %s", exc)
                _BENCHWARMER_AVAILABLE = False
                _ORACLE_INSTANCE = None
        return _ORACLE_INSTANCE


def reset_benchwarmer_cache() -> None:
    """Flush in-memory oracle caches (for tests or DB reload)."""
    global _ORACLE_INSTANCE, _BENCHWARMER_AVAILABLE
    with _ORACLE_LOCK:
        _PRIOR_CACHE.clear()
        _RECORD_CACHE.clear()
        _AUDIT_CACHE.clear()
        _HARDWARE_FIT_CACHE.clear()
        if _ORACLE_INSTANCE is not None:
            try:
                _ORACLE_INSTANCE.reload()
            except Exception:
                pass


def resolve_model_record(model_id: str, model_name: str | None = None) -> Any:
    """Resolve model_id or model_name to ModelRecord via BenchWarmer cascade."""
    if not model_id:
        return None

    cache_key = f"{model_id}:{model_name or ''}"
    if cache_key in _RECORD_CACHE:
        return _RECORD_CACHE[cache_key]

    oracle = get_benchwarmer()
    if oracle is None:
        _RECORD_CACHE[cache_key] = None
        return None

    # Try model_id first, then model_name fallback
    rec = None
    for query in (model_id, model_name):
        if not query:
            continue
        try:
            rec = oracle.get_model(query)
            if rec is not None:
                break
        except Exception:
            rec = None

    _RECORD_CACHE[cache_key] = rec
    return rec


def get_benchmark_prior(
    model_id: str,
    band: int,
    model_cfg: dict | None = None,
    vision: bool = False,
) -> tuple[float, float] | None:
    """Calculate empirical Beta(alpha_0, beta_0) prior for (model_id, band).

    Band definition (AuraRouter v2.1 §4):
      - Band 0 (complexity <= 3): Instruction following (`ifeval`) & conversational (`arena_elo`)
      - Band 1 (complexity 4..7): Balanced instruction & reasoning (`ifeval`, `bbh`, `mmlu_pro`)
      - Band 2 (complexity >= 8): Hard reasoning & logic (`math_hard`, `gpqa`, `musr`)

    When *vision* is True, prioritizes multimodal benchmarks (`mmmu`, `ocrbench`,
    `docvqa`, `mathvista`, `vision_arena_elo`) when available, falling back to
    the standard text band prior if absent.

    Returns:
      (alpha_0, beta_0) with effective pseudo-sample weight nu=5.0,
      or None if the model cannot be resolved to benchmark data.
    """
    cache_key = (model_id, band, bool(vision))
    if cache_key in _PRIOR_CACHE:
        return _PRIOR_CACHE[cache_key]

    model_name = model_cfg.get("model_name") if isinstance(model_cfg, dict) else None
    rec = resolve_model_record(model_id, model_name=model_name)
    if rec is None or not rec.scores:
        _PRIOR_CACHE[cache_key] = None
        return None

    scores = rec.scores
    composite_score: float | None = None

    if vision:
        v_scores: list[tuple[float, float]] = []
        for key, w in (
            ("mmmu", 0.30),
            ("ocrbench", 0.25),
            ("docvqa", 0.20),
            ("mathvista", 0.15),
            ("vision_arena_elo", 0.10),
        ):
            if key in scores and scores[key] is not None:
                val = float(scores[key])
                if key == "vision_arena_elo" and val > 100.0:
                    val = max(0.0, min(100.0, (val - 1000.0) / 4.0))
                v_scores.append((val, w))
        if v_scores:
            tot_w = sum(w for _, w in v_scores)
            composite_score = sum(s * w for s, w in v_scores) / tot_w

    if composite_score is None:
        if band == 0:
            # Band 0: prioritize ifeval (60%) and arena_elo (40%)
            b0_scores: list[tuple[float, float]] = []
            for key, w in (("ifeval", 0.60), ("arena_elo", 0.40)):
                if key in scores and scores[key] is not None:
                    val = float(scores[key])
                    if key == "arena_elo" and val > 100.0:
                        val = max(0.0, min(100.0, (val - 1000.0) / 4.0))
                    b0_scores.append((val, w))
            if b0_scores:
                tot_w = sum(w for _, w in b0_scores)
                composite_score = sum(s * w for s, w in b0_scores) / tot_w
            else:
                composite_score = 50.0
        elif band == 1:
            # Band 1: balanced ifeval (35%), bbh (35%), mmlu_pro (30%)
            w_scores = []
            for key, w in (("ifeval", 0.35), ("bbh", 0.35), ("mmlu_pro", 0.30)):
                if key in scores and scores[key] is not None:
                    w_scores.append((float(scores[key]), w))
            if w_scores:
                tot_w = sum(w for _, w in w_scores)
                composite_score = sum(s * w for s, w in w_scores) / tot_w
            else:
                composite_score = 50.0
        else:
            # Band 2: hard math/reasoning: math_hard (40%), gpqa (30%), musr (30%)
            w_scores = []
            for key, w in (("math_hard", 0.40), ("gpqa", 0.30), ("musr", 0.30)):
                if key in scores and scores[key] is not None:
                    w_scores.append((float(scores[key]), w))
            if w_scores:
                tot_w = sum(w for _, w in w_scores)
                composite_score = sum(s * w for s, w in w_scores) / tot_w
            else:
                composite_score = 45.0

    # Map normalized score [0..100] to prior mean mu in [0.20, 0.95]
    mu = max(0.20, min(0.95, composite_score / 100.0))
    nu = 5.0  # Equivalent pseudo-observations
    alpha_0 = round(mu * nu, 2)
    beta_0 = round((1.0 - mu) * nu, 2)
    prior = (alpha_0, beta_0)

    _PRIOR_CACHE[cache_key] = prior
    return prior


def check_hardware_fit(
    model_id: str,
    max_vram_gb: float,
    quant: str = "q4_k_m",
    model_name: str | None = None,
) -> tuple[bool, float | None]:
    """Check if model fits within target host VRAM envelope under quantization.

    Returns:
      (fits: bool, estimated_vram_gb: float | None)
      If model is unknown to benchmark store, returns (True, None) permissively.
    """
    cache_key = (model_id, round(max_vram_gb, 2), quant.lower())
    if cache_key in _HARDWARE_FIT_CACHE:
        return _HARDWARE_FIT_CACHE[cache_key]

    rec = resolve_model_record(model_id, model_name=model_name)
    if rec is None or not rec.estimated_vram_gb:
        res = (True, None)
        _HARDWARE_FIT_CACHE[cache_key] = res
        return res

    quant_clean = quant.lower()
    est_vram = rec.estimated_vram_gb.get(quant_clean)
    if est_vram is None:
        # Fallback to q4_k_m or nearest
        est_vram = rec.estimated_vram_gb.get("q4_k_m") or rec.estimated_vram_gb.get("bf16")

    if est_vram is None:
        res = (True, None)
    else:
        fits = est_vram <= max_vram_gb
        res = (fits, est_vram)

    _HARDWARE_FIT_CACHE[cache_key] = res
    return res


def check_model_contamination(
    model_id: str,
    model_name: str | None = None,
) -> tuple[bool, float, list[str]]:
    """Audit candidate model for Goodhart contamination and reasoning cliffs.

    Returns:
      (is_high_risk: bool, anomaly_score: float, flags: list[str])
    """
    cache_key = f"{model_id}:{model_name or ''}"
    if cache_key in _AUDIT_CACHE:
        return _AUDIT_CACHE[cache_key]

    oracle = get_benchwarmer()
    if oracle is None:
        res = (False, 0.0, [])
        _AUDIT_CACHE[cache_key] = res
        return res

    for query in (model_id, model_name):
        if not query:
            continue
        try:
            report = oracle.detect_anomalies(query)
            if report is not None:
                is_high_risk = report.status == "HIGH_CONTAMINATION_RISK"
                res = (is_high_risk, float(report.anomaly_score), list(report.flags))
                _AUDIT_CACHE[cache_key] = res
                return res
        except Exception:
            pass

    res = (False, 0.0, [])
    _AUDIT_CACHE[cache_key] = res
    return res


def enrich_model_metadata(model_id: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Auto-enrich a model configuration dictionary with empirical benchmark metrics.

    Non-destructive: only sets fields that are not already explicitly configured.
    """
    model_name = model_cfg.get("model_name")
    rec = resolve_model_record(model_id, model_name=model_name)
    if rec is None:
        return model_cfg

    enriched = dict(model_cfg)
    if "params_b" not in enriched and rec.params_b is not None:
        enriched["params_b"] = rec.params_b

    if "benchmarks" not in enriched and rec.scores:
        enriched["benchmarks"] = dict(rec.scores)

    if "estimated_vram_gb" not in enriched and rec.estimated_vram_gb:
        enriched["estimated_vram_gb"] = dict(rec.estimated_vram_gb)

    if "architecture" not in enriched and rec.architecture:
        enriched["architecture"] = rec.architecture

    return enriched


def compare_candidates(model_a: str, model_b: str) -> dict[str, Any] | None:
    """Compare two candidate models side-by-side with exact micro-deltas."""
    oracle = get_benchwarmer()
    if oracle is None:
        return None
    try:
        return oracle.compare(model_a, model_b).to_dict()
    except Exception as exc:
        logger.warning("BenchWarmer compare failed for (%s, %s): %s", model_a, model_b, exc)
        return None


def get_pareto_frontier(
    metric: str = "hf_v2_avg",
    max_vram_gb: float | None = None,
    quant: str = "q4_k_m",
) -> list[dict[str, Any]]:
    """Calculate capability vs VRAM Pareto-optimal efficiency frontier."""
    oracle = get_benchwarmer()
    if oracle is None:
        return []
    try:
        pts = oracle.pareto_frontier(metric=metric, max_vram_gb=max_vram_gb, quant=quant)
        return [pt.to_dict() for pt in pts]
    except Exception as exc:
        logger.warning("BenchWarmer pareto_frontier failed: %s", exc)
        return []


def score_candidates_weighted(
    candidates: list[str],
    weights: dict[str, float] | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Score candidate models against task-specific benchmark weightings."""
    oracle = get_benchwarmer()
    if oracle is None:
        return []
    w = weights or {"ifeval": 0.4, "math_hard": 0.4, "mmlu_pro": 0.2}
    try:
        results = oracle.score_candidates(candidates=candidates, weights=w, strict_benchmarks=strict)
        return [r.to_dict() for r in results]
    except Exception as exc:
        logger.warning("BenchWarmer score_candidates failed: %s", exc)
        return []

