"""Comprehensive tests for BenchWarmer integration in AuraRouter.

Tests:
  1. Oracle initialization, resolution cascade, and caching
  2. Empirical Beta(alpha_0, beta_0) prior calculation across complexity bands
  3. Hardware VRAM feasibility checking across quantizations
  4. Goodhart contamination smell & anomaly detection
  5. Automatic model metadata catalog enrichment
  6. Stage 1 CapabilityFilter hardware VRAM gating
  7. Stage 2 ContaminationAuditFactor utility dampening
  8. Two-Stage RoutingPolicyStack integration with benchmark priors & audit factor
  9. AuraRouterAPI & MCP tools surfaces
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.benchmarks import (
    check_hardware_fit,
    check_model_contamination,
    compare_candidates,
    enrich_model_metadata,
    get_benchmark_prior,
    get_benchwarmer,
    get_pareto_frontier,
    is_benchwarmer_available,
    reset_benchwarmer_cache,
    resolve_model_record,
    score_candidates_weighted,
)
from aurarouter.config import ConfigLoader
from aurarouter.mcp_tools import (
    benchmarks_compare,
    benchmarks_model,
    benchmarks_pareto,
)
from aurarouter.routing_stack.cache import PosteriorSnapshotCache
from aurarouter.routing_stack.context import CandidateEvalContext
from aurarouter.routing_stack.factors import (
    ContaminationAuditFactor,
    resolve_cold_start_prior,
)
from aurarouter.routing_stack.filters import CapabilityFilter
from aurarouter.routing_stack.stack import RoutingPolicyStack


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_benchwarmer_cache()
    yield
    reset_benchwarmer_cache()


# ---------------------------------------------------------------------------
# 1. Oracle Core & Resolution
# ---------------------------------------------------------------------------


class TestBenchWarmerOracleCore:
    def test_oracle_is_available(self):
        assert is_benchwarmer_available() is True
        oracle = get_benchwarmer()
        assert oracle is not None

    def test_resolve_exact_model_id(self):
        rec = resolve_model_record("meta-llama/Llama-3.1-70B-Instruct")
        assert rec is not None
        assert rec.model_id == "meta-llama/Llama-3.1-70B-Instruct"
        assert rec.params_b == pytest.approx(70.55, abs=0.1)
        assert "ifeval" in rec.scores
        assert "math_hard" in rec.scores

    def test_resolve_short_name_cascade(self):
        # Case-insensitive / stripped / FTS5 cascade
        rec_phi = resolve_model_record("phi-4")
        assert rec_phi is not None
        assert rec_phi.model_id == "microsoft/Phi-4"

        rec_qwen = resolve_model_record("qwen2.5-72b")
        assert rec_qwen is not None
        assert rec_qwen.model_id == "Qwen/Qwen2.5-72B-Instruct"

    def test_resolve_unknown_model_returns_none(self):
        rec = resolve_model_record("unknown-vendor/non-existent-model-xyz")
        assert rec is None


# ---------------------------------------------------------------------------
# 2. Empirical Beta Priors Across Bands
# ---------------------------------------------------------------------------


class TestEmpiricalBetaPriors:
    def test_empirical_prior_band_0_instruction_following(self):
        prior = get_benchmark_prior("meta-llama/Llama-3.1-70B-Instruct", band=0)
        assert prior is not None
        alpha_0, beta_0 = prior
        # Llama 3.1 70B has high ifeval (84.6) and arena_elo (73.2)
        assert alpha_0 > 3.5
        assert beta_0 < 1.5
        assert alpha_0 + beta_0 == pytest.approx(5.0, abs=0.1)

    def test_empirical_prior_band_2_hard_reasoning(self):
        prior_70b = get_benchmark_prior("meta-llama/Llama-3.1-70B-Instruct", band=2)
        prior_8b = get_benchmark_prior("meta-llama/Llama-3.1-8B-Instruct", band=2)
        assert prior_70b is not None
        assert prior_8b is not None

        # 70B significantly outperforms 8B on hard math/gpqa/musr
        alpha_70b, _ = prior_70b
        alpha_8b, _ = prior_8b
        assert alpha_70b > alpha_8b

    def test_unbenchmarked_model_returns_none(self):
        prior = get_benchmark_prior("custom-finetune/unbenchmarked-lora", band=1)
        assert prior is None

    def test_resolve_cold_start_prior_integrates_oracle_with_heuristic_fallback(self):
        # 1. Model in benchmark store gets empirical prior
        mcfg_benchmarked = {"provider": "ollama", "model_name": "Llama-3.1-70B-Instruct"}
        prior_bench = resolve_cold_start_prior("llama70b", mcfg_benchmarked, band=0)
        assert prior_bench[0] > 3.5  # empirical

        # 2. Unknown model falls back cleanly to keyword heuristic
        mcfg_heuristic = {"provider": "anthropic", "model_name": "claude-future-sonnet", "hosting_tier": "cloud"}
        prior_heur = resolve_cold_start_prior("claude-custom-sonnet", mcfg_heuristic, band=0)
        assert prior_heur == (4.0, 1.0)  # frontier cloud mean = 0.80


# ---------------------------------------------------------------------------
# 3. Hardware Feasibility & Quantization
# ---------------------------------------------------------------------------


class TestHardwareVRAMFeasibility:
    def test_vram_fit_under_different_envelopes(self):
        # Llama-3.1-70B requires ~48.7 GB under q4_k_m
        fits_24gb, est_24gb = check_hardware_fit("meta-llama/Llama-3.1-70B-Instruct", max_vram_gb=24.0, quant="q4_k_m")
        assert fits_24gb is False
        assert est_24gb is not None
        assert est_24gb > 40.0

        fits_80gb, est_80gb = check_hardware_fit("meta-llama/Llama-3.1-70B-Instruct", max_vram_gb=80.0, quant="q4_k_m")
        assert fits_80gb is True
        assert est_80gb == est_24gb

    def test_vram_fit_quantization_variance(self):
        # 8B model fits in 8GB VRAM under q4_k_m (~5.5 GB) but exceeds in fp16 (~18.4 GB)
        fits_q4, est_q4 = check_hardware_fit("meta-llama/Llama-3.1-8B-Instruct", max_vram_gb=8.0, quant="q4_k_m")
        assert fits_q4 is True
        assert est_q4 is not None and est_q4 < 8.0

        fits_fp16, est_fp16 = check_hardware_fit("meta-llama/Llama-3.1-8B-Instruct", max_vram_gb=8.0, quant="fp16")
        assert fits_fp16 is False
        assert est_fp16 is not None and est_fp16 > 15.0

    def test_unbenchmarked_model_permissive_pass(self):
        fits, est = check_hardware_fit("mystery-model", max_vram_gb=16.0)
        assert fits is True
        assert est is None


# ---------------------------------------------------------------------------
# 4. Goodhart Smell & Contamination Anomaly Detection
# ---------------------------------------------------------------------------


class TestContaminationAudit:
    def test_detect_goodhart_overfit_trap(self):
        is_high_risk, score, flags = check_model_contamination("overfit-ai/Goodhart-Overfit-7B")
        assert is_high_risk is True
        assert score >= 80.0
        assert len(flags) > 0
        assert any("IFEval" in f or "Reasoning cliff" in f for f in flags)

    def test_clean_model_passes_audit(self):
        is_high_risk, score, flags = check_model_contamination("meta-llama/Llama-3.1-70B-Instruct")
        assert is_high_risk is False
        assert score == 0.0
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# 5. Model Configuration Auto-Enrichment
# ---------------------------------------------------------------------------


class TestModelCatalogEnrichment:
    def test_enrich_model_metadata(self):
        raw_cfg = {
            "provider": "ollama",
            "model_name": "phi-4",
            "tags": ["local", "coding"],
        }
        enriched = enrich_model_metadata("phi-4", raw_cfg)

        assert enriched["params_b"] == pytest.approx(14.0, abs=0.5)
        assert "benchmarks" in enriched
        assert "ifeval" in enriched["benchmarks"]
        assert "estimated_vram_gb" in enriched
        assert "q4_k_m" in enriched["estimated_vram_gb"]
        assert enriched["tags"] == ["local", "coding"]  # preserved


# ---------------------------------------------------------------------------
# 6. Stage 1 CapabilityFilter Hardware Gating
# ---------------------------------------------------------------------------


class TestCapabilityFilterHardwareGating:
    def test_local_model_filtered_when_vram_exceeded(self):
        flt = CapabilityFilter()
        model_cfg = {
            "provider": "ollama",
            "model_name": "Llama-3.1-70B-Instruct",
            "hosting_tier": "local",
        }
        # Context requesting 24 GB VRAM limit
        ctx = CandidateEvalContext(
            role="coding",
            max_vram_gb=24.0,
            quant="q4_k_m",
        )
        verdict = flt.evaluate("meta-llama/Llama-3.1-70B-Instruct", model_cfg, ctx)
        assert verdict.passed is False
        assert "vram_exceeded" in verdict.reason

    def test_local_model_passes_when_vram_sufficient(self):
        flt = CapabilityFilter()
        model_cfg = {
            "provider": "ollama",
            "model_name": "Llama-3.1-8B-Instruct",
            "hosting_tier": "local",
        }
        ctx = CandidateEvalContext(
            role="coding",
            max_vram_gb=24.0,
            quant="q4_k_m",
        )
        verdict = flt.evaluate("meta-llama/Llama-3.1-8B-Instruct", model_cfg, ctx)
        assert verdict.passed is True


# ---------------------------------------------------------------------------
# 7. Stage 2 ContaminationAuditFactor
# ---------------------------------------------------------------------------


class TestContaminationAuditFactor:
    def test_audit_factor_dampens_high_risk_model(self):
        factor = ContaminationAuditFactor(enabled=True)
        model_cfg = {"model_name": "Goodhart-Overfit-7B"}
        score = factor.score("overfit-ai/Goodhart-Overfit-7B", model_cfg)

        assert score.factor_name == "contamination_audit"
        assert score.value == pytest.approx(0.60, abs=1e-3)
        assert "high_contamination_risk" in score.detail

    def test_audit_factor_neutral_for_clean_model(self):
        factor = ContaminationAuditFactor(enabled=True)
        model_cfg = {"model_name": "Llama-3.1-70B-Instruct"}
        score = factor.score("meta-llama/Llama-3.1-70B-Instruct", model_cfg)

        assert score.value == 1.0
        assert score.detail == "clean"

    def test_audit_factor_disabled_returns_unity(self):
        factor = ContaminationAuditFactor(enabled=False)
        model_cfg = {"model_name": "Goodhart-Overfit-7B"}
        score = factor.score("overfit-ai/Goodhart-Overfit-7B", model_cfg)

        assert score.value == 1.0
        assert score.detail == "disabled"


# ---------------------------------------------------------------------------
# 8. Full Stack Ranking with Benchmark Oracle
# ---------------------------------------------------------------------------


class TestRoutingPolicyStackWithBenchmarks:
    def test_stack_ranks_and_penalizes_contaminated_model(self, tmp_path):
        config_data = {
            "models": {
                "clean_model": {
                    "provider": "ollama",
                    "model_name": "meta-llama/Llama-3.1-70B-Instruct",
                    "hosting_tier": "local",
                },
                "trap_model": {
                    "provider": "ollama",
                    "model_name": "overfit-ai/Goodhart-Overfit-7B",
                    "hosting_tier": "local",
                },
            },
            "roles": {
                "reasoning": ["trap_model", "clean_model"],
            },
        }
        cfg_file = tmp_path / "auraconfig.yaml"
        import yaml
        cfg_file.write_text(yaml.dump(config_data))

        config_loader = ConfigLoader(config_path=str(cfg_file))
        cache = PosteriorSnapshotCache()
        stack = RoutingPolicyStack(
            config=config_loader,
            posterior_cache=cache,
            stack_config={"mode": "active", "bandit_enabled": True, "audit_enabled": True},
        )

        ctx = CandidateEvalContext(
            role="reasoning",
            complexity_score=8,  # Hard task
            deterministic=True,
        )

        trace = stack.evaluate_and_rank(["trap_model", "clean_model"], ctx)
        # Even though trap_model was first in the YAML chain, clean_model wins utility
        assert trace.ranked_chain[0] == "clean_model"

        # Check factor breakdown on trap_model
        trap_score = next(s for s in trace.scores if s.model_id == "trap_model")
        assert trap_score.audit_mult == pytest.approx(0.60, abs=1e-3)
        audit_factor = next(f for f in trap_score.factors if f.factor_name == "contamination_audit")
        assert "high_contamination_risk" in audit_factor.detail


# ---------------------------------------------------------------------------
# 9. Public API & MCP Tools Surfaces
# ---------------------------------------------------------------------------


class TestBenchmarkAPIAndMCPTools:
    def test_api_benchmark_methods(self, tmp_path):
        cfg_file = tmp_path / "auraconfig.yaml"
        cfg_file.write_text("models: {}\nroles: {}\n")
        api = AuraRouterAPI(APIConfig(config_path=str(cfg_file)))

        # 1. get_model_benchmarks
        bench = api.get_model_benchmarks("phi-4")
        assert bench is not None
        assert bench["model_id"] == "microsoft/Phi-4"

        # 2. compare_models
        comp = api.compare_models("microsoft/Phi-4", "meta-llama/Llama-3.1-8B-Instruct")
        assert comp is not None
        assert "composite_delta" in comp
        assert "deltas" in comp

        # 3. get_pareto_frontier
        frontier = api.get_pareto_frontier(metric="hf_v2_avg", max_vram_gb=24.0, quant="q4_k_m")
        assert isinstance(frontier, list)
        assert len(frontier) > 0

        # 4. audit_model_contamination
        audit = api.audit_model_contamination("overfit-ai/Goodhart-Overfit-7B")
        assert audit["is_high_risk"] is True
        assert audit["anomaly_score"] >= 80.0

    def test_mcp_tools_json_serialization(self):
        # 1. benchmarks_model
        res_m = json.loads(benchmarks_model("microsoft/Phi-4"))
        assert res_m["model_id"] == "microsoft/Phi-4"
        assert "contamination_audit" in res_m

        # 2. benchmarks_compare
        res_c = json.loads(benchmarks_compare("microsoft/Phi-4", "meta-llama/Llama-3.1-8B-Instruct"))
        assert "composite_delta" in res_c

        # 3. benchmarks_pareto
        res_p = json.loads(benchmarks_pareto(metric="hf_v2_avg", max_vram_gb=24.0, quant="q4_k_m"))
        assert isinstance(res_p, list)
        assert len(res_p) > 0
