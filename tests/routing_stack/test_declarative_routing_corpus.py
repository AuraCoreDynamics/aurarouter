"""Declarative Routing Evaluation Corpus Test Harness.

Executes and verifies end-to-end prompt intent triage and stacked routing
matrices declared in the external artifact `tests/fixtures/routing_evaluation_corpus.json`.

Tests:
1. Corpus schema integrity and cross-platform artifact exportability (for auraxlm).
2. Combinatorial intent analysis and stacked routing dispatch across all declared scenarios.
3. Multi-candidate competition verifying highest-scoring model with highest limits wins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.testing import (
    CORPUS_FIXTURE_PATH,
    execute_scenario_intent_and_routing,
    get_routing_scenario,
    list_routing_scenario_ids,
    load_routing_evaluation_corpus,
    setup_corpus_environment,
)


# ==============================================================================
# 1. Corpus Schema & Artifact Integrity
# ==============================================================================


class TestDeclarativeRoutingCorpusIntegrity:
    """Verifies that the external JSON test artifact adheres to schema standards."""

    def test_corpus_file_exists_and_loads(self):
        """External JSON corpus artifact must exist and load cleanly."""
        assert CORPUS_FIXTURE_PATH.exists(), f"Corpus file missing at {CORPUS_FIXTURE_PATH}"
        corpus = load_routing_evaluation_corpus()
        assert isinstance(corpus, dict)
        assert corpus.get("schema_version") == "1.0.0"
        assert "scenarios" in corpus
        assert len(corpus["scenarios"]) >= 12

    def test_unique_scenario_ids(self):
        """All scenario IDs must be unique non-empty strings."""
        scenario_ids = list_routing_scenario_ids()
        assert len(scenario_ids) == len(set(scenario_ids)), "Duplicate scenario IDs detected!"
        for sid in scenario_ids:
            assert isinstance(sid, str) and len(sid) > 0

    def test_scenario_contract_structure(self):
        """Every scenario must define all required contract fields."""
        corpus = load_routing_evaluation_corpus()
        required_top_level = {
            "id",
            "name",
            "description",
            "intent_evaluation",
            "stack_configuration",
            "runtime_state",
            "execution_options",
            "expected_outcome",
        }
        required_intent_fields = {
            "raw_prompt",
            "simulated_router_response",
            "expected_intent",
            "expected_role",
            "complexity",
        }
        required_outcome_fields = {
            "winning_model",
            "ranked_chain",
            "pruned_candidates",
            "dominant_factor",
        }

        for scenario in corpus["scenarios"]:
            sid = scenario.get("id", "<unknown>")
            missing_top = required_top_level - set(scenario.keys())
            assert not missing_top, f"Scenario '{sid}' missing fields: {missing_top}"

            intent_eval = scenario["intent_evaluation"]
            missing_intent = required_intent_fields - set(intent_eval.keys())
            assert not missing_intent, f"Scenario '{sid}' missing intent fields: {missing_intent}"

            outcome = scenario["expected_outcome"]
            missing_outcome = required_outcome_fields - set(outcome.keys())
            assert not missing_outcome, f"Scenario '{sid}' missing outcome fields: {missing_outcome}"

            # Verify router response parses as valid JSON
            parsed_router = json.loads(intent_eval["simulated_router_response"])
            assert "intent" in parsed_router
            assert "complexity" in parsed_router

    def test_cross_scenario_corpus_exportability(self):
        """Corpus must serialize to clean standard JSON without Python-specific artifacts."""
        corpus = load_routing_evaluation_corpus()
        dumped = json.dumps(corpus, indent=2)
        reloaded = json.loads(dumped)
        assert reloaded == corpus
        assert "auraxlm-dotnet" in corpus.get("metadata", {}).get("compatible_engines", [])


# ==============================================================================
# 2. End-to-End Declarative Execution Matrix
# ==============================================================================


class TestDeclarativeRoutingMatrix:
    """Parametrized execution harness evaluating every declared scenario."""

    @pytest.mark.parametrize("scenario_id", list_routing_scenario_ids())
    def test_declarative_scenario_execution(self, scenario_id: str, tmp_path: Path):
        """Execute a declared scenario through live Intent Analysis and Routing Stack dispatch."""
        scenario = get_routing_scenario(scenario_id)
        env = setup_corpus_environment(scenario, tmp_path)

        # Run intent analysis + explain_route + live execute
        execution_results = execute_scenario_intent_and_routing(env, scenario)

        triage = execution_results["triage"]
        target_role = execution_results["target_role"]
        trace = execution_results["trace"]
        result = execution_results["result"]

        exp_intent = scenario["intent_evaluation"]["expected_intent"]
        exp_role = scenario["intent_evaluation"]["expected_role"]
        exp_winner = scenario["expected_outcome"]["winning_model"]
        exp_pruned = scenario["expected_outcome"]["pruned_candidates"]

        # 1. Assert Intent Analysis & Role Resolution
        assert triage.intent == exp_intent, (
            f"[{scenario_id}] Intent mismatch: got '{triage.intent}', expected '{exp_intent}'"
        )
        assert target_role == exp_role, (
            f"[{scenario_id}] Role mismatch: got '{target_role}', expected '{exp_role}'"
        )

        # 2. Assert Stage 1 Pruned Candidates
        for pruned_mid, expected_filter in exp_pruned.items():
            assert pruned_mid in trace.pruned, (
                f"[{scenario_id}] Model '{pruned_mid}' was not pruned! Active pruned: {trace.pruned}"
            )
            assert expected_filter in trace.pruned[pruned_mid], (
                f"[{scenario_id}] Model '{pruned_mid}' pruned by '{trace.pruned[pruned_mid]}', "
                f"expected filter: '{expected_filter}'"
            )

        # 3. Assert Winning Model from Stack Ranking & Live Execution
        assert len(trace.ranked_chain) > 0, f"[{scenario_id}] No eligible models survived Stage 1 gates!"
        assert trace.ranked_chain[0] == exp_winner, (
            f"[{scenario_id}] Stack ranked winner '{trace.ranked_chain[0]}' != expected '{exp_winner}'. "
            f"Ranked chain: {trace.ranked_chain}"
        )
        assert result.model_id == exp_winner, (
            f"[{scenario_id}] Live executed model '{result.model_id}' != expected '{exp_winner}'"
        )
        assert result.text, f"[{scenario_id}] Model returned empty execution result text"


# ==============================================================================
# 3. Specific Multi-Model / Highest-Limits Rule Verification
# ==============================================================================


class TestMultiCandidateCompetitionRules:
    """In-depth verification of scenarios with multi-model limit and scoring competition."""

    def test_highest_scoring_model_with_highest_limits_wins(self, tmp_path: Path):
        """Scenario 12: 4 models satisfy intent/role; model with highest quality + limits wins."""
        scenario = get_routing_scenario(
            "scenario_12_stacked_multi_candidate_highest_scoring_with_highest_limits_wins"
        )
        env = setup_corpus_environment(scenario, tmp_path)
        res = execute_scenario_intent_and_routing(env, scenario)

        trace = res["trace"]
        scores_by_model = {s.model_id: s for s in trace.scores}

        champion_id = "model-gamma-champion-highest-quality-highest-headroom"
        alpha_id = "model-alpha-good-quality-low-headroom"
        beta_id = "model-beta-low-quality-high-headroom"
        delta_id = "model-delta-high-quality-low-headroom"

        assert champion_id in scores_by_model
        assert alpha_id in scores_by_model
        assert beta_id in scores_by_model
        assert delta_id in scores_by_model

        champion_score = scores_by_model[champion_id]
        alpha_score = scores_by_model[alpha_id]
        delta_score = scores_by_model[delta_id]

        # Champion has highest headroom (1.0) and high quality prior
        assert champion_score.headroom_mult >= 0.95
        # Alpha and Delta suffered severe rate limit degradation (low limits remaining)
        assert alpha_score.headroom_mult < 0.96
        assert delta_score.headroom_mult < 0.88

        # Champion dominates in total utility
        assert champion_score.utility > alpha_score.utility
        assert champion_score.utility > delta_score.utility
        assert res["result"].model_id == champion_id

    def test_intent_to_role_dynamic_analyzer_binding(self, tmp_path: Path):
        """Scenario 10: Verifies custom intent analyzer declarations route to specialist roles."""
        scenario = get_routing_scenario(
            "scenario_10_custom_analyzer_intent_registry_dispatch"
        )
        env = setup_corpus_environment(scenario, tmp_path)
        res = execute_scenario_intent_and_routing(env, scenario)

        assert res["triage"].intent == "sar_radar_processing"
        assert res["target_role"] == "earth_observation"
        assert res["result"].model_id == "eo-sar-specialist"

