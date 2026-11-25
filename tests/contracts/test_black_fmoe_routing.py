"""Unit tests for expanded AuraCode DevSecOps/BLACK contract and MCP routing policy tool."""

from __future__ import annotations

import json

import pytest

from aurarouter.analyzer_schema import validate_analyzer_spec
from aurarouter.config import ConfigLoader
from aurarouter.contracts.auracode import (
    AURACODE_CAPABILITIES,
    AURACODE_INTENTS,
    create_auracode_analyzer_spec,
    map_black_scoring_to_routing_policy,
)
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import IntentRegistry
from aurarouter.mcp_tools import recommend_for_taskgroup
from aurarouter.server import _MCP_TOOL_DEFAULTS, create_mcp_server


def _make_config() -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "local_qwen": {
                "provider": "ollama",
                "model_name": "qwen2.5-coder:7b",
                "endpoint": "http://127.0.0.1:11434/api/generate",
                "tags": ["local", "coding", "reasoning"],
            },
            "cloud_claude": {
                "provider": "openapi",
                "model_name": "claude-sonnet-4",
                "endpoint": "https://api.anthropic.com/v1",
                "hosting_tier": "cloud",
                "tags": ["coding", "reasoning"],
            },
        },
        "roles": {
            "router": ["local_qwen"],
            "coding": ["local_qwen", "cloud_claude"],
            "reasoning": ["cloud_claude", "local_qwen"],
        },
    }
    return cfg


class TestExpandedAuraCodeVocabulary:
    """Verify expanded DevSecOps & BLACK intents and capabilities preserve backwards compat."""

    def test_legacy_intents_preserved_exact_roles(self):
        expected_legacy = {
            "generate_code": "coding",
            "edit_code": "coding",
            "complete_code": "coding",
            "explain_code": "reasoning",
            "review": "reasoning",
            "chat": "reasoning",
            "plan": "reasoning",
        }
        for key, role in expected_legacy.items():
            assert AURACODE_INTENTS[key] == role

    def test_devsecops_and_black_intents_present(self):
        expected_new = {
            "refactor": "coding",
            "review_diff": "reasoning",
            "security_review": "reasoning",
            "generate_tests": "coding",
            "cross_file_edit": "coding",
            "architecture_trace": "reasoning",
            "ops_triage": "reasoning",
            "remediate": "coding",
        }
        for key, role in expected_new.items():
            assert AURACODE_INTENTS[key] == role

    def test_expanded_capabilities_present_in_order(self):
        assert AURACODE_CAPABILITIES[:4] == ["code", "reasoning", "review", "planning"]
        for cap in (
            "diff_aware",
            "security_analysis",
            "test_aware",
            "cross_file",
            "ops_diagnostics",
        ):
            assert cap in AURACODE_CAPABILITIES

    def test_spec_validates_and_registers_all_expanded_intents(self):
        spec = create_auracode_analyzer_spec()
        validation = validate_analyzer_spec(spec, available_roles=["coding", "reasoning"])
        assert validation.valid
        assert not validation.errors
        assert not validation.warnings

        registry = IntentRegistry()
        registry.register_from_role_bindings("auracode", AURACODE_INTENTS)
        for intent_name, role in AURACODE_INTENTS.items():
            assert registry.resolve_role(intent_name) == role


class TestMapBlackScoringToRoutingPolicy:
    """Verify deterministic FMoE policy derivation across complexity, profiles, and sovereignty."""

    def test_low_complexity_default_coding(self):
        policy = map_black_scoring_to_routing_policy(
            {
                "tg_id": "TG1",
                "total_score": 2,
                "complexity_tier": "LOW",
                "boundaries_crossed": 0,
                "profile": "default",
                "sovereignty": "auto",
            }
        )
        assert policy["tg_id"] == "TG1"
        assert policy["complexity_score"] == 2
        assert policy["complexity_tier"] == "LOW"
        assert policy["recommended_intent"] == "generate_code"
        assert policy["recommended_role"] == "coding"
        assert "code" in policy["required_capabilities"]
        assert policy["hard_routed"] is False
        assert policy["local_only"] is False
        assert policy["max_remediation_attempts"] == 3

    def test_high_complexity_cross_file_escalates_role_to_reasoning(self):
        policy = map_black_scoring_to_routing_policy(
            {
                "tg_id": "TG2",
                "complexity_score": 8,
                "complexity_tier": "HIGH",
                "boundaries_crossed": ["aurarouter", "auracode", "auraxlm"],
                "profile": "domain-builder",
            }
        )
        assert policy["complexity_score"] == 8
        assert policy["complexity_tier"] == "HIGH"
        assert policy["boundaries_crossed"] == 3
        assert policy["recommended_intent"] == "cross_file_edit"
        assert policy["recommended_role"] == "reasoning"
        assert "code" in policy["required_capabilities"]
        assert "reasoning" in policy["required_capabilities"]
        assert "cross_file" in policy["required_capabilities"]
        assert policy["max_remediation_attempts"] == 3

    def test_critical_security_profile_and_faster_escalation(self):
        policy = map_black_scoring_to_routing_policy(
            {
                "tg_id": "TG3",
                "complexity_tier": "CRITICAL",
                "profile": "security",
                "boundaries_crossed": 2,
                "security_local_only": True,
            }
        )
        assert policy["complexity_score"] == 10
        assert policy["complexity_tier"] == "CRITICAL"
        assert policy["recommended_intent"] == "security_review"
        assert policy["recommended_role"] == "reasoning"
        assert "security_analysis" in policy["required_capabilities"]
        assert "review" in policy["required_capabilities"]
        assert "cross_file" in policy["required_capabilities"]
        assert policy["hard_routed"] is True
        assert policy["local_only"] is True
        assert policy["max_remediation_attempts"] == 2

    def test_architecture_profile_low_score_still_uses_reasoning(self):
        policy = map_black_scoring_to_routing_policy(
            {
                "tg_id": "TG4",
                "complexity_score": 4,
                "profile": "architecture",
            }
        )
        assert policy["complexity_score"] == 4
        assert policy["complexity_tier"] == "MEDIUM"
        assert policy["recommended_intent"] in ("architecture_trace", "plan")
        assert policy["recommended_intent"] == "architecture_trace"
        assert policy["recommended_role"] == "reasoning"
        assert "planning" in policy["required_capabilities"]

    @pytest.mark.parametrize("sov_mode", ["local_only", "strict"])
    def test_sovereignty_modes_enforce_local_only(self, sov_mode: str):
        policy = map_black_scoring_to_routing_policy(
            {
                "tg_id": "TG-SOV",
                "complexity_score": 5,
                "sovereignty": sov_mode,
            }
        )
        assert policy["hard_routed"] is True
        assert policy["local_only"] is True
        assert policy["sovereignty"] == sov_mode

    def test_overall_scoring_document_with_tg_lookup_and_determinism(self):
        scoring_doc = {
            "taskgroup_id": "20261124-w2c",
            "overall_complexity": "high",
            "sovereignty": "strict",
            "per_task_group": [
                {
                    "group_id": "TG1",
                    "complexity": "low",
                    "recommended_profile": "domain-builder",
                },
                {
                    "group_id": "TG2",
                    "complexity": "critical",
                    "recommended_profile": "contract-architect",
                    "boundaries_crossed": 2,
                },
            ],
        }
        first = map_black_scoring_to_routing_policy(scoring_doc, tg_id="TG2")
        second = map_black_scoring_to_routing_policy(scoring_doc, tg_id="TG2")
        assert first == second
        assert first["tg_id"] == "TG2"
        assert first["complexity_tier"] == "CRITICAL"
        assert first["recommended_intent"] == "architecture_trace"
        assert first["recommended_role"] == "reasoning"
        assert first["local_only"] is True
        assert first["max_remediation_attempts"] == 2


class TestRecommendForTaskgroupMcpTool:
    """Verify standalone MCP tool implementation and FastMCP server registration."""

    def test_recommend_for_taskgroup_accepts_dict_and_json_string(self):
        payload = {
            "tg_id": "TG5",
            "total_score": 16,
            "complexity_tier": "HIGH",
            "boundaries_crossed": 2,
            "sovereignty": "local_only",
        }
        res_from_dict = json.loads(recommend_for_taskgroup(payload))
        res_from_json = json.loads(recommend_for_taskgroup(json.dumps(payload)))
        assert res_from_dict == res_from_json
        assert res_from_dict["tg_id"] == "TG5"
        assert res_from_dict["recommended_intent"] == "cross_file_edit"
        assert res_from_dict["recommended_role"] == "reasoning"
        assert res_from_dict["local_only"] is True

    def test_recommend_for_taskgroup_with_fabric_filters_local_candidates(self):
        cfg = _make_config()
        fabric = ComputeFabric(cfg)

        sovereign_res = json.loads(
            recommend_for_taskgroup(
                {"tg_id": "TG-LOCAL", "complexity_tier": "HIGH", "sovereignty": "strict"},
                fabric=fabric,
            )
        )
        assert sovereign_res["recommended_role"] == "reasoning"
        assert sovereign_res["local_only"] is True
        assert sovereign_res["candidate_models"] == ["local_qwen"]

        standard_res = json.loads(
            recommend_for_taskgroup(
                {"tg_id": "TG-CLOUD", "complexity_tier": "HIGH", "sovereignty": "auto"},
                fabric=fabric,
            )
        )
        assert standard_res["local_only"] is False
        assert standard_res["candidate_models"] == ["cloud_claude", "local_qwen"]

    def test_recommend_for_taskgroup_invalid_json_returns_structured_error(self):
        res = json.loads(recommend_for_taskgroup("{not valid json"))
        assert "error" in res

    def test_mcp_server_registers_recommend_for_taskgroup_by_default(self):
        assert _MCP_TOOL_DEFAULTS.get("aurarouter.routing.recommend_for_taskgroup") is True
        cfg = _make_config()
        mcp = create_mcp_server(cfg)
        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        assert "aurarouter.routing.recommend_for_taskgroup" in tools

        tool = tools["aurarouter.routing.recommend_for_taskgroup"]
        raw = tool.fn(
            scoring_entry=json.dumps(
                {
                    "tg_id": "TG9",
                    "complexity_tier": "CRITICAL",
                    "profile": "security",
                    "sovereignty": "strict",
                }
            )
        )
        parsed = json.loads(raw)
        assert parsed["tg_id"] == "TG9"
        assert parsed["recommended_intent"] == "security_review"
        assert parsed["recommended_role"] == "reasoning"
        assert parsed["hard_routed"] is True
        assert parsed["max_remediation_attempts"] == 2
        assert parsed["candidate_models"] == ["local_qwen"]

    def test_mcp_server_respects_config_disable_flag(self):
        cfg = _make_config()
        cfg.config["mcp"] = {
            "tools": {
                "aurarouter.routing.recommend_for_taskgroup": {"enabled": False},
            }
        }
        mcp = create_mcp_server(cfg)
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "aurarouter.routing.recommend_for_taskgroup" not in tool_names
