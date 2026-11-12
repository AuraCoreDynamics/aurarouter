"""Tests for sovereignty local qualification and route execution mode configurability."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.sovereignty.gate import (
    SovereigntyGate,
    SovereigntyVerdict,
    SovereigntyViolationError,
)
from aurarouter.routing_stack.filters import SovereigntyFilter
from aurarouter.routing_stack.context import CandidateEvalContext
from aurarouter.mcp_tools import route_task


def _create_config(tmp_path: Path) -> ConfigLoader:
    cfg_data = {
        "system": {
            "log_level": "INFO",
            "sovereignty_enforcement": True,
            "execution_mode": "standard",
        },
        "models": {
            "implicit_local": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "llama3",
                "tags": ["local"],
            },
            "implicit_cloud": {
                "provider": "anthropic",
                "endpoint": "https://api.anthropic.com",
                "model_name": "claude-3-opus",
                "hosting_tier": "cloud",
            },
            "restricted_local": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "tinyllama",
                "tags": ["local"],
                "allowed_data_categories": ["GENERAL"],
            },
        },
        "roles": {
            "general": ["implicit_local", "implicit_cloud"],
            "high_assurance": {
                "chain": ["implicit_local"],
                "execution_mode": "speculative",
            },
        },
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(cfg_data), encoding="utf-8")
    return ConfigLoader(config_path=str(p))


def test_implicit_local_qualification(tmp_path):
    """Local model with no explicit allowed_data_categories should default to wildcard '*'."""
    config = _create_config(tmp_path)

    # Local model gets wildcard '*'
    local_cats = config.get_model_allowed_data_categories("implicit_local")
    assert local_cats == ["*"]

    # Cloud model defaults to GENERAL
    cloud_cats = config.get_model_allowed_data_categories("implicit_cloud")
    assert cloud_cats == ["GENERAL"]

    # Restricted local model honors explicit configuration
    restricted_cats = config.get_model_allowed_data_categories("restricted_local")
    assert restricted_cats == ["GENERAL"]


def test_sovereignty_gate_with_implicit_local(tmp_path):
    """Implicit local model should pass gate enforcement for sovereign prompts without manual config."""
    config = _create_config(tmp_path)
    gate = SovereigntyGate(config)

    # Sensitive prompt with detected categories
    prompt = "Employee SSN: 000-12-3456 confidential medical report"
    res = gate.evaluate(prompt, effective_categories=["SSN", "Medical Record"])
    assert res.verdict == SovereigntyVerdict.SOVEREIGN
    assert len(res.matched_patterns) > 0

    # Enforce against implicit local -> should succeed without error and keep model
    passed = gate.enforce(["implicit_local"], config, res)
    assert passed == ["implicit_local"]

    # Enforce against cloud model -> should raise SovereigntyViolationError
    with pytest.raises(SovereigntyViolationError):
        gate.enforce(["implicit_cloud"], config, res)

    # Enforce against restricted local -> should raise SovereigntyViolationError
    with pytest.raises(SovereigntyViolationError):
        gate.enforce(["restricted_local"], config, res)


def test_sovereignty_routing_stack_filter(tmp_path):
    """SovereigntyFilter should evaluate implicit local model as passed and cloud as failed."""
    config = _create_config(tmp_path)
    filt = SovereigntyFilter(config)

    ctx = CandidateEvalContext(
        role="general",
        intent="coding",
        complexity_score=5,
        effective_data_categories={"Email Address", "PII"},
    )

    # Implicit local passes
    local_cfg = config.get_model_config("implicit_local")
    v_local = filt.evaluate("implicit_local", local_cfg, ctx)
    assert v_local.passed is True

    # Implicit cloud fails
    cloud_cfg = config.get_model_config("implicit_cloud")
    v_cloud = filt.evaluate("implicit_cloud", cloud_cfg, ctx)
    assert v_cloud.passed is False
    assert "unauthorized_data_categories" in v_cloud.reason

    # Restricted local fails
    rest_cfg = config.get_model_config("restricted_local")
    v_rest = filt.evaluate("restricted_local", rest_cfg, ctx)
    assert v_rest.passed is False


def test_route_execution_mode_config(tmp_path):
    """Role-level and default execution mode configurability."""
    config = _create_config(tmp_path)

    assert config.get_default_execution_mode() == "standard"
    assert config.get_role_execution_mode("high_assurance") == "speculative"
    assert config.get_role_execution_mode("general") is None


def test_route_task_execution_mode_overrides():
    """Verify query-time execution_mode overrides via parameter and options."""
    mock_fabric = MagicMock()

    async def mock_spec(*args, **kwargs):
        return "speculative_result"

    async def mock_mono(*args, **kwargs):
        return "monologue_result"

    mock_fabric.execute_speculative.side_effect = mock_spec
    mock_fabric.execute_monologue.side_effect = mock_mono
    mock_fabric.get_max_review_iterations.return_value = 0

    res1 = route_task(mock_fabric, None, task="test", execution_mode="speculative")
    assert res1 == "speculative_result"
    assert mock_fabric.execute_speculative.called

    res2 = route_task(mock_fabric, None, task="test", options={"execution_mode": "monologue"})
    assert res2 == "monologue_result"
    assert mock_fabric.execute_monologue.called
