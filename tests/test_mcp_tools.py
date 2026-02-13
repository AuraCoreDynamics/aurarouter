"""Tests for MCP tool implementations (mcp_tools.py)."""

import json
from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import (
    compare_models,
    generate_code,
    local_inference,
    route_task,
)
from aurarouter.savings.models import GenerateResult


def _make_fabric(models=None, roles=None) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": models or {
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            "m2": {"provider": "google", "model_name": "gemini-flash", "api_key": "k"},
        },
        "roles": roles or {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1", "m2"],
        },
    }
    return ComputeFabric(cfg)


# ------------------------------------------------------------------
# route_task
# ------------------------------------------------------------------

class TestRouteTask:
    def test_simple_intent(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE", "complexity": 3}),
            "result text",
        ]):
            result = route_task(fabric, None, task="hello world")
            assert result == "result text"

    def test_complex_intent(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "COMPLEX_REASONING", "complexity": 8}),
            json.dumps(["step 1", "step 2"]),
            "step 1 output",
            "step 2 output",
        ]):
            result = route_task(fabric, None, task="complex task")
            assert "Step 1" in result
            assert "Step 2" in result

    def test_all_models_fail(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),
            None,
        ]):
            result = route_task(fabric, None, task="test")
            assert "Error" in result


# ------------------------------------------------------------------
# local_inference
# ------------------------------------------------------------------

class TestLocalInference:
    def test_filters_to_local_only(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute") as mock:
            mock.return_value = "local result"
            result = local_inference(fabric, prompt="test")
            assert result == "local result"
            # Verify chain_override was passed with only local models
            call_kwargs = mock.call_args
            override = call_kwargs.kwargs.get("chain_override")
            assert override is not None
            assert "m1" in override      # ollama (local)
            assert "m2" not in override  # google (cloud)

    def test_error_when_no_local_models(self):
        fabric = _make_fabric(
            models={"m2": {"provider": "google", "model_name": "g", "api_key": "k"}},
            roles={"coding": ["m2"]},
        )
        result = local_inference(fabric, prompt="test")
        assert "Error" in result
        assert "local" in result.lower()

    def test_includes_context(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute") as mock:
            mock.return_value = "ok"
            local_inference(fabric, prompt="test", context="extra context")
            prompt_sent = mock.call_args.args[1]
            assert "extra context" in prompt_sent


# ------------------------------------------------------------------
# generate_code
# ------------------------------------------------------------------

class TestGenerateCode:
    def test_simple_code(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),
            "def add(a, b): return a + b",
        ]):
            result = generate_code(
                fabric, None,
                task_description="write an add function",
                language="python",
            )
            assert "def add" in result

    def test_complex_code(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "COMPLEX_REASONING"}),
            json.dumps(["create module", "add tests"]),
            "# module code",
            "# test code",
        ]):
            result = generate_code(
                fabric, None,
                task_description="build a module with tests",
            )
            assert "Step 1" in result
            assert "Step 2" in result


# ------------------------------------------------------------------
# compare_models
# ------------------------------------------------------------------

class TestCompareModels:
    def test_returns_all_results(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute_all", return_value=[
            {"model_id": "m1", "provider": "ollama", "success": True,
             "text": "result1", "elapsed_s": 1.0, "input_tokens": 10, "output_tokens": 20},
            {"model_id": "m2", "provider": "google", "success": True,
             "text": "result2", "elapsed_s": 2.0, "input_tokens": 10, "output_tokens": 30},
        ]):
            result = compare_models(fabric, prompt="test")
            assert "m1" in result
            assert "m2" in result
            assert "result1" in result
            assert "result2" in result
            assert "SUCCESS" in result

    def test_empty_results(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute_all", return_value=[]):
            result = compare_models(fabric, prompt="test")
            assert "Error" in result

    def test_passes_model_ids(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute_all") as mock:
            mock.return_value = []
            compare_models(fabric, prompt="test", models="m1, m2")
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs.get("model_ids") == ["m1", "m2"]
