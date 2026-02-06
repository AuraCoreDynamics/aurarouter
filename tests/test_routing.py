import json
from unittest.mock import patch

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import analyze_intent, generate_plan


def _make_fabric() -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return ComputeFabric(cfg)


def test_analyze_intent_simple():
    fabric = _make_fabric()
    with patch.object(
        fabric, "execute", return_value=json.dumps({"intent": "SIMPLE_CODE"})
    ):
        assert analyze_intent(fabric, "add two numbers") == "SIMPLE_CODE"


def test_analyze_intent_complex():
    fabric = _make_fabric()
    with patch.object(
        fabric, "execute", return_value=json.dumps({"intent": "COMPLEX_REASONING"})
    ):
        assert analyze_intent(fabric, "design a distributed system") == "COMPLEX_REASONING"


def test_analyze_intent_defaults_on_bad_json():
    fabric = _make_fabric()
    with patch.object(fabric, "execute", return_value="not json"):
        assert analyze_intent(fabric, "anything") == "SIMPLE_CODE"


def test_generate_plan_returns_list():
    fabric = _make_fabric()
    plan_json = json.dumps(["step 1", "step 2", "step 3"])
    with patch.object(fabric, "execute", return_value=plan_json):
        result = generate_plan(fabric, "build something", "")
    assert result == ["step 1", "step 2", "step 3"]


def test_generate_plan_strips_fences():
    fabric = _make_fabric()
    raw = '```json\n["a", "b"]\n```'
    with patch.object(fabric, "execute", return_value=raw):
        assert generate_plan(fabric, "task", "") == ["a", "b"]


def test_generate_plan_fallback_on_failure():
    fabric = _make_fabric()
    with patch.object(fabric, "execute", return_value=None):
        assert generate_plan(fabric, "my task", "") == ["my task"]
