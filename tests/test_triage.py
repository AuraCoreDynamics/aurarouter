import json
from unittest.mock import patch

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import TriageResult, analyze_intent
from aurarouter.savings.triage import TriageRouter, TriageRule


# ---------------------------------------------------------------------------
# TriageRouter.select_role tests
# ---------------------------------------------------------------------------

def _make_router() -> TriageRouter:
    """Router with two rules: <=3 → coding_lite, <=7 → coding."""
    return TriageRouter(
        rules=[
            TriageRule(max_complexity=3, preferred_role="coding_lite", description="Simple"),
            TriageRule(max_complexity=7, preferred_role="coding", description="Medium"),
        ],
        default_role="coding",
    )


def test_select_role_simple():
    router = _make_router()
    assert router.select_role(2) == "coding_lite"


def test_select_role_medium():
    router = _make_router()
    assert router.select_role(5) == "coding"


def test_select_role_complex():
    router = _make_router()
    assert router.select_role(9) == "coding"  # falls through to default


def test_no_rules_returns_default():
    router = TriageRouter(rules=[], default_role="coding")
    assert router.select_role(1) == "coding"
    assert router.select_role(10) == "coding"


# ---------------------------------------------------------------------------
# TriageRouter.from_config tests
# ---------------------------------------------------------------------------

def test_from_config():
    config = {
        "enabled": True,
        "rules": [
            {"max_complexity": 3, "preferred_role": "coding_lite", "description": "Simple"},
            {"max_complexity": 7, "preferred_role": "coding", "description": "Medium"},
        ],
        "default_role": "coding_heavy",
    }
    router = TriageRouter.from_config(config)

    assert len(router.rules) == 2
    assert router.rules[0].max_complexity == 3
    assert router.rules[0].preferred_role == "coding_lite"
    assert router.rules[1].max_complexity == 7
    assert router.rules[1].preferred_role == "coding"
    assert router.default_role == "coding_heavy"

    # Verify routing behaviour from parsed config
    assert router.select_role(2) == "coding_lite"
    assert router.select_role(5) == "coding"
    assert router.select_role(9) == "coding_heavy"


# ---------------------------------------------------------------------------
# analyze_intent with complexity scoring
# ---------------------------------------------------------------------------

def _make_fabric() -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": {"router": ["m1"]},
    }
    return ComputeFabric(cfg)


def test_analyze_intent_with_complexity(monkeypatch):
    fabric = _make_fabric()
    monkeypatch.setattr(
        fabric,
        "execute",
        lambda *a, **kw: json.dumps({"intent": "SIMPLE_CODE", "complexity": 2}),
    )
    result = analyze_intent(fabric, "add two numbers")
    assert isinstance(result, TriageResult)
    assert result.intent == "SIMPLE_CODE"
    assert result.complexity == 2


def test_analyze_intent_backwards_compat(monkeypatch):
    fabric = _make_fabric()
    monkeypatch.setattr(
        fabric,
        "execute",
        lambda *a, **kw: json.dumps({"intent": "SIMPLE_CODE"}),
    )
    result = analyze_intent(fabric, "add two numbers")
    assert isinstance(result, TriageResult)
    assert result.intent == "SIMPLE_CODE"
    assert result.complexity == 5  # default when key is absent
