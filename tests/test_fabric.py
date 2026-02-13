from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult


def _make_fabric(models: dict, roles: dict) -> ComputeFabric:
    """Build a fabric from inline config dicts."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg)


def test_execute_returns_first_success():
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="hello world code"),
    ):
        result = fabric.execute("coding", "test prompt")
    assert result == "hello world code"


def test_execute_skips_empty_response():
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1", "m2"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=[
            GenerateResult(text=""),
            GenerateResult(text="valid result here"),
        ],
    ):
        result = fabric.execute("coding", "prompt")
    assert result == "valid result here"


def test_execute_returns_none_when_all_fail():
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=Exception("boom"),
    ):
        result = fabric.execute("coding", "prompt")
    assert result is None


def test_execute_unknown_role():
    fabric = _make_fabric(models={}, roles={})
    result = fabric.execute("nonexistent", "prompt")
    assert result is not None
    assert "ERROR" in result


# ------------------------------------------------------------------
# chain_override
# ------------------------------------------------------------------

def test_execute_with_chain_override():
    """chain_override should bypass the role's configured chain."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1"]},  # Only m1 in role chain
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="from m2"),
    ):
        # Override to use m2 instead
        result = fabric.execute("coding", "test", chain_override=["m2"])
    assert result == "from m2"


def test_execute_chain_override_none_uses_role():
    """chain_override=None should use the normal role chain."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="normal result"),
    ):
        result = fabric.execute("coding", "test", chain_override=None)
    assert result == "normal result"


# ------------------------------------------------------------------
# execute_all
# ------------------------------------------------------------------

def test_execute_all_collects_all():
    """execute_all should return results from every model, not just the first."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1", "m2"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=[
            GenerateResult(text="response1", input_tokens=10, output_tokens=20),
            GenerateResult(text="response2", input_tokens=15, output_tokens=25),
        ],
    ):
        results = fabric.execute_all("coding", "test prompt")

    assert len(results) == 2
    assert results[0]["model_id"] == "m1"
    assert results[0]["success"] is True
    assert results[0]["text"] == "response1"
    assert results[1]["model_id"] == "m2"
    assert results[1]["success"] is True
    assert results[1]["text"] == "response2"


def test_execute_all_with_explicit_model_ids():
    """execute_all should accept explicit model_ids."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1", "m2"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="only m2"),
    ):
        results = fabric.execute_all("coding", "test", model_ids=["m2"])

    assert len(results) == 1
    assert results[0]["model_id"] == "m2"


def test_execute_all_handles_failures():
    """execute_all should record failures without stopping."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=Exception("connection refused"),
    ):
        results = fabric.execute_all("coding", "test")

    assert len(results) == 1
    assert results[0]["success"] is False
    assert "ERROR" in results[0]["text"]
