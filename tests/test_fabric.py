from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric


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

    with patch("aurarouter.providers.ollama.OllamaProvider.generate", return_value="hello world code"):
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
        "aurarouter.providers.ollama.OllamaProvider.generate",
        side_effect=["", "valid result here"],
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
        "aurarouter.providers.ollama.OllamaProvider.generate",
        side_effect=Exception("boom"),
    ):
        result = fabric.execute("coding", "prompt")
    assert result is None


def test_execute_unknown_role():
    fabric = _make_fabric(models={}, roles={})
    result = fabric.execute("nonexistent", "prompt")
    assert result is not None
    assert "ERROR" in result
