import pytest
import yaml
from pathlib import Path

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric


@pytest.fixture
def config(tmp_path: Path):
    """Create a minimal, valid config file in a temporary directory."""
    config_content = {
        "system": {"log_level": "INFO", "default_timeout": 120.0},
        "models": {
            "mock_ollama": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "mock_model",
            },
            "mock_google": {
                "provider": "google",
                "model_name": "gemini-pro",
                "api_key": "MOCK_API_KEY",
            },
        },
        "roles": {
            "router": ["mock_ollama", "mock_google"],
            "reasoning": ["mock_google"],
            "coding": ["mock_ollama"],
        },
    }
    config_path = tmp_path / "auraconfig.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    return ConfigLoader(config_path=str(config_path))


@pytest.fixture
def config_empty():
    """A ConfigLoader with no actual config (for installer/offline tests)."""
    return ConfigLoader(allow_missing=True)


@pytest.fixture
def fabric(config):
    return ComputeFabric(config)
