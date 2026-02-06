import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric


@pytest.fixture
def config():
    """Load the dev config from the repo root."""
    return ConfigLoader(config_path="auraconfig.yaml")


@pytest.fixture
def config_empty():
    """A ConfigLoader with no actual config (for installer/offline tests)."""
    return ConfigLoader(allow_missing=True)


@pytest.fixture
def fabric(config):
    return ComputeFabric(config)
