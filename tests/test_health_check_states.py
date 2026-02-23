"""Tests that health checks return correct status for each service state."""

from __future__ import annotations

import pytest

# PySide6 may not be installed in CI/test environments
pyside6 = pytest.importorskip("PySide6", reason="PySide6 not installed")

from aurarouter.config import ConfigLoader  # noqa: E402
from aurarouter.gui.env_local import LocalEnvironmentContext  # noqa: E402
from aurarouter.gui.environment import ServiceState  # noqa: E402


@pytest.fixture
def context(tmp_path):
    """Create a LocalEnvironmentContext with a minimal config."""
    config_file = tmp_path / "auraconfig.yaml"
    config_file.write_text(
        "models:\n"
        "  test_model:\n"
        "    provider: ollama\n"
        "    endpoint: http://localhost:11434/api/generate\n"
        "    model_name: qwen\n"
        "roles:\n"
        "  coding: [test_model]\n"
    )
    config = ConfigLoader(config_path=str(config_file))
    return LocalEnvironmentContext(config=config)


class TestHealthCheckStates:
    def test_stopped_returns_not_healthy(self, context):
        assert context.get_state() == ServiceState.STOPPED
        status = context.check_health()
        assert not status.healthy
        assert "stopped" in status.message.lower()
        assert status.details == {}

    def test_error_returns_not_healthy(self, context):
        context._state = ServiceState.ERROR
        status = context.check_health()
        assert not status.healthy
        assert "error" in status.message.lower()
        assert status.details == {}

    def test_starting_returns_not_healthy(self, context):
        context._state = ServiceState.STARTING
        status = context.check_health()
        assert not status.healthy
        assert "starting" in status.message.lower()
        assert status.details == {}

    def test_stopping_returns_not_healthy(self, context):
        context._state = ServiceState.STOPPING
        status = context.check_health()
        assert not status.healthy
        assert "stopping" in status.message.lower()
        assert status.details == {}

    def test_paused_returns_not_healthy(self, context):
        context._state = ServiceState.PAUSED
        status = context.check_health()
        assert not status.healthy
        assert "paused" in status.message.lower()
        assert status.details == {}

    def test_pausing_returns_not_healthy(self, context):
        context._state = ServiceState.PAUSING
        status = context.check_health()
        assert not status.healthy
        assert "pausing" in status.message.lower()
        assert status.details == {}

    def test_running_without_process_returns_not_healthy(self, context):
        context._state = ServiceState.RUNNING
        context._process = None
        status = context.check_health()
        assert not status.healthy
        assert "not running" in status.message.lower()
