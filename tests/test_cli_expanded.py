"""Tests for the expanded CLI subcommands.

All commands go through AuraRouterAPI, which is mocked to avoid real
config loading and provider calls.
"""

import json
import sys
from dataclasses import dataclass, field
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.cli import main


# ======================================================================
# Helpers
# ======================================================================

def _run_cli(*args: str) -> str:
    """Invoke main() with the given CLI arguments and capture stdout."""
    buf = StringIO()
    with patch.object(sys, "argv", ["aurarouter", *args]), \
         patch("sys.stdout", buf):
        try:
            main()
        except SystemExit:
            pass
    return buf.getvalue()


def _mock_api():
    """Return a pre-configured MagicMock mimicking AuraRouterAPI."""
    api = MagicMock()
    api.__enter__ = MagicMock(return_value=api)
    api.__exit__ = MagicMock(return_value=False)
    return api


@dataclass
class _ModelInfo:
    model_id: str = "test-model"
    provider: str = "ollama"
    config: dict = field(default_factory=lambda: {"provider": "ollama", "tags": ["local"], "hosting_tier": "local"})


@dataclass
class _RoleChain:
    role: str = "coding"
    chain: list = field(default_factory=lambda: ["model-a", "model-b"])


@dataclass
class _TrafficSummary:
    total_tokens: int = 5000
    input_tokens: int = 3000
    output_tokens: int = 2000
    by_model: list = field(default_factory=list)
    total_spend: float = 1.23
    spend_by_provider: dict = field(default_factory=lambda: {"ollama": 0.0, "gemini": 1.23})
    projection: dict = field(default_factory=dict)


@dataclass
class _HealthReport:
    model_id: str = "test-model"
    healthy: bool = True
    message: str = "OK"
    latency: float = 0.15


@dataclass
class _CatalogEntry:
    name: str = "ollama"
    provider_type: str = "ollama"
    source: str = "builtin"
    installed: bool = True
    running: bool = True
    version: str = "0.3.0"
    description: str = "Local Ollama server"


@dataclass
class _TaskResult:
    output: str = "Hello World"
    intent: str = "SIMPLE_CODE"
    complexity: int = 3
    plan: list = field(default_factory=lambda: ["Generate code"])
    steps_executed: int = 1
    review_verdict: str = "PASS"
    review_feedback: str = "Looks good"
    total_elapsed: float = 2.5


@dataclass
class _GenerateResult:
    text: str = "result text"
    model_id: str = "model-a"
    provider: str = "ollama"


# ======================================================================
# Tests: model commands
# ======================================================================

class TestModelList:
    """Test `aurarouter model list`."""

    def test_human_readable(self):
        api = _mock_api()
        api.list_models.return_value = [_ModelInfo(), _ModelInfo(model_id="model-b", provider="gemini")]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("model", "list")
        assert "test-model" in output
        assert "model-b" in output
        assert "2 model(s)" in output

    def test_json_output(self):
        api = _mock_api()
        api.list_models.return_value = [_ModelInfo()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("model", "list", "--json")
        data = json.loads(output)
        assert isinstance(data, list)
        assert data[0]["model_id"] == "test-model"

    def test_filter(self):
        api = _mock_api()
        api.list_models.return_value = [
            _ModelInfo(model_id="ollama-qwen", provider="ollama"),
            _ModelInfo(model_id="gemini-pro", provider="gemini"),
        ]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("model", "list", "--filter", "gemini")
        assert "gemini-pro" in output
        assert "ollama-qwen" not in output

    def test_empty(self):
        api = _mock_api()
        api.list_models.return_value = []
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("model", "list")
        assert "No models configured" in output


# ======================================================================
# Tests: route commands
# ======================================================================

class TestRouteList:
    """Test `aurarouter route list`."""

    def test_human_readable(self):
        api = _mock_api()
        api.list_roles.return_value = [_RoleChain()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("route", "list")
        assert "coding" in output
        assert "model-a" in output

    def test_json_output(self):
        api = _mock_api()
        api.list_roles.return_value = [_RoleChain()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("route", "list", "--json")
        data = json.loads(output)
        assert data[0]["role"] == "coding"
        assert data[0]["chain"] == ["model-a", "model-b"]


# ======================================================================
# Tests: run command (mocked execution)
# ======================================================================

class TestRun:
    """Test `aurarouter run`."""

    def test_basic_run(self):
        api = _mock_api()
        api.execute_task.return_value = _TaskResult()
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("run", "Write hello world")
        assert "Hello World" in output
        api.execute_task.assert_called_once()

    def test_run_json(self):
        api = _mock_api()
        api.execute_task.return_value = _TaskResult()
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("run", "Write hello", "--json")
        data = json.loads(output)
        assert data["output"] == "Hello World"
        assert data["review_verdict"] == "PASS"

    def test_run_no_review(self):
        api = _mock_api()
        api.execute_direct.return_value = _GenerateResult()
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("run", "Quick task", "--no-review")
        assert "result text" in output
        api.execute_direct.assert_called_once()


# ======================================================================
# Tests: monitoring commands
# ======================================================================

class TestTraffic:
    """Test `aurarouter traffic`."""

    def test_human_readable(self):
        api = _mock_api()
        api.get_traffic.return_value = _TrafficSummary()
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("traffic")
        assert "5,000" in output
        assert "$1.23" in output

    def test_json_output(self):
        api = _mock_api()
        api.get_traffic.return_value = _TrafficSummary()
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("traffic", "--json")
        data = json.loads(output)
        assert data["total_tokens"] == 5000


class TestHealth:
    """Test `aurarouter health`."""

    def test_all_models(self):
        api = _mock_api()
        api.check_health.return_value = [_HealthReport()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("health")
        assert "[OK]" in output
        assert "test-model" in output

    def test_specific_model(self):
        api = _mock_api()
        api.check_health.return_value = [_HealthReport()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("health", "test-model")
        api.check_health.assert_called_once_with(model_id="test-model")

    def test_json_output(self):
        api = _mock_api()
        api.check_health.return_value = [_HealthReport()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("health", "--json")
        data = json.loads(output)
        assert data[0]["healthy"] is True


# ======================================================================
# Tests: catalog commands
# ======================================================================

class TestCatalogList:
    """Test `aurarouter catalog list`."""

    def test_human_readable(self):
        api = _mock_api()
        api.list_catalog.return_value = [_CatalogEntry()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("catalog", "list")
        assert "ollama" in output
        assert "Running" in output

    def test_json_output(self):
        api = _mock_api()
        api.list_catalog.return_value = [_CatalogEntry()]
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("catalog", "list", "--json")
        data = json.loads(output)
        assert data[0]["name"] == "ollama"
        assert data[0]["running"] is True


# ======================================================================
# Tests: --json flag consistency
# ======================================================================

class TestJsonFlag:
    """Verify --json produces valid JSON for key commands."""

    @pytest.mark.parametrize("cmd,setup", [
        (["model", "list", "--json"], lambda api: setattr(api, "list_models", MagicMock(return_value=[_ModelInfo()]))),
        (["route", "list", "--json"], lambda api: setattr(api, "list_roles", MagicMock(return_value=[_RoleChain()]))),
        (["traffic", "--json"], lambda api: setattr(api, "get_traffic", MagicMock(return_value=_TrafficSummary()))),
        (["health", "--json"], lambda api: setattr(api, "check_health", MagicMock(return_value=[_HealthReport()]))),
        (["catalog", "list", "--json"], lambda api: setattr(api, "list_catalog", MagicMock(return_value=[_CatalogEntry()]))),
        (["budget", "--json"], lambda api: setattr(api, "get_budget_status", MagicMock(return_value={"allowed": True, "daily_spend": 0.5, "monthly_spend": 2.0, "daily_limit": 5.0, "monthly_limit": 50.0, "reason": ""}))),
    ])
    def test_json_valid(self, cmd, setup):
        api = _mock_api()
        setup(api)
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli(*cmd)
        # Must parse as valid JSON
        data = json.loads(output)
        assert data is not None


# ======================================================================
# Tests: backward compatibility
# ======================================================================

class TestBackwardCompat:
    """Legacy subcommands still work."""

    def test_list_models_legacy(self):
        mock_storage = MagicMock()
        mock_storage.models_dir = "/fake/models"
        mock_storage.list_models.return_value = [
            {"filename": "test.gguf", "size_bytes": 1024 * 1024 * 100, "repo": "Qwen/test"},
        ]
        with patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            output = _run_cli("list-models")
        assert "test.gguf" in output
        assert "100 MB" in output

    def test_download_model_legacy(self):
        with patch("aurarouter.models.downloader.download_model") as mock_dl:
            _run_cli("download-model", "--repo", "Qwen/test", "--file", "model.gguf")
        mock_dl.assert_called_once_with(repo="Qwen/test", filename="model.gguf", dest=None)

    def test_remove_model_legacy(self):
        mock_storage = MagicMock()
        mock_storage.remove.return_value = True
        with patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            output = _run_cli("remove-model", "--file", "old.gguf")
        assert "Removed and deleted" in output


# ======================================================================
# Tests: config commands
# ======================================================================

class TestConfigShow:
    """Test `aurarouter config show`."""

    def test_shows_yaml(self):
        api = _mock_api()
        api.get_config_yaml.return_value = "models:\n  test: {}\n"
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("config", "show")
        assert "models:" in output

    def test_shows_json(self):
        api = _mock_api()
        api.get_config_yaml.return_value = "models:\n  test: {}\n"
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("config", "show", "--json")
        data = json.loads(output)
        assert "models" in data


# ======================================================================
# Tests: budget command
# ======================================================================

class TestBudget:
    """Test `aurarouter budget`."""

    def test_budget_enabled(self):
        api = _mock_api()
        api.get_budget_status.return_value = {
            "allowed": True, "reason": "",
            "daily_spend": 1.50, "monthly_spend": 10.00,
            "daily_limit": 5.00, "monthly_limit": 50.00,
        }
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("budget")
        assert "$1.50" in output
        assert "$5.00" in output

    def test_budget_disabled(self):
        api = _mock_api()
        api.get_budget_status.return_value = None
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("budget")
        assert "disabled" in output.lower()


# ======================================================================
# Tests: privacy command
# ======================================================================

class TestPrivacy:
    """Test `aurarouter privacy`."""

    def test_human_readable(self):
        api = _mock_api()
        api.get_privacy_events.return_value = MagicMock(
            total_events=3,
            by_severity={"high": 1, "medium": 2},
            by_pattern={"API Key": 1, "Email Address": 2},
        )
        with patch("aurarouter.cli._make_api", return_value=api):
            output = _run_cli("privacy")
        assert "3" in output
        assert "high" in output
