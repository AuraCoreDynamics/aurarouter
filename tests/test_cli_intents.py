"""Tests for CLI intent flag and intent subcommand group (Task Group 4).

Covers:
- ``run --intent SIMPLE_CODE`` passes intent through to execute_task
- ``run --intent unknown_intent`` exits with error
- ``intent list`` shows built-in and analyzer intents
- ``intent list --json`` produces valid JSON
- ``intent describe SIMPLE_CODE`` shows intent details
- ``catalog artifacts --kind analyzer`` includes declared intents
"""

import json
import sys
from dataclasses import dataclass, field
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aurarouter.cli import main


# ======================================================================
# Helpers
# ======================================================================

def _run_cli(*args: str) -> tuple[str, str, int]:
    """Invoke main() with the given CLI arguments and capture stdout/stderr.

    Returns (stdout, stderr, exit_code).  exit_code is 0 unless SystemExit
    was raised.
    """
    out = StringIO()
    err = StringIO()
    code = 0
    with (
        patch.object(sys, "argv", ["aurarouter", *args]),
        patch("sys.stdout", out),
        patch("sys.stderr", err),
    ):
        try:
            main()
        except SystemExit as exc:
            code = exc.code if exc.code is not None else 0
    return out.getvalue(), err.getvalue(), code


def _make_config(tmp_path, extra_catalog=None):
    """Write a minimal auraconfig.yaml and return the path string."""
    cfg = {
        "system": {
            "log_level": "INFO",
            "default_timeout": 120.0,
            "active_analyzer": "aurarouter-default",
        },
        "models": {
            "mock_ollama": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "mock",
            },
        },
        "roles": {
            "router": ["mock_ollama"],
            "coding": ["mock_ollama"],
            "reasoning": ["mock_ollama"],
            "reviewer": ["mock_ollama"],
        },
        "catalog": {
            "aurarouter-default": {
                "kind": "analyzer",
                "display_name": "AuraRouter Default",
                "description": "Built-in IPE pipeline",
                "analyzer_kind": "intent_triage",
                "role_bindings": {
                    "simple_code": "coding",
                    "complex_reasoning": "reasoning",
                    "review": "reviewer",
                },
                "capabilities": ["code", "reasoning", "review", "planning"],
            },
        },
    }
    if extra_catalog:
        cfg["catalog"].update(extra_catalog)

    path = tmp_path / "auraconfig.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


# ======================================================================
# T4.1: --intent flag on run command
# ======================================================================

class TestRunIntentFlag:
    """Verify --intent / -i on the run command."""

    def test_run_passes_intent_to_execute_task(self, tmp_path):
        """run --intent SIMPLE_CODE should forward intent to execute_task."""
        config_path = _make_config(tmp_path)

        @dataclass
        class _TaskResult:
            output: str = "ok"
            intent: str = "SIMPLE_CODE"
            complexity: int = 3
            plan: list = field(default_factory=list)
            steps_executed: int = 1
            review_verdict: str = "PASS"
            review_feedback: str = ""
            total_elapsed: float = 0.1

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)
        api_mock.execute_task.return_value = _TaskResult()

        # We need a real ConfigLoader so intent validation works
        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "run", "--intent", "SIMPLE_CODE", "Write hello world",
            )

        assert code == 0
        api_mock.execute_task.assert_called_once()
        call_kwargs = api_mock.execute_task.call_args
        assert call_kwargs.kwargs.get("intent") == "SIMPLE_CODE" or \
            (call_kwargs[1].get("intent") == "SIMPLE_CODE" if len(call_kwargs) > 1 else False)

    def test_run_short_flag_works(self, tmp_path):
        """run -i DIRECT should also work."""
        config_path = _make_config(tmp_path)

        @dataclass
        class _TaskResult:
            output: str = "ok"
            intent: str = "DIRECT"
            complexity: int = 1
            plan: list = field(default_factory=list)
            steps_executed: int = 1
            review_verdict: str = "PASS"
            review_feedback: str = ""
            total_elapsed: float = 0.1

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)
        api_mock.execute_task.return_value = _TaskResult()

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "run", "-i", "DIRECT", "Tell me a joke",
            )

        assert code == 0

    def test_run_unknown_intent_exits_with_error(self, tmp_path):
        """run --intent bogus_intent should fail with error code 1."""
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "run", "--intent", "bogus_intent", "Do something",
            )

        assert code == 1
        assert "Unknown intent" in stderr
        assert "bogus_intent" in stderr
        assert "DIRECT" in stderr  # should list available intents

    def test_run_without_intent_works(self, tmp_path):
        """run without --intent should not break (backward compat)."""
        config_path = _make_config(tmp_path)

        @dataclass
        class _TaskResult:
            output: str = "hello"
            intent: str = "DIRECT"
            complexity: int = 1
            plan: list = field(default_factory=list)
            steps_executed: int = 1
            review_verdict: str = "PASS"
            review_feedback: str = ""
            total_elapsed: float = 0.05

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)
        api_mock.execute_task.return_value = _TaskResult()

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "run", "Tell me a joke",
            )

        assert code == 0
        # intent should be None (auto-classify)
        call_kwargs = api_mock.execute_task.call_args
        assert call_kwargs.kwargs.get("intent") is None


# ======================================================================
# T4.2: intent subcommand group
# ======================================================================

class TestIntentList:
    """Verify intent list output."""

    def test_intent_list_shows_builtin_and_analyzer(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)
        api_mock.get_active_analyzer.return_value = "aurarouter-default"

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "intent", "list",
            )

        assert code == 0
        # Check built-in intents appear
        assert "Built-in Intents:" in stdout
        assert "DIRECT" in stdout
        assert "SIMPLE_CODE" in stdout
        assert "COMPLEX_REASONING" in stdout
        # Check analyzer intents appear
        assert "Analyzer Intents" in stdout
        assert "simple_code" in stdout or "review" in stdout
        # Check active analyzer is shown
        assert "Active Analyzer:" in stdout

    def test_intent_list_json(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)
        api_mock.get_active_analyzer.return_value = "aurarouter-default"

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "intent", "list", "--json",
            )

        assert code == 0
        data = json.loads(stdout)
        assert "intents" in data
        assert "active_analyzer" in data
        names = [i["name"] for i in data["intents"]]
        assert "DIRECT" in names
        assert "SIMPLE_CODE" in names
        assert "COMPLEX_REASONING" in names
        # Analyzer-declared intents (from role_bindings with priority 10)
        # These may override built-in names; check at least one analyzer intent
        sources = {i["source"] for i in data["intents"]}
        assert "builtin" in sources or "aurarouter-default" in sources


class TestIntentDescribe:
    """Verify intent describe output."""

    def test_describe_builtin_intent(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "intent", "describe", "DIRECT",
            )

        assert code == 0
        assert "Intent:" in stdout
        assert "DIRECT" in stdout
        assert "Target Role:" in stdout
        assert "Source:" in stdout

    def test_describe_intent_json(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "intent", "describe", "SIMPLE_CODE", "--json",
            )

        assert code == 0
        data = json.loads(stdout)
        assert data["name"] == "SIMPLE_CODE"
        assert "target_role" in data
        assert "role_chain" in data

    def test_describe_unknown_intent_exits_error(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "intent", "describe", "NONEXISTENT",
            )

        assert code == 1
        assert "not found" in stderr
        assert "NONEXISTENT" in stderr

    def test_describe_shows_role_chain(self, tmp_path):
        """intent describe should show the models in the target role's chain."""
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "intent", "describe", "DIRECT",
            )

        assert code == 0
        # DIRECT -> coding role -> mock_ollama
        assert "mock_ollama" in stdout


# ======================================================================
# T4.3: catalog artifacts --kind analyzer includes declared intents
# ======================================================================

class TestCatalogAnalyzerIntents:
    """Verify catalog artifacts --kind analyzer shows declared intents."""

    def test_catalog_artifacts_analyzer_shows_intents(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        # catalog_list returns enriched dicts
        api_mock.catalog_list.return_value = [
            {
                "artifact_id": "aurarouter-default",
                "kind": "analyzer",
                "display_name": "AuraRouter Default",
                "provider": "",
                "status": "registered",
                "role_bindings": {
                    "simple_code": "coding",
                    "complex_reasoning": "reasoning",
                    "review": "reviewer",
                },
            },
        ]

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "catalog", "artifacts", "--kind", "analyzer",
            )

        assert code == 0
        assert "DECLARED INTENTS" in stdout
        # The intents from role_bindings should appear
        assert "simple_code" in stdout or "review" in stdout

    def test_catalog_artifacts_analyzer_json_includes_intents(self, tmp_path):
        config_path = _make_config(tmp_path)

        api_mock = MagicMock()
        api_mock.__enter__ = MagicMock(return_value=api_mock)
        api_mock.__exit__ = MagicMock(return_value=False)

        from aurarouter.config import ConfigLoader

        real_config = ConfigLoader(config_path=config_path)
        api_mock._config = real_config

        api_mock.catalog_list.return_value = [
            {
                "artifact_id": "aurarouter-default",
                "kind": "analyzer",
                "display_name": "AuraRouter Default",
                "provider": "",
                "status": "registered",
                "role_bindings": {
                    "simple_code": "coding",
                    "complex_reasoning": "reasoning",
                    "review": "reviewer",
                },
            },
        ]

        with patch("aurarouter.cli._make_api", return_value=api_mock):
            stdout, stderr, code = _run_cli(
                "--config", config_path,
                "catalog", "artifacts", "--kind", "analyzer", "--json",
            )

        assert code == 0
        data = json.loads(stdout)
        assert len(data) == 1
        assert "declared_intents" in data[0]
        intents = data[0]["declared_intents"]
        assert "simple_code" in intents
        assert "review" in intents


# ======================================================================
# T4.4: API execute_task intent parameter
# ======================================================================

class TestApiExecuteTaskIntent:
    """Verify the API layer properly handles the intent parameter."""

    def test_execute_task_with_intent_skips_classification(self, tmp_path):
        """When intent is provided, analyze_intent should not be called."""
        config_path = _make_config(tmp_path)

        from aurarouter.api import AuraRouterAPI, APIConfig

        # Patch analyze_intent inside the routing module (lazy-imported)
        with patch("aurarouter.routing.analyze_intent") as mock_analyze, \
             patch("aurarouter.routing.generate_plan", return_value=["step 1"]), \
             patch("aurarouter.routing.review_output") as mock_review:

            mock_review.return_value = MagicMock(verdict="PASS", feedback="ok")

            api = AuraRouterAPI(APIConfig(config_path=config_path))
            with api:
                # Mock the fabric execute to return something
                api._fabric = MagicMock()
                api._fabric.execute.return_value = "result text"

                result = api.execute_task(
                    task="Do something",
                    intent="SIMPLE_CODE",
                )

            # analyze_intent should NOT have been called since intent was forced
            mock_analyze.assert_not_called()
            assert result.intent == "SIMPLE_CODE"
