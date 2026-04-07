"""Integration tests for the AuraRouter CLI (v0.5.4).

Verifies that CLI help text renders for all commands and that the
version output matches 0.5.4.
"""

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``python -m aurarouter`` with the given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "aurarouter", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersion:
    """Version is 0.5.4 everywhere."""

    def test_python_version_import(self):
        from aurarouter import __version__

        assert __version__ == "0.5.4"

    def test_api_importable(self):
        from aurarouter import AuraRouterAPI, __version__

        assert __version__ == "0.5.4"
        assert AuraRouterAPI is not None


# ---------------------------------------------------------------------------
# CLI help rendering
# ---------------------------------------------------------------------------


class TestCLIHelp:
    """Each CLI subcommand's --help renders without error."""

    def test_main_help(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "aurarouter" in result.stdout.lower()

    @pytest.mark.parametrize(
        "subcommand",
        [
            "model",
            "route",
            "run",
            "compare",
            "traffic",
            "privacy",
            "health",
            "budget",
            "config",
            "catalog",
            "gui",
            "download-model",
            "list-models",
            "remove-model",
        ],
    )
    def test_subcommand_help(self, subcommand):
        result = _run_cli(subcommand, "--help")
        assert result.returncode == 0, (
            f"{subcommand} --help failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    @pytest.mark.parametrize(
        "group,sub",
        [
            ("model", "list"),
            ("model", "add"),
            ("model", "edit"),
            ("model", "remove"),
            ("model", "test"),
            ("model", "auto-tune"),
            ("route", "list"),
            ("route", "set"),
            ("route", "append"),
            ("route", "remove-model"),
            ("route", "delete"),
            ("config", "show"),
            ("config", "set"),
            ("config", "mcp-tool"),
            ("config", "save"),
            ("config", "reload"),
            ("catalog", "list"),
            ("catalog", "add"),
            ("catalog", "remove"),
            ("catalog", "start"),
            ("catalog", "stop"),
            ("catalog", "health"),
            ("catalog", "discover"),
        ],
    )
    def test_nested_subcommand_help(self, group, sub):
        result = _run_cli(group, sub, "--help")
        assert result.returncode == 0, (
            f"{group} {sub} --help failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
