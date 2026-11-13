"""Tests ensuring local tooling and CI workflow stay aligned."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_defines_explicit_ruff_lint_select():
    """pyproject.toml must pin [tool.ruff.lint].select for deterministic CI lint."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ruff_lint = data.get("tool", {}).get("ruff", {}).get("lint", {})
    select = ruff_lint.get("select", [])
    assert isinstance(select, list) and len(select) > 0
    for rule in ("E4", "E7", "E9", "F"):
        assert rule in select


def test_ci_workflow_and_local_script_parity():
    """Verify .github/workflows/ci.yml steps match scripts/ci_local.py."""
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    ci_cfg = yaml.safe_load(ci_path.read_text(encoding="utf-8"))

    test_steps = ci_cfg["jobs"]["test"]["steps"]
    build_steps = ci_cfg["jobs"]["build"]["steps"]

    test_runs = "\n".join(str(step.get("run", "")) for step in test_steps)
    build_runs = "\n".join(str(step.get("run", "")) for step in build_steps)

    assert "ruff check src tests" in test_runs
    assert "pytest tests/ -x -q" in test_runs
    assert "pyproject-build" in build_runs
    assert "twine check dist/*" in build_runs

    local_script = (REPO_ROOT / "scripts" / "ci_local.py").read_text(encoding="utf-8")
    assert "ruff" in local_script
    assert "pytest" in local_script
    assert "QT_QPA_PLATFORM" in local_script
    assert "pyproject-build" in local_script
    assert "twine" in local_script
