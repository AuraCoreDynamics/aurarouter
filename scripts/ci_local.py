"""Run the exact GitHub Actions CI checks locally before pushing.

Mirrors `.github/workflows/ci.yml`:
1. ``ruff check src tests``
2. ``QT_QPA_PLATFORM=offscreen pytest tests/ -x -q``
3. ``pyproject-build`` + ``twine check``
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tool_cmd(name: str) -> list[str]:
    scripts_dir = Path(sys.executable).resolve().parent
    for candidate in (scripts_dir / name, scripts_dir / f"{name}.exe"):
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    raise FileNotFoundError(
        f"Required CI tool '{name}' not found in {scripts_dir} or PATH. "
        'Install dev/build dependencies with: pip install -e ".[dev]" build twine'
    )


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n==> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> int:
    test_env = os.environ.copy()
    test_env.setdefault("QT_QPA_PLATFORM", "offscreen")

    _run([*_tool_cmd("ruff"), "check", "src", "tests"])
    _run([sys.executable, "-m", "pytest", "tests/", "-x", "-q"], env=test_env)

    with tempfile.TemporaryDirectory(prefix="aurarouter-ci-dist-") as tmp_dist:
        _run([*_tool_cmd("pyproject-build"), "--outdir", tmp_dist])
        dist_files = [str(p) for p in Path(tmp_dist).iterdir()]
        _run([*_tool_cmd("twine"), "check", *dist_files])

    print("\nAll local CI parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
