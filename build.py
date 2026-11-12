#!/usr/bin/env python3
"""Build script to create a standalone executable for AuraRouter using PyInstaller.

Usage:
    python build.py [--onedir] [--clean]

Produces:
    dist/aurarouter.exe (or dist/aurarouter on Linux/macOS)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone AuraRouter executable.")
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Create a directory containing the executable and dependencies (default: single-file).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean PyInstaller cache and remove temporary build artifacts before building.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    entry_point = project_root / "src" / "aurarouter" / "__main__.py"

    if not entry_point.exists():
        print(f"Error: Entry point not found at {entry_point}", file=sys.stderr)
        sys.exit(1)

    try:
        import PyInstaller
    except ImportError:
        print(
            "Error: PyInstaller is required to build standalone executables.\n"
            "Install with:  pip install pyinstaller",
            file=sys.stderr,
        )
        sys.exit(1)

    # Clean existing build/dist if requested
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"
    if args.clean:
        print("Cleaning previous build and dist artifacts...")
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        if dist_dir.exists():
            shutil.rmtree(dist_dir, ignore_errors=True)

    # Collect package data: onnx models, tokenizer, resources
    resource_dir = project_root / "src" / "aurarouter" / "resources"
    sep = ";" if sys.platform.startswith("win") else ":"
    data_args = []
    if resource_dir.exists():
        data_args.extend(["--add-data", f"{resource_dir}{sep}aurarouter/resources"])

    hidden_imports = [
        "aurarouter",
        "aurarouter.cli",
        "aurarouter.server",
        "aurarouter.fabric",
        "aurarouter.config",
        "aurarouter.mcp_tools",
        "aurarouter.routing_stack",
        "aurarouter.routing_stack.stack",
        "aurarouter.routing_stack.filters",
        "aurarouter.routing_stack.factors",
        "aurarouter.ratelimit",
        "aurarouter.ratelimit.gcra",
        "aurarouter.ratelimit.tracker",
        "aurarouter.ratelimit.parsers",
        "aurarouter.benchmarks",
        "aurarouter.sovereignty",
        "aurarouter.sovereignty.gate",
        "aurarouter.sovereignty.privacy",
        "aurarouter.sessions",
        "aurarouter.sessions.projector",
        "aurarouter.sessions.manager",
        "aurarouter.providers",
        "aurarouter.providers.ollama",
        "aurarouter.providers.openapi",
        "aurarouter.providers.onnx",
        "aurarouter.savings",
        "aurarouter.savings.budget",
        "aurarouter.savings.pricing",
        "aurarouter.savings.triage",
        "aurarouter.savings.usage_store",
        "mcp",
        "mcp.server",
        "mcp.server.fastmcp",
        "pyyaml",
        "httpx",
        "numpy",
        "onnxruntime",
        "tokenizers",
        "benchwarmer",
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        "aurarouter",
        "--noconfirm",
    ]

    if not args.onedir:
        cmd.append("--onefile")

    cmd.extend(data_args)

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    cmd.append(str(entry_point))

    print(f"Building AuraRouter standalone executable from {entry_point}...")
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print("Build failed.", file=sys.stderr)
        sys.exit(result.returncode)

    exe_name = "aurarouter.exe" if sys.platform.startswith("win") else "aurarouter"
    output_path = dist_dir / exe_name if not args.onedir else dist_dir / "aurarouter" / exe_name
    print(f"\nBuild successful! Executable created at: {output_path}")


if __name__ == "__main__":
    main()
