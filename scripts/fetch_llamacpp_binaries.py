"""Build-time script: fetch llama.cpp pre-built binaries from GitHub releases.

Usage:
    python scripts/fetch_llamacpp_binaries.py [--release TAG] [--dest DIR]

Downloads llama-server (and required shared libraries) for all three
platforms into src/aurarouter/bin/{win-x64,linux-x64,macos-x64}/.

This script is NOT shipped with the package -- it runs during development
or CI to populate the bin/ directory before building the wheel/sdist.
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Asset name patterns per platform
PLATFORM_ASSETS = {
    "win-x64": {
        "pattern": "bin-win-avx2-x64.zip",
        "binary": "llama-server.exe",
        "format": "zip",
    },
    "linux-x64": {
        "pattern": "bin-ubuntu-x64.tar.gz",
        "binary": "llama-server",
        "format": "tar.gz",
    },
    "macos-x64": {
        "pattern": "bin-macos-x64.tar.gz",
        "binary": "llama-server",
        "format": "tar.gz",
    },
}

API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def get_release_assets(tag: str | None = None) -> list[dict]:
    """Fetch release asset metadata from GitHub API."""
    if tag:
        url = f"{API_URL}/tags/{tag}"
    else:
        url = f"{API_URL}/latest"

    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    print(f"Release: {data.get('tag_name', 'unknown')}")
    return data.get("assets", [])


def find_asset(assets: list[dict], pattern: str) -> dict | None:
    """Find an asset whose name contains the pattern."""
    for asset in assets:
        if pattern in asset.get("name", ""):
            return asset
    return None


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  Downloading... {pct}% ({downloaded}/{total})", end="")
        print()


def extract_binaries(archive_path: Path, dest_dir: Path, fmt: str) -> list[str]:
    """Extract llama-server and shared libraries from archive."""
    extracted = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if not name:
                    continue
                # Extract llama-server.exe and DLLs
                if name == "llama-server.exe" or name.endswith(".dll"):
                    target = dest_dir / name
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(name)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                name = Path(member.name).name
                if not name or not member.isfile():
                    continue
                # Extract llama-server and shared libs
                if name == "llama-server" or name.endswith(".so") or name.endswith(".dylib"):
                    target = dest_dir / name
                    with tf.extractfile(member) as src:
                        with open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    # Make executable on Unix
                    os.chmod(target, 0o755)
                    extracted.append(name)

    return extracted


def main() -> None:
    default_dest = Path(__file__).resolve().parent.parent / "src" / "aurarouter" / "bin"

    parser = argparse.ArgumentParser(
        description="Fetch llama.cpp pre-built binaries from GitHub releases."
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Git tag of the release to download (default: latest).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help=f"Destination directory (default: {default_dest}).",
    )
    args = parser.parse_args()

    print("Fetching release metadata from GitHub...")
    assets = get_release_assets(args.release)

    if not assets:
        print("ERROR: No assets found in the release.", file=sys.stderr)
        sys.exit(1)

    summary = []

    for plat_key, plat_info in PLATFORM_ASSETS.items():
        dest_dir = args.dest / plat_key
        binary_target = dest_dir / plat_info["binary"]

        # Idempotency check
        asset = find_asset(assets, plat_info["pattern"])
        if asset is None:
            print(f"  [{plat_key}] WARNING: No matching asset for '{plat_info['pattern']}'")
            summary.append(f"  {plat_key}: SKIPPED (no matching asset)")
            continue

        if binary_target.is_file():
            existing_size = binary_target.stat().st_size
            if existing_size > 0:
                print(f"  [{plat_key}] Binary already exists ({existing_size} bytes), skipping.")
                summary.append(f"  {plat_key}: SKIPPED (already exists)")
                continue

        print(f"  [{plat_key}] Downloading {asset['name']}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / asset["name"]
            download_file(asset["browser_download_url"], archive_path)

            print(f"  [{plat_key}] Extracting...")
            extracted = extract_binaries(archive_path, dest_dir, plat_info["format"])
            summary.append(f"  {plat_key}: {len(extracted)} files -> {dest_dir}")
            for name in extracted:
                print(f"    - {name}")

    print("\n--- Summary ---")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
