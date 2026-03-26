"""Setup wizard helpers — hardware detection, provider discovery, recommendations.

Pure logic, no Qt dependencies. All functions are designed to be called from
background threads in the wizard UI.
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Summary of detected hardware capabilities."""
    gpu_name: str = "No GPU detected"
    vram_mb: int = 0
    has_nvidia: bool = False
    cuda_version: str | None = None
    installed_sidecars: list[str] = field(default_factory=list)
    cpu_cores: int = 1


def detect_hardware() -> HardwareInfo:
    """Detect GPU, VRAM, CUDA version, installed sidecars, CPU cores.

    Uses nvidia-smi for GPU name and CUDA version, tuning._detect_vram_bytes()
    for VRAM, importlib.metadata for sidecar discovery.
    """
    info = HardwareInfo(cpu_cores=os.cpu_count() or 1)

    # GPU detection via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                info.gpu_name = parts[0].strip()
                info.vram_mb = int(float(parts[1].strip()))
                info.has_nvidia = True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # If nvidia-smi failed, try tuning._detect_vram_bytes()
    if info.vram_mb == 0:
        try:
            from aurarouter.tuning import _detect_vram_bytes
            vram = _detect_vram_bytes()
            if vram > 0:
                info.vram_mb = vram // (1024 * 1024)
                info.has_nvidia = True
                info.gpu_name = "NVIDIA GPU"  # generic if nvidia-smi not available
        except Exception:
            pass

    # CUDA version detection
    if info.has_nvidia:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # Also try nvcc for CUDA toolkit version
                try:
                    nvcc = subprocess.run(
                        ["nvcc", "--version"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if nvcc.returncode == 0:
                        for line in nvcc.stdout.split("\n"):
                            if "release" in line.lower():
                                # "Cuda compilation tools, release 12.4, V12.4.131"
                                parts = line.split("release")[-1].strip().split(",")
                                info.cuda_version = parts[0].strip()
                                break
                except (FileNotFoundError, Exception):
                    pass
        except Exception:
            pass

    # Sidecar discovery
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] or ""
        if name.startswith("aurarouter-cuda") or name.startswith("aurarouter_cuda"):
            info.installed_sidecars.append(name)

    return info


def detect_ollama() -> dict:
    """Enhanced Ollama detection — returns available models, not just status.

    Returns:
        {"available": bool, "endpoint": str, "models": [{"name": str, "size": str}]}
    """
    import httpx

    endpoint = "http://localhost:11434"
    result = {"available": False, "endpoint": endpoint, "models": []}

    try:
        resp = httpx.get(f"{endpoint}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            result["available"] = True
            for model in data.get("models", []):
                name = model.get("name", "")
                size_bytes = model.get("size", 0)
                size_str = f"{size_bytes / (1024**3):.1f} GB" if size_bytes else "unknown"
                result["models"].append({"name": name, "size": size_str})
    except Exception:
        # Fallback: try CLI
        try:
            proc = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                result["available"] = True
                # Parse "NAME  ID  SIZE  MODIFIED" table
                for line in proc.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if parts:
                        result["models"].append({"name": parts[0], "size": parts[2] if len(parts) > 2 else "unknown"})
        except Exception:
            pass

    return result


def detect_cloud_providers() -> list[dict]:
    """Check which cloud provider packages are installed.

    Returns:
        [{"name": "gemini", "package": "aurarouter-gemini", "installed": bool,
          "display_name": "Google Gemini"}, ...]
    """
    providers = [
        {"name": "gemini", "package": "aurarouter-gemini", "display_name": "Google Gemini"},
        {"name": "claude", "package": "aurarouter-claude", "display_name": "Anthropic Claude"},
    ]

    installed_packages = {dist.metadata["Name"] for dist in importlib.metadata.distributions()}

    for p in providers:
        p["installed"] = p["package"] in installed_packages

    return providers


# Model recommendation table — curated GGUF models for different VRAM tiers
_RECOMMENDED_MODELS: list[dict] = [
    {
        "display_name": "Qwen 2.5 Coder 1.5B (tiny, fast)",
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-coder-1.5b-instruct-q8_0.gguf",
        "size_gb": 1.6,
        "min_vram_mb": 0,  # CPU-friendly
        "max_vram_mb": 4096,
        "reason": "Lightweight model that runs on any hardware",
    },
    {
        "display_name": "Qwen 2.5 Coder 7B (balanced)",
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.7,
        "min_vram_mb": 4096,
        "max_vram_mb": 12288,
        "reason": "Best balance of quality and speed for 6-8 GB VRAM",
    },
    {
        "display_name": "Qwen 2.5 Coder 14B (capable)",
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "size_gb": 9.0,
        "min_vram_mb": 12288,
        "max_vram_mb": 24576,
        "reason": "High-quality code generation for 12+ GB VRAM",
    },
    {
        "display_name": "Qwen 2.5 Coder 32B (frontier)",
        "repo": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "filename": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "size_gb": 20.0,
        "min_vram_mb": 24576,
        "max_vram_mb": 999999,
        "reason": "Near-frontier code quality for 24+ GB VRAM",
    },
]


def get_recommended_models(hw: HardwareInfo) -> list[dict]:
    """Return models appropriate for the detected hardware.

    Each model gets a 'recommended' flag (True for the best fit) and the
    full list is filtered to what the hardware can run.
    """
    vram = hw.vram_mb
    results = []

    for model in _RECOMMENDED_MODELS:
        if vram == 0:
            # No GPU — only recommend the smallest model for CPU
            if model["min_vram_mb"] == 0:
                results.append({**model, "recommended": True})
            break
        elif model["min_vram_mb"] <= vram <= model["max_vram_mb"]:
            results.append({**model, "recommended": True})
        elif model["min_vram_mb"] <= vram:
            results.append({**model, "recommended": False})

    # Always include the tiny model as a fallback option
    if results and not any(r["display_name"].startswith("Qwen 2.5 Coder 1.5B") for r in results):
        tiny = _RECOMMENDED_MODELS[0]
        results.append({**tiny, "recommended": False})

    return results


def suggest_cuda_sidecar(hw: HardwareInfo) -> str | None:
    """Suggest a CUDA sidecar package to install, or None if not needed.

    Returns the pip package name (e.g. "aurarouter-cuda13") or None.
    """
    if not hw.has_nvidia:
        return None
    if hw.installed_sidecars:
        return None  # Already have one

    # Suggest based on CUDA version
    if hw.cuda_version:
        major = hw.cuda_version.split(".")[0]
        if int(major) >= 13:
            return "aurarouter-cuda13"
        elif int(major) >= 12:
            return "aurarouter-cuda12"

    # Default to CUDA 12 if we can't detect version
    return "aurarouter-cuda12"


def pip_install_sync(package: str) -> tuple[bool, str]:
    """Install a pip package synchronously. Returns (success, output_text)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=300,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Installation timed out after 5 minutes"
    except Exception as e:
        return False, str(e)
