"""Auto-tuning for local GGUF models.

Reads GGUF metadata (context length, chat template, architecture) via
native binary header parsing and recommends optimal inference parameters.
Used by the fabric at provider instantiation, the downloader after a
download, and the CLI/GUI for explicit ``auto-tune`` requests.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Tuning")

# Hard cap: even if the model supports 131K, most consumer GPUs can't
# handle much more than 32K tokens of KV cache comfortably.
MAX_RECOMMENDED_CTX = 32768

# Default generation parameters recommended for instruction-tuned models.
_DEFAULT_GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
}


# ------------------------------------------------------------------
# GPU VRAM detection
# ------------------------------------------------------------------

def _detect_vram_bytes() -> int:
    """Return total GPU VRAM in bytes, or 0 if no GPU is detected."""
    # Try torch first (most common in ML environments)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem
    except Exception:
        pass

    # Fallback: try pynvml (lighter weight)
    try:
        import pynvml  # type: ignore[import-untyped]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total
    except Exception:
        pass

    return 0


# ------------------------------------------------------------------
# GGUF metadata extraction (pure-Python binary parser)
# ------------------------------------------------------------------

# GGUF value type IDs and their struct format strings.
_GGUF_TYPE_FORMAT: dict[int, str] = {
    0: "<B",   # UINT8
    1: "<b",   # INT8
    2: "<H",   # UINT16
    3: "<h",   # INT16
    4: "<I",   # UINT32
    5: "<i",   # INT32
    6: "<f",   # FLOAT32
    7: "<?",   # BOOL (1 byte)
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}


def _read_gguf_value(f, value_type: int) -> Any:
    """Read a single GGUF typed value from a file object."""
    if value_type == 8:  # STRING
        (str_len,) = struct.unpack("<Q", f.read(8))
        return f.read(str_len).decode("utf-8", errors="replace")
    elif value_type == 9:  # ARRAY
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        elements = []
        for _ in range(count):
            elements.append(_read_gguf_value(f, elem_type))
        return elements
    elif value_type in _GGUF_TYPE_FORMAT:
        fmt = _GGUF_TYPE_FORMAT[value_type]
        size = struct.calcsize(fmt)
        (val,) = struct.unpack(fmt, f.read(size))
        return val
    else:
        raise ValueError(f"Unknown GGUF value type: {value_type}")


def _parse_gguf_metadata(file_path: Path) -> dict[str, Any]:
    """Parse GGUF file header and return all metadata key-value pairs.

    Only reads the metadata section -- does not load tensor data.
    Supports GGUF v2 and v3.

    Raises
    ------
    ValueError
        If the file has invalid magic bytes or an unsupported version.
    """
    metadata: dict[str, Any] = {}

    with open(file_path, "rb") as f:
        # Magic: 4 bytes "GGUF"
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(
                f"Not a valid GGUF file (magic={magic!r}, expected b'GGUF'): "
                f"{file_path}"
            )

        # Version: uint32 LE
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(
                f"Unsupported GGUF version {version} (expected 2 or 3): "
                f"{file_path}"
            )

        # Tensor count: uint64 LE
        (tensor_count,) = struct.unpack("<Q", f.read(8))

        # Metadata KV count: uint64 LE
        (kv_count,) = struct.unpack("<Q", f.read(8))

        # Read each key-value pair
        for _ in range(kv_count):
            # Key: uint64 LE length + UTF-8 string
            (key_len,) = struct.unpack("<Q", f.read(8))
            key = f.read(key_len).decode("utf-8", errors="replace")

            # Value type: uint32 LE
            (value_type,) = struct.unpack("<I", f.read(4))

            # Value: type-dependent
            value = _read_gguf_value(f, value_type)
            metadata[key] = value

    return metadata


def extract_gguf_metadata(model_path: str | Path) -> dict[str, Any]:
    """Extract metadata from a GGUF model file via native binary parsing.

    Parameters
    ----------
    model_path:
        Path to the ``.gguf`` file.

    Returns
    -------
    dict with keys:
        - ``context_length`` (int): trained context window size
        - ``has_chat_template`` (bool): whether the model has a chat template
        - ``architecture`` (str): model architecture name
        - ``model_size_bytes`` (int): file size on disk
        - ``parameter_count`` (int): number of model parameters

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    ValueError
        If the file is not a valid GGUF file (wrong magic or unsupported version).
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {model_path}")

    raw = _parse_gguf_metadata(model_path)

    architecture = raw.get("general.architecture", "unknown")
    context_length = raw.get(f"{architecture}.context_length", 0)
    has_chat_template = "tokenizer.chat_template" in raw
    parameter_count = raw.get("general.parameter_count", 0)

    return {
        "context_length": context_length,
        "has_chat_template": has_chat_template,
        "architecture": architecture,
        "model_size_bytes": model_path.stat().st_size,
        "parameter_count": parameter_count,
    }


# ------------------------------------------------------------------
# Parameter recommendation
# ------------------------------------------------------------------

def recommend_llamacpp_params(
    model_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate recommended inference parameters for a GGUF model.

    Parameters
    ----------
    model_path:
        Path to the ``.gguf`` file.
    metadata:
        Pre-extracted metadata dict (from :func:`extract_gguf_metadata`).
        If ``None``, metadata will be extracted automatically.

    Returns
    -------
    dict of recommended parameters (ready for the ``parameters`` key in
    ``auraconfig.yaml``).
    """
    model_path = Path(model_path)

    if metadata is None:
        metadata = extract_gguf_metadata(model_path)

    # Context window: use trained value, capped at MAX_RECOMMENDED_CTX
    ctx = metadata.get("context_length", 4096)
    if ctx <= 0:
        ctx = 4096
    n_ctx = min(ctx, MAX_RECOMMENDED_CTX)

    # GPU layers: offload everything if model fits in ~80% of VRAM
    vram = _detect_vram_bytes()
    model_size = metadata.get("model_size_bytes", 0)
    if vram > 0 and model_size > 0 and model_size < vram * 0.8:
        n_gpu_layers = -1
    elif vram > 0:
        # GPU exists but model is large — partial offload, let user tune
        n_gpu_layers = -1
        logger.info(
            "Model size (%d MB) may exceed 80%% VRAM (%d MB). "
            "Setting n_gpu_layers=-1; reduce if OOM occurs.",
            model_size // (1024 * 1024),
            vram // (1024 * 1024),
        )
    else:
        n_gpu_layers = 0

    # Thread count: approximate physical cores
    cpu_count = os.cpu_count() or 4
    n_threads = max(1, cpu_count // 2)

    params: dict[str, Any] = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads,
        **_DEFAULT_GEN_PARAMS,
    }

    logger.info("Recommended parameters for %s: %s", model_path.name, params)
    return params


# ------------------------------------------------------------------
# Top-level entry point
# ------------------------------------------------------------------

def auto_tune_model(provider: str, model_cfg: dict) -> dict:
    """Apply auto-tuned defaults to a model configuration.

    For the ``llamacpp`` provider, extracts GGUF metadata and merges
    recommended parameters into *model_cfg*.  User-specified parameters
    always take precedence over auto-tuned values.

    For other providers, returns *model_cfg* unchanged.

    An ``auto_tune: false`` key in *model_cfg* opts out entirely.

    Parameters
    ----------
    provider:
        Provider name (e.g. ``"llamacpp"``).
    model_cfg:
        The raw model configuration dict from ``auraconfig.yaml``.

    Returns
    -------
    A (possibly new) config dict with auto-tuned parameters merged in.
    """
    if model_cfg.get("auto_tune") is False:
        logger.info("Auto-tune disabled for this model (auto_tune: false).")
        return model_cfg

    if provider != "llamacpp":
        return model_cfg

    model_path = model_cfg.get("model_path")
    if not model_path or not Path(model_path).is_file():
        logger.debug("No model_path or file missing; skipping auto-tune.")
        return model_cfg

    try:
        metadata = extract_gguf_metadata(model_path)
        recommended = recommend_llamacpp_params(model_path, metadata)
    except Exception as exc:
        logger.warning("Auto-tune failed: %s", exc)
        return model_cfg

    # Merge: user-specified params override auto-tuned ones.
    user_params = model_cfg.get("parameters", {}) or {}
    merged = {**recommended, **user_params}

    result = dict(model_cfg)
    result["parameters"] = merged

    # Stash metadata so downstream code (llamacpp provider) can use it
    # without re-loading the model just for metadata.
    result["_gguf_metadata"] = metadata

    logger.info(
        "Auto-tuned parameters (user overrides preserved): %s", merged
    )
    return result
