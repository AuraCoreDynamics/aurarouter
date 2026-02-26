"""Tests for the auto-tuning module."""

import struct
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ------------------------------------------------------------------
# GGUF test file helpers
# ------------------------------------------------------------------

def _build_gguf_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    key_encoded = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_encoded))
        + key_encoded
        + struct.pack("<I", value_type)
        + value_bytes
    )


def _build_gguf_string_value(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_minimal_gguf(kv_pairs: list[tuple[str, int, bytes]], version: int = 3) -> bytes:
    header = b"GGUF"
    header += struct.pack("<I", version)
    header += struct.pack("<Q", 0)
    header += struct.pack("<Q", len(kv_pairs))
    for key, vtype, vbytes in kv_pairs:
        header += _build_gguf_kv(key, vtype, vbytes)
    return header


def _make_gguf_file(path: Path, *, architecture="phi3", context_length=16384,
                    param_count=3_800_000_000, has_chat_template=True) -> Path:
    """Write a minimal valid GGUF file with specified metadata."""
    kv_pairs = [
        ("general.architecture", 8, _build_gguf_string_value(architecture)),
        (f"{architecture}.context_length", 4, struct.pack("<I", context_length)),
        ("general.parameter_count", 10, struct.pack("<Q", param_count)),
    ]
    if has_chat_template:
        kv_pairs.append(
            ("tokenizer.chat_template", 8, _build_gguf_string_value("{% ... %}"))
        )
    path.write_bytes(_build_minimal_gguf(kv_pairs))
    return path


# ------------------------------------------------------------------
# extract_gguf_metadata
# ------------------------------------------------------------------

def test_extract_gguf_metadata(tmp_path):
    from aurarouter.tuning import extract_gguf_metadata

    model_file = _make_gguf_file(
        tmp_path / "test.gguf",
        architecture="phi3", context_length=16384,
        param_count=3_800_000_000, has_chat_template=True,
    )

    meta = extract_gguf_metadata(model_file)

    assert meta["context_length"] == 16384
    assert meta["has_chat_template"] is True
    assert meta["architecture"] == "phi3"
    assert meta["model_size_bytes"] == model_file.stat().st_size
    assert meta["parameter_count"] == 3_800_000_000


def test_extract_gguf_metadata_no_chat_template(tmp_path):
    from aurarouter.tuning import extract_gguf_metadata

    model_file = _make_gguf_file(
        tmp_path / "base.gguf",
        architecture="llama", context_length=2048,
        param_count=7_000_000_000, has_chat_template=False,
    )

    meta = extract_gguf_metadata(model_file)

    assert meta["has_chat_template"] is False
    assert meta["architecture"] == "llama"


def test_extract_gguf_metadata_file_not_found():
    from aurarouter.tuning import extract_gguf_metadata

    with pytest.raises(FileNotFoundError, match="GGUF file not found"):
        extract_gguf_metadata("/nonexistent/model.gguf")


def test_extract_gguf_metadata_minimal_file(tmp_path):
    """A GGUF file with no architecture-specific metadata returns defaults."""
    from aurarouter.tuning import extract_gguf_metadata

    # Write a valid GGUF file with no metadata entries
    model_file = tmp_path / "minimal.gguf"
    model_file.write_bytes(_build_minimal_gguf([]))

    meta = extract_gguf_metadata(model_file)

    assert meta["context_length"] == 0
    assert meta["parameter_count"] == 0
    assert meta["has_chat_template"] is False
    assert meta["architecture"] == "unknown"


# ------------------------------------------------------------------
# recommend_llamacpp_params
# ------------------------------------------------------------------

def test_recommend_params_with_gpu(tmp_path):
    from aurarouter.tuning import recommend_llamacpp_params

    model_file = tmp_path / "small.gguf"
    model_file.write_bytes(b"x" * 1000)

    metadata = {
        "context_length": 8192,
        "has_chat_template": True,
        "architecture": "phi3",
        "model_size_bytes": 1000,
        "parameter_count": 1_000_000,
    }

    with patch("aurarouter.tuning._detect_vram_bytes", return_value=16 * 1024**3):
        params = recommend_llamacpp_params(model_file, metadata)

    assert params["n_ctx"] == 8192
    assert params["n_gpu_layers"] == -1
    assert "temperature" in params
    assert "max_tokens" in params


def test_recommend_params_without_gpu(tmp_path):
    from aurarouter.tuning import recommend_llamacpp_params

    model_file = tmp_path / "cpu.gguf"
    model_file.write_bytes(b"x" * 1000)

    metadata = {
        "context_length": 4096,
        "has_chat_template": False,
        "architecture": "llama",
        "model_size_bytes": 1000,
        "parameter_count": 7_000_000_000,
    }

    with patch("aurarouter.tuning._detect_vram_bytes", return_value=0):
        params = recommend_llamacpp_params(model_file, metadata)

    assert params["n_gpu_layers"] == 0


def test_recommend_params_caps_context(tmp_path):
    from aurarouter.tuning import MAX_RECOMMENDED_CTX, recommend_llamacpp_params

    model_file = tmp_path / "big_ctx.gguf"
    model_file.write_bytes(b"x" * 1000)

    metadata = {
        "context_length": 131072,
        "has_chat_template": True,
        "architecture": "phi3",
        "model_size_bytes": 1000,
        "parameter_count": 3_800_000_000,
    }

    with patch("aurarouter.tuning._detect_vram_bytes", return_value=0):
        params = recommend_llamacpp_params(model_file, metadata)

    assert params["n_ctx"] == MAX_RECOMMENDED_CTX


def test_recommend_params_zero_context_gets_default(tmp_path):
    from aurarouter.tuning import recommend_llamacpp_params

    model_file = tmp_path / "zero_ctx.gguf"
    model_file.write_bytes(b"x" * 100)

    metadata = {
        "context_length": 0,
        "has_chat_template": True,
        "architecture": "llama",
        "model_size_bytes": 100,
        "parameter_count": 0,
    }

    with patch("aurarouter.tuning._detect_vram_bytes", return_value=0):
        params = recommend_llamacpp_params(model_file, metadata)

    assert params["n_ctx"] == 4096


def test_recommend_params_includes_threads(tmp_path):
    from aurarouter.tuning import recommend_llamacpp_params

    model_file = tmp_path / "threads.gguf"
    model_file.write_bytes(b"x" * 100)

    metadata = {
        "context_length": 4096,
        "has_chat_template": True,
        "architecture": "llama",
        "model_size_bytes": 100,
        "parameter_count": 0,
    }

    with (
        patch("aurarouter.tuning._detect_vram_bytes", return_value=0),
        patch("os.cpu_count", return_value=16),
    ):
        params = recommend_llamacpp_params(model_file, metadata)

    assert params["n_threads"] == 8


# ------------------------------------------------------------------
# auto_tune_model
# ------------------------------------------------------------------

def test_auto_tune_preserves_user_params(tmp_path):
    from aurarouter.tuning import auto_tune_model

    model_file = _make_gguf_file(
        tmp_path / "user.gguf",
        architecture="phi3", context_length=16384,
    )

    cfg = {
        "provider": "llamacpp",
        "model_path": str(model_file),
        "parameters": {
            "temperature": 0.1,
            "n_ctx": 8192,
        },
    }

    with patch("aurarouter.tuning._detect_vram_bytes", return_value=0):
        result = auto_tune_model("llamacpp", cfg)

    assert result["parameters"]["temperature"] == 0.1
    assert result["parameters"]["n_ctx"] == 8192
    assert "n_gpu_layers" in result["parameters"]
    assert "max_tokens" in result["parameters"]


def test_auto_tune_opt_out():
    from aurarouter.tuning import auto_tune_model

    cfg = {
        "provider": "llamacpp",
        "model_path": "/some/model.gguf",
        "auto_tune": False,
    }
    result = auto_tune_model("llamacpp", cfg)
    assert result is cfg


def test_auto_tune_non_llamacpp_noop():
    from aurarouter.tuning import auto_tune_model

    cfg = {"provider": "ollama", "model_name": "llama3"}
    result = auto_tune_model("ollama", cfg)
    assert result is cfg


def test_auto_tune_missing_model_path():
    from aurarouter.tuning import auto_tune_model

    cfg = {"provider": "llamacpp"}
    result = auto_tune_model("llamacpp", cfg)
    assert result is cfg


def test_auto_tune_stashes_metadata(tmp_path):
    from aurarouter.tuning import auto_tune_model

    model_file = _make_gguf_file(
        tmp_path / "meta.gguf",
        architecture="llama", context_length=4096,
        param_count=0, has_chat_template=False,
    )

    cfg = {
        "provider": "llamacpp",
        "model_path": str(model_file),
    }

    with patch("aurarouter.tuning._detect_vram_bytes", return_value=0):
        result = auto_tune_model("llamacpp", cfg)

    assert "_gguf_metadata" in result
    assert result["_gguf_metadata"]["architecture"] == "llama"


# ------------------------------------------------------------------
# _detect_vram_bytes
# ------------------------------------------------------------------

def test_detect_vram_no_gpu():
    """When neither torch nor pynvml is available, return 0."""
    from aurarouter.tuning import _detect_vram_bytes

    result = _detect_vram_bytes()
    assert isinstance(result, int)
