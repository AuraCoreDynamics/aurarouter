"""Tests for the auto-tuning module."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ------------------------------------------------------------------
# Mock llama_cpp if not installed
# ------------------------------------------------------------------

_llama_cpp_mock = None

@pytest.fixture(autouse=True)
def _ensure_llama_cpp_mock():
    """Ensure llama_cpp is available (as a mock) for all tests in this file."""
    global _llama_cpp_mock
    if "llama_cpp" not in sys.modules:
        mod = ModuleType("llama_cpp")
        mod.Llama = MagicMock  # type: ignore[attr-defined]
        sys.modules["llama_cpp"] = mod
        _llama_cpp_mock = mod
    yield
    # Don't remove — other imports may have cached references


def _make_mock_llm(*, context_length=16384, param_count=3_800_000_000,
                   architecture="phi3", has_chat_template=True):
    """Build a mock Llama instance with controllable metadata."""
    mock_model = MagicMock()
    mock_model.n_ctx_train.return_value = context_length
    mock_model.n_params.return_value = param_count

    mock_llm = MagicMock()
    metadata = {"general.architecture": architecture}
    if has_chat_template:
        metadata["tokenizer.chat_template"] = "{% ... %}"
    mock_llm.metadata = metadata
    mock_llm._model = mock_model
    return mock_llm


# ------------------------------------------------------------------
# extract_gguf_metadata
# ------------------------------------------------------------------

def test_extract_gguf_metadata(tmp_path):
    from aurarouter.tuning import extract_gguf_metadata

    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake-gguf-data")

    mock_llm = _make_mock_llm()

    with patch("llama_cpp.Llama", return_value=mock_llm):
        meta = extract_gguf_metadata(model_file)

    assert meta["context_length"] == 16384
    assert meta["has_chat_template"] is True
    assert meta["architecture"] == "phi3"
    assert meta["model_size_bytes"] == len(b"fake-gguf-data")
    assert meta["parameter_count"] == 3_800_000_000


def test_extract_gguf_metadata_no_chat_template(tmp_path):
    from aurarouter.tuning import extract_gguf_metadata

    model_file = tmp_path / "base.gguf"
    model_file.write_bytes(b"base-model")

    mock_llm = _make_mock_llm(has_chat_template=False, architecture="llama",
                              context_length=2048, param_count=7_000_000_000)

    with patch("llama_cpp.Llama", return_value=mock_llm):
        meta = extract_gguf_metadata(model_file)

    assert meta["has_chat_template"] is False
    assert meta["architecture"] == "llama"


def test_extract_gguf_metadata_file_not_found():
    from aurarouter.tuning import extract_gguf_metadata

    with pytest.raises(FileNotFoundError, match="GGUF file not found"):
        extract_gguf_metadata("/nonexistent/model.gguf")


def test_extract_gguf_metadata_graceful_on_method_error(tmp_path):
    """If _model methods raise, metadata should still return with defaults."""
    from aurarouter.tuning import extract_gguf_metadata

    model_file = tmp_path / "broken.gguf"
    model_file.write_bytes(b"broken")

    mock_model = MagicMock()
    mock_model.n_ctx_train.side_effect = RuntimeError("unsupported")
    mock_model.n_params.side_effect = RuntimeError("unsupported")

    mock_llm = MagicMock()
    mock_llm.metadata = {}
    mock_llm._model = mock_model

    with patch("llama_cpp.Llama", return_value=mock_llm):
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

    model_file = tmp_path / "user.gguf"
    model_file.write_bytes(b"x" * 1000)

    mock_llm = _make_mock_llm()

    cfg = {
        "provider": "llamacpp",
        "model_path": str(model_file),
        "parameters": {
            "temperature": 0.1,
            "n_ctx": 8192,
        },
    }

    with (
        patch("llama_cpp.Llama", return_value=mock_llm),
        patch("aurarouter.tuning._detect_vram_bytes", return_value=0),
    ):
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

    model_file = tmp_path / "meta.gguf"
    model_file.write_bytes(b"x" * 1000)

    mock_llm = _make_mock_llm(architecture="llama", context_length=4096,
                              param_count=0, has_chat_template=False)

    cfg = {
        "provider": "llamacpp",
        "model_path": str(model_file),
    }

    with (
        patch("llama_cpp.Llama", return_value=mock_llm),
        patch("aurarouter.tuning._detect_vram_bytes", return_value=0),
    ):
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
