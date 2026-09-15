"""Tests for the pure-Python GGUF parser in aurarouter.tuning."""

import struct
from pathlib import Path

import pytest

from aurarouter.tuning import _parse_gguf_metadata, extract_gguf_metadata


def _build_gguf_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    """Build a single GGUF key-value pair as raw bytes."""
    key_encoded = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_encoded))
        + key_encoded
        + struct.pack("<I", value_type)
        + value_bytes
    )


def _build_gguf_string_value(s: str) -> bytes:
    """Build a GGUF STRING value (type 8)."""
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_minimal_gguf(kv_pairs: list[tuple[str, int, bytes]], version: int = 3) -> bytes:
    """Build a minimal valid GGUF file with given KV pairs."""
    header = b"GGUF"
    header += struct.pack("<I", version)      # version
    header += struct.pack("<Q", 0)            # tensor_count
    header += struct.pack("<Q", len(kv_pairs))  # kv_count

    for key, vtype, vbytes in kv_pairs:
        header += _build_gguf_kv(key, vtype, vbytes)

    return header


class TestParseGgufMetadata:
    def test_parse_v3_header(self, tmp_path):
        """Parse a minimal valid GGUF v3 file with known metadata."""
        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("llama")),
            ("llama.context_length", 4, struct.pack("<I", 4096)),  # UINT32
            ("general.parameter_count", 10, struct.pack("<Q", 7_000_000_000)),  # UINT64
        ]
        data = _build_minimal_gguf(kv_pairs, version=3)
        fpath = tmp_path / "test.gguf"
        fpath.write_bytes(data)

        meta = _parse_gguf_metadata(fpath)
        assert meta["general.architecture"] == "llama"
        assert meta["llama.context_length"] == 4096
        assert meta["general.parameter_count"] == 7_000_000_000

    def test_parse_v2_header(self, tmp_path):
        """Parse a minimal valid GGUF v2 file."""
        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("qwen2")),
        ]
        data = _build_minimal_gguf(kv_pairs, version=2)
        fpath = tmp_path / "test.gguf"
        fpath.write_bytes(data)

        meta = _parse_gguf_metadata(fpath)
        assert meta["general.architecture"] == "qwen2"

    def test_invalid_magic(self, tmp_path):
        """Raise ValueError for non-GGUF files."""
        fpath = tmp_path / "bad.gguf"
        fpath.write_bytes(b"NOTG" + b"\x00" * 20)

        with pytest.raises(ValueError, match="Not a valid GGUF file"):
            _parse_gguf_metadata(fpath)

    def test_unsupported_version(self, tmp_path):
        """Raise ValueError for unsupported GGUF versions."""
        data = b"GGUF" + struct.pack("<I", 99) + b"\x00" * 16
        fpath = tmp_path / "bad.gguf"
        fpath.write_bytes(data)

        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            _parse_gguf_metadata(fpath)

    def test_all_scalar_types(self, tmp_path):
        """Parse all scalar value types correctly."""
        kv_pairs = [
            ("test.uint8", 0, struct.pack("<B", 42)),
            ("test.int8", 1, struct.pack("<b", -5)),
            ("test.uint16", 2, struct.pack("<H", 1000)),
            ("test.int16", 3, struct.pack("<h", -1000)),
            ("test.uint32", 4, struct.pack("<I", 100000)),
            ("test.int32", 5, struct.pack("<i", -100000)),
            ("test.float32", 6, struct.pack("<f", 3.14)),
            ("test.bool", 7, struct.pack("<?", True)),
            ("test.string", 8, _build_gguf_string_value("hello")),
            ("test.uint64", 10, struct.pack("<Q", 2**40)),
            ("test.int64", 11, struct.pack("<q", -(2**40))),
            ("test.float64", 12, struct.pack("<d", 2.718281828)),
        ]
        data = _build_minimal_gguf(kv_pairs, version=3)
        fpath = tmp_path / "test.gguf"
        fpath.write_bytes(data)

        meta = _parse_gguf_metadata(fpath)
        assert meta["test.uint8"] == 42
        assert meta["test.int8"] == -5
        assert meta["test.uint16"] == 1000
        assert meta["test.int16"] == -1000
        assert meta["test.uint32"] == 100000
        assert meta["test.int32"] == -100000
        assert abs(meta["test.float32"] - 3.14) < 0.001
        assert meta["test.bool"] is True
        assert meta["test.string"] == "hello"
        assert meta["test.uint64"] == 2**40
        assert meta["test.int64"] == -(2**40)
        assert abs(meta["test.float64"] - 2.718281828) < 1e-6

    def test_array_type_parsed(self, tmp_path):
        """Arrays are read and stored as lists."""
        # Build an array of 3 UINT32 values
        array_bytes = (
            struct.pack("<I", 4)  # element type: UINT32
            + struct.pack("<Q", 3)  # count
            + struct.pack("<I", 10)
            + struct.pack("<I", 20)
            + struct.pack("<I", 30)
        )
        kv_pairs = [
            ("test.array", 9, array_bytes),
            ("test.after", 4, struct.pack("<I", 999)),  # Verify parsing continues
        ]
        data = _build_minimal_gguf(kv_pairs, version=3)
        fpath = tmp_path / "test.gguf"
        fpath.write_bytes(data)

        meta = _parse_gguf_metadata(fpath)
        assert meta["test.array"] == [10, 20, 30]
        assert meta["test.after"] == 999


class TestExtractGgufMetadata:
    def test_returns_expected_keys(self, tmp_path):
        """extract_gguf_metadata returns all expected dict keys."""
        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("llama")),
            ("llama.context_length", 4, struct.pack("<I", 8192)),
            ("tokenizer.chat_template", 8, _build_gguf_string_value("{% ... %}")),
            ("general.parameter_count", 10, struct.pack("<Q", 7_000_000_000)),
        ]
        data = _build_minimal_gguf(kv_pairs)
        fpath = tmp_path / "model.gguf"
        fpath.write_bytes(data)

        meta = extract_gguf_metadata(fpath)
        assert meta["context_length"] == 8192
        assert meta["has_chat_template"] is True
        assert meta["architecture"] == "llama"
        assert meta["parameter_count"] == 7_000_000_000
        assert meta["model_size_bytes"] == fpath.stat().st_size

    def test_detects_chat_template(self, tmp_path):
        """has_chat_template is True when tokenizer.chat_template key exists."""
        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("qwen2")),
            ("tokenizer.chat_template", 8, _build_gguf_string_value("template")),
        ]
        data = _build_minimal_gguf(kv_pairs)
        fpath = tmp_path / "model.gguf"
        fpath.write_bytes(data)

        meta = extract_gguf_metadata(fpath)
        assert meta["has_chat_template"] is True

    def test_no_chat_template(self, tmp_path):
        """has_chat_template is False when tokenizer.chat_template is absent."""
        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("llama")),
        ]
        data = _build_minimal_gguf(kv_pairs)
        fpath = tmp_path / "model.gguf"
        fpath.write_bytes(data)

        meta = extract_gguf_metadata(fpath)
        assert meta["has_chat_template"] is False

    def test_file_not_found(self):
        """Raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            extract_gguf_metadata("/nonexistent/model.gguf")

    def test_context_length_from_architecture(self, tmp_path):
        """Context length is read from {arch}.context_length key."""
        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("qwen2")),
            ("qwen2.context_length", 4, struct.pack("<I", 32768)),
        ]
        data = _build_minimal_gguf(kv_pairs)
        fpath = tmp_path / "model.gguf"
        fpath.write_bytes(data)

        meta = extract_gguf_metadata(fpath)
        assert meta["context_length"] == 32768
        assert meta["architecture"] == "qwen2"


class TestRecommendParams:
    def test_cpu_only(self, tmp_path, monkeypatch):
        """n_gpu_layers=0 when no GPU detected."""
        monkeypatch.setattr("aurarouter.tuning._detect_vram_bytes", lambda: 0)

        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("llama")),
            ("llama.context_length", 4, struct.pack("<I", 4096)),
        ]
        data = _build_minimal_gguf(kv_pairs)
        fpath = tmp_path / "model.gguf"
        fpath.write_bytes(data)

        from aurarouter.tuning import recommend_llamacpp_params
        params = recommend_llamacpp_params(fpath)
        assert params["n_gpu_layers"] == 0

    def test_with_gpu(self, tmp_path, monkeypatch):
        """n_gpu_layers=-1 when GPU has sufficient VRAM."""
        monkeypatch.setattr(
            "aurarouter.tuning._detect_vram_bytes", lambda: 16 * 1024**3
        )

        kv_pairs = [
            ("general.architecture", 8, _build_gguf_string_value("llama")),
            ("llama.context_length", 4, struct.pack("<I", 4096)),
        ]
        data = _build_minimal_gguf(kv_pairs)
        fpath = tmp_path / "model.gguf"
        # Make file small relative to 16GB VRAM
        fpath.write_bytes(data)

        from aurarouter.tuning import recommend_llamacpp_params
        params = recommend_llamacpp_params(fpath)
        assert params["n_gpu_layers"] == -1
