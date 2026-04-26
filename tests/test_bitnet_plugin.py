"""Tests for the aurarouter-bitnet backend plugin."""

import pytest
from aurarouter_bitnet import METADATA, run_diagnostic, setup_runtime_environment


class TestMetadata:
    def test_required_fields_present(self):
        for key in ("package_name", "flavor", "compute_type", "version", "score"):
            assert key in METADATA, f"Missing METADATA key: {key}"

    def test_compute_type_is_cpu(self):
        assert METADATA["compute_type"] == "CPU"

    def test_flavor(self):
        assert METADATA["flavor"] == "BitNet"

    def test_score(self):
        assert METADATA["score"] == 70

    def test_version(self):
        assert METADATA["version"] == "0.1.0"

    def test_is_gpu_false(self):
        assert METADATA["is_gpu"] is False


class TestDiagnostics:
    def test_return_shape(self):
        result = run_diagnostic()
        assert isinstance(result, dict)
        for key in ("supported", "features", "binary_found", "platform"):
            assert key in result, f"Missing diagnostic key: {key}"

    def test_supported_is_bool(self):
        result = run_diagnostic()
        assert isinstance(result["supported"], bool)

    def test_features_is_list(self):
        result = run_diagnostic()
        assert isinstance(result["features"], list)

    def test_binary_found_is_bool(self):
        result = run_diagnostic()
        assert isinstance(result["binary_found"], bool)

    def test_platform_is_string(self):
        result = run_diagnostic()
        assert isinstance(result["platform"], str)
        assert "/" in result["platform"]


class TestRuntime:
    def test_raises_for_missing_binary(self):
        with pytest.raises(FileNotFoundError):
            setup_runtime_environment()
