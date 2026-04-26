"""Tests for the aurarouter-bitnet backend plugin."""

import pytest
from aurarouter_bitnet import METADATA, run_diagnostic, setup_runtime_environment, get_catalog_artifact


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


class TestCatalogArtifact:
    def test_artifact_id(self):
        artifact = get_catalog_artifact()
        assert artifact["artifact_id"] == "aurarouter-bitnet"

    def test_kind_is_model(self):
        artifact = get_catalog_artifact()
        assert artifact["kind"] == "model"

    def test_display_name(self):
        artifact = get_catalog_artifact()
        assert artifact["display_name"] == "BitNet 1.58-bit (CPU Ternary)"

    def test_capabilities(self):
        artifact = get_catalog_artifact()
        assert "ternary-inference" in artifact["capabilities"]
        assert "cpu-only" in artifact["capabilities"]
        assert "edge-deployment" in artifact["capabilities"]

    def test_supported_intents(self):
        artifact = get_catalog_artifact()
        assert artifact["supported_intents"] == ["LOCAL_INFERENCE"]

    def test_spec_compute_type(self):
        artifact = get_catalog_artifact()
        assert artifact["spec"]["compute_type"] == "CPU"

    def test_spec_flavor(self):
        artifact = get_catalog_artifact()
        assert artifact["spec"]["flavor"] == "BitNet"

    def test_spec_weight_bits(self):
        artifact = get_catalog_artifact()
        assert artifact["spec"]["weight_bits"] == "1.58"

    def test_returns_new_dict_each_call(self):
        a = get_catalog_artifact()
        b = get_catalog_artifact()
        assert a == b
        assert a is not b
