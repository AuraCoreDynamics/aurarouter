"""Tests for aurarouter.analyzers — create_default_analyzer and friends."""

import pytest

from aurarouter.analyzers import create_default_analyzer
from aurarouter.catalog_model import ArtifactKind, CatalogArtifact


# ------------------------------------------------------------------
# create_default_analyzer
# ------------------------------------------------------------------


class TestCreateDefaultAnalyzer:
    def test_returns_catalog_artifact(self):
        artifact = create_default_analyzer()
        assert isinstance(artifact, CatalogArtifact)

    def test_kind_is_analyzer(self):
        artifact = create_default_analyzer()
        assert artifact.kind is ArtifactKind.ANALYZER

    def test_artifact_id(self):
        artifact = create_default_analyzer()
        assert artifact.artifact_id == "aurarouter-default"

    def test_display_name(self):
        artifact = create_default_analyzer()
        assert artifact.display_name == "AuraRouter Default"

    def test_provider_is_aurarouter(self):
        artifact = create_default_analyzer()
        assert artifact.provider == "aurarouter"

    def test_version(self):
        artifact = create_default_analyzer()
        assert artifact.version == "1.0"

    def test_capabilities_populated(self):
        artifact = create_default_analyzer()
        assert "code" in artifact.capabilities
        assert "reasoning" in artifact.capabilities

    def test_spec_contains_role_bindings(self):
        artifact = create_default_analyzer()
        assert "role_bindings" in artifact.spec
        bindings = artifact.spec["role_bindings"]
        assert "simple_code" in bindings
        assert "complex_reasoning" in bindings
        assert "review" in bindings

    def test_spec_analyzer_kind(self):
        artifact = create_default_analyzer()
        assert artifact.spec["analyzer_kind"] == "intent_triage"

    def test_is_not_remote(self):
        artifact = create_default_analyzer()
        assert artifact.is_remote is False

    def test_round_trip_through_dict(self):
        """to_dict -> from_dict preserves all meaningful fields."""
        original = create_default_analyzer()
        d = original.to_dict()
        restored = CatalogArtifact.from_dict(original.artifact_id, d)

        assert restored.kind is ArtifactKind.ANALYZER
        assert restored.display_name == original.display_name
        assert restored.provider == original.provider
        assert restored.version == original.version
        assert restored.capabilities == original.capabilities
        assert restored.spec["role_bindings"] == original.spec["role_bindings"]
        assert restored.spec["analyzer_kind"] == original.spec["analyzer_kind"]
