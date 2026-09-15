"""Tests for the CatalogArtifact domain model.

Exercises real parsing, serialization, and property behaviour rather than
testing enum values in isolation.
"""

import pytest

from aurarouter.catalog_model import ArtifactKind, CatalogArtifact


# ------------------------------------------------------------------
# from_dict behavioural parsing tests
# ------------------------------------------------------------------


class TestFromDictParsing:
    def test_kind_model_produces_enum(self):
        """from_dict with kind='model' produces ArtifactKind.MODEL."""
        a = CatalogArtifact.from_dict("x", {"kind": "model", "display_name": "X"})
        assert a.kind is ArtifactKind.MODEL

    def test_kind_service_produces_enum(self):
        a = CatalogArtifact.from_dict("x", {"kind": "service", "display_name": "X"})
        assert a.kind is ArtifactKind.SERVICE

    def test_kind_analyzer_produces_enum(self):
        a = CatalogArtifact.from_dict("x", {"kind": "analyzer", "display_name": "X"})
        assert a.kind is ArtifactKind.ANALYZER

    def test_missing_kind_defaults_to_model(self):
        """from_dict with no 'kind' key defaults to MODEL."""
        a = CatalogArtifact.from_dict("x", {"display_name": "X"})
        assert a.kind is ArtifactKind.MODEL

    def test_invalid_kind_raises_value_error(self):
        """from_dict with an invalid kind value raises ValueError."""
        with pytest.raises(ValueError):
            CatalogArtifact.from_dict("x", {"kind": "invalid", "display_name": "X"})

    def test_unknown_keys_go_to_spec(self):
        """Keys not in the known set are placed into spec."""
        data = {
            "kind": "model",
            "display_name": "M",
            "custom_field": "value",
            "another_unknown": 42,
        }
        a = CatalogArtifact.from_dict("m", data)
        assert a.spec["custom_field"] == "value"
        assert a.spec["another_unknown"] == 42
        # Known keys should NOT be in spec
        assert "kind" not in a.spec
        assert "display_name" not in a.spec

    def test_display_name_defaults_to_artifact_id(self):
        a = CatalogArtifact.from_dict("my-id", {"kind": "model"})
        assert a.display_name == "my-id"


# ------------------------------------------------------------------
# to_dict round-trip and omission behaviour
# ------------------------------------------------------------------


class TestToDictBehaviour:
    def test_round_trip_preserves_kind(self):
        """Kind survives a to_dict -> from_dict cycle for all three kinds."""
        for kind_str in ("model", "service", "analyzer"):
            original = CatalogArtifact.from_dict(
                "rt", {"kind": kind_str, "display_name": "RT"},
            )
            d = original.to_dict()
            restored = CatalogArtifact.from_dict("rt", d)
            assert restored.kind.value == kind_str

    def test_omits_empty_tags(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.MODEL, display_name="X", tags=[],
        )
        assert "tags" not in a.to_dict()

    def test_omits_empty_capabilities(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.MODEL, display_name="X", capabilities=[],
        )
        assert "capabilities" not in a.to_dict()

    def test_omits_empty_description(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.MODEL, display_name="X", description="",
        )
        assert "description" not in a.to_dict()

    def test_includes_nonempty_tags(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.MODEL, display_name="X", tags=["t1"],
        )
        assert a.to_dict()["tags"] == ["t1"]

    def test_registered_status_omitted(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.MODEL, display_name="X", status="registered",
        )
        assert "status" not in a.to_dict()

    def test_custom_status_included(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.MODEL, display_name="X", status="active",
        )
        assert a.to_dict()["status"] == "active"

    def test_spec_merges_at_top_level(self):
        a = CatalogArtifact(
            artifact_id="x", kind=ArtifactKind.SERVICE, display_name="X",
            spec={"mcp_endpoint": "http://x", "timeout": 30},
        )
        d = a.to_dict()
        assert d["mcp_endpoint"] == "http://x"
        assert d["timeout"] == 30

    def test_full_round_trip_analyzer(self):
        original = CatalogArtifact(
            artifact_id="a1",
            kind=ArtifactKind.ANALYZER,
            display_name="Analyzer 1",
            description="desc",
            provider="prov",
            version="2.0",
            tags=["ai"],
            capabilities=["code"],
            status="active",
            spec={"analyzer_kind": "intent_triage"},
        )
        d = original.to_dict()
        restored = CatalogArtifact.from_dict("a1", d)
        assert restored.kind is ArtifactKind.ANALYZER
        assert restored.description == "desc"
        assert restored.provider == "prov"
        assert restored.version == "2.0"
        assert restored.tags == ["ai"]
        assert restored.capabilities == ["code"]
        assert restored.status == "active"
        assert restored.spec["analyzer_kind"] == "intent_triage"


# ------------------------------------------------------------------
# is_remote property
# ------------------------------------------------------------------


class TestIsRemote:
    def test_is_remote_true_with_mcp_endpoint(self):
        a = CatalogArtifact(
            artifact_id="r",
            kind=ArtifactKind.MODEL,
            display_name="R",
            spec={"mcp_endpoint": "http://remote:8080"},
        )
        assert a.is_remote is True

    def test_is_remote_false_without_endpoint(self):
        a = CatalogArtifact(
            artifact_id="local",
            kind=ArtifactKind.MODEL,
            display_name="Local",
        )
        assert a.is_remote is False

    def test_is_remote_false_with_none_endpoint(self):
        a = CatalogArtifact(
            artifact_id="x",
            kind=ArtifactKind.MODEL,
            display_name="X",
            spec={"mcp_endpoint": None},
        )
        assert a.is_remote is False
