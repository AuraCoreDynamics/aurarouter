"""Tests for Task Group 6: Tags-to-Intents Bridge & Model Eligibility.

Covers:
- supported_intents field on CatalogArtifact (to_dict / from_dict round-trip)
- catalog_query(supported_intents=...) filtering
- auto_join_roles() with and without intent registry
- filter_chain_by_intent() filtering and backwards compatibility
- Composition of tag + supported_intents filters
"""

import pytest

from aurarouter.catalog_model import ArtifactKind, CatalogArtifact
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import IntentDefinition, IntentRegistry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(catalog: dict | None = None, models: dict | None = None,
                 roles: dict | None = None) -> ConfigLoader:
    """Create a ConfigLoader with in-memory config (no YAML file)."""
    cfg = ConfigLoader(allow_missing=True)
    if catalog:
        cfg.config["catalog"] = catalog
    if models:
        cfg.config["models"] = models
    if roles:
        cfg.config["roles"] = roles
    return cfg


# ------------------------------------------------------------------
# T6.1: supported_intents field on CatalogArtifact
# ------------------------------------------------------------------


class TestSupportedIntentsField:
    def test_from_dict_extracts_supported_intents(self):
        data = {
            "kind": "model",
            "display_name": "SAR Model",
            "tags": ["geoint", "sar"],
            "supported_intents": ["sar_processing", "geoint_analysis"],
        }
        a = CatalogArtifact.from_dict("sar-model", data)
        assert a.supported_intents == ["sar_processing", "geoint_analysis"]

    def test_from_dict_defaults_to_empty_list(self):
        data = {"kind": "model", "display_name": "Plain Model"}
        a = CatalogArtifact.from_dict("plain", data)
        assert a.supported_intents == []

    def test_to_dict_includes_supported_intents(self):
        a = CatalogArtifact(
            artifact_id="m1",
            kind=ArtifactKind.MODEL,
            display_name="M1",
            supported_intents=["intent_a", "intent_b"],
        )
        d = a.to_dict()
        assert d["supported_intents"] == ["intent_a", "intent_b"]

    def test_to_dict_omits_empty_supported_intents(self):
        a = CatalogArtifact(
            artifact_id="m1",
            kind=ArtifactKind.MODEL,
            display_name="M1",
            supported_intents=[],
        )
        d = a.to_dict()
        assert "supported_intents" not in d

    def test_round_trip_preserves_supported_intents(self):
        original = CatalogArtifact(
            artifact_id="rt",
            kind=ArtifactKind.MODEL,
            display_name="RT",
            tags=["geoint"],
            capabilities=["reasoning"],
            supported_intents=["sar_processing", "geoint_analysis"],
        )
        d = original.to_dict()
        restored = CatalogArtifact.from_dict("rt", d)
        assert restored.supported_intents == ["sar_processing", "geoint_analysis"]
        assert restored.tags == ["geoint"]
        assert restored.capabilities == ["reasoning"]

    def test_supported_intents_not_in_spec(self):
        """supported_intents should be a proper field, not leaked into spec."""
        data = {
            "kind": "model",
            "display_name": "M",
            "supported_intents": ["x"],
            "custom_field": "val",
        }
        a = CatalogArtifact.from_dict("m", data)
        assert "supported_intents" not in a.spec
        assert a.spec["custom_field"] == "val"
        assert a.supported_intents == ["x"]


# ------------------------------------------------------------------
# T6.2: catalog_query with supported_intents filter
# ------------------------------------------------------------------


class TestCatalogQuerySupportedIntents:
    def test_filter_returns_matching_models(self):
        cfg = _make_config(catalog={
            "sar-model": {
                "kind": "model",
                "display_name": "SAR",
                "supported_intents": ["sar_processing", "geoint_analysis"],
            },
            "code-model": {
                "kind": "model",
                "display_name": "Code",
                "supported_intents": ["code_generation"],
            },
            "plain-model": {
                "kind": "model",
                "display_name": "Plain",
            },
        })
        results = cfg.catalog_query(supported_intents=["sar_processing"])
        ids = [r["artifact_id"] for r in results]
        assert ids == ["sar-model"]

    def test_filter_any_match(self):
        """At least one of the requested intents must match."""
        cfg = _make_config(catalog={
            "multi-model": {
                "kind": "model",
                "display_name": "Multi",
                "supported_intents": ["sar_processing", "geoint_analysis"],
            },
        })
        results = cfg.catalog_query(supported_intents=["geoint_analysis", "other"])
        assert len(results) == 1
        assert results[0]["artifact_id"] == "multi-model"

    def test_filter_excludes_models_without_supported_intents(self):
        cfg = _make_config(catalog={
            "no-intents": {
                "kind": "model",
                "display_name": "No Intents",
            },
        })
        results = cfg.catalog_query(supported_intents=["anything"])
        assert results == []

    def test_composition_tags_and_supported_intents(self):
        """Both tag and supported_intents filters must match (intersection)."""
        cfg = _make_config(catalog={
            "sar-geoint": {
                "kind": "model",
                "display_name": "SAR GEOINT",
                "tags": ["geoint"],
                "supported_intents": ["sar_processing"],
            },
            "sar-other": {
                "kind": "model",
                "display_name": "SAR Other",
                "tags": ["other"],
                "supported_intents": ["sar_processing"],
            },
            "geoint-plain": {
                "kind": "model",
                "display_name": "GEOINT Plain",
                "tags": ["geoint"],
            },
        })
        results = cfg.catalog_query(
            tags=["geoint"], supported_intents=["sar_processing"],
        )
        ids = [r["artifact_id"] for r in results]
        assert ids == ["sar-geoint"]

    def test_no_supported_intents_filter_returns_all(self):
        """When supported_intents is None, no filtering on that dimension."""
        cfg = _make_config(catalog={
            "a": {"kind": "model", "display_name": "A"},
            "b": {"kind": "model", "display_name": "B", "supported_intents": ["x"]},
        })
        results = cfg.catalog_query(kind="model")
        assert len(results) == 2


# ------------------------------------------------------------------
# T6.3: auto_join_roles with intent registry
# ------------------------------------------------------------------


class TestAutoJoinRolesWithIntents:
    def test_tag_based_auto_join_still_works(self):
        """Existing tag-based auto-join is unaffected."""
        cfg = _make_config(roles={"coding": ["existing-model"]})
        joined = cfg.auto_join_roles("new-model", ["coding"])
        assert "coding" in joined
        assert cfg.get_role_chain("coding") == ["existing-model", "new-model"]

    def test_without_intent_registry_preserves_behaviour(self):
        """Passing intent_registry=None behaves like before."""
        cfg = _make_config(roles={"coding": []})
        joined = cfg.auto_join_roles(
            "m1", ["coding"],
            intent_registry=None,
            supported_intents=["sar_processing"],
        )
        assert "coding" in joined
        # No intent-based joins should happen
        assert len(joined) == 1

    def test_intent_based_auto_join(self):
        """Models with supported_intents get auto-joined to the intent's target role."""
        registry = IntentRegistry()
        registry.register(IntentDefinition(
            name="sar_processing",
            description="SAR processing",
            target_role="reasoning",
            source="test-analyzer",
            priority=10,
        ))
        cfg = _make_config(roles={
            "coding": [],
            "reasoning": ["existing"],
        })
        joined = cfg.auto_join_roles(
            "sar-model", ["coding"],
            intent_registry=registry,
            supported_intents=["sar_processing"],
        )
        assert "coding" in joined
        assert "reasoning" in joined
        assert cfg.get_role_chain("reasoning") == ["existing", "sar-model"]

    def test_intent_auto_join_skips_unknown_intents(self):
        """Intents not in the registry are silently skipped."""
        registry = IntentRegistry()
        cfg = _make_config(roles={"coding": []})
        joined = cfg.auto_join_roles(
            "m1", ["coding"],
            intent_registry=registry,
            supported_intents=["unknown_intent"],
        )
        # Only tag-based join
        assert joined == ["coding"]

    def test_intent_auto_join_skips_nonexistent_roles(self):
        """If the intent maps to a role that doesn't exist in config, skip it."""
        registry = IntentRegistry()
        registry.register(IntentDefinition(
            name="special",
            description="Special",
            target_role="nonexistent_role",
            source="test",
            priority=10,
        ))
        cfg = _make_config(roles={"coding": []})
        joined = cfg.auto_join_roles(
            "m1", ["coding"],
            intent_registry=registry,
            supported_intents=["special"],
        )
        assert joined == ["coding"]

    def test_no_duplicate_join(self):
        """If a model is already in the chain, don't add it again."""
        registry = IntentRegistry()
        registry.register(IntentDefinition(
            name="code_gen",
            description="Code generation",
            target_role="coding",
            source="test",
            priority=10,
        ))
        cfg = _make_config(roles={"coding": ["m1"]})
        joined = cfg.auto_join_roles(
            "m1", [],
            intent_registry=registry,
            supported_intents=["code_gen"],
        )
        # Already in chain, should not be added again
        assert joined == []
        assert cfg.get_role_chain("coding") == ["m1"]


# ------------------------------------------------------------------
# T6.4: filter_chain_by_intent
# ------------------------------------------------------------------


class TestFilterChainByIntent:
    def _make_fabric(self, catalog: dict | None = None,
                     models: dict | None = None) -> ComputeFabric:
        cfg = _make_config(catalog=catalog, models=models)
        return ComputeFabric(cfg)

    def test_filters_to_supporting_models(self):
        fabric = self._make_fabric(catalog={
            "sar-model": {
                "kind": "model",
                "display_name": "SAR",
                "supported_intents": ["sar_processing"],
            },
            "code-model": {
                "kind": "model",
                "display_name": "Code",
                "supported_intents": ["code_generation"],
            },
        })
        result = fabric.filter_chain_by_intent(
            ["sar-model", "code-model"], "sar_processing",
        )
        assert result == ["sar-model"]

    def test_returns_full_chain_when_no_models_declare_intents(self):
        """Backwards compat: if no models declare supported_intents, return full chain."""
        fabric = self._make_fabric(catalog={
            "model-a": {"kind": "model", "display_name": "A"},
            "model-b": {"kind": "model", "display_name": "B"},
        })
        result = fabric.filter_chain_by_intent(
            ["model-a", "model-b"], "sar_processing",
        )
        assert result == ["model-a", "model-b"]

    def test_keeps_models_without_intents_when_mixed(self):
        """Models without supported_intents are kept alongside those that match."""
        fabric = self._make_fabric(catalog={
            "sar-model": {
                "kind": "model",
                "display_name": "SAR",
                "supported_intents": ["sar_processing"],
            },
            "plain-model": {
                "kind": "model",
                "display_name": "Plain",
            },
            "other-model": {
                "kind": "model",
                "display_name": "Other",
                "supported_intents": ["code_gen"],
            },
        })
        result = fabric.filter_chain_by_intent(
            ["sar-model", "plain-model", "other-model"], "sar_processing",
        )
        # sar-model matches; plain-model has no intents so kept;
        # other-model declares intents but not sar_processing so excluded
        assert result == ["sar-model", "plain-model"]

    def test_unknown_model_kept_in_chain(self):
        """Models not found in catalog are kept (don't break chains)."""
        fabric = self._make_fabric(catalog={})
        result = fabric.filter_chain_by_intent(
            ["unknown-model"], "sar_processing",
        )
        assert result == ["unknown-model"]

    def test_empty_chain(self):
        fabric = self._make_fabric(catalog={})
        result = fabric.filter_chain_by_intent([], "sar_processing")
        assert result == []

    def test_legacy_models_section(self):
        """Models in the legacy 'models' section are also checked."""
        fabric = self._make_fabric(
            models={
                "legacy-sar": {
                    "provider": "ollama",
                    "supported_intents": ["sar_processing"],
                },
                "legacy-plain": {
                    "provider": "ollama",
                },
            },
        )
        result = fabric.filter_chain_by_intent(
            ["legacy-sar", "legacy-plain"], "sar_processing",
        )
        # legacy-sar declares and matches; legacy-plain has no intents so kept
        # But since legacy-sar declares, any_declares is True
        assert "legacy-sar" in result
        assert "legacy-plain" in result
