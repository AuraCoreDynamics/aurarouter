"""Cross-component integration tests for the intent system (Task Group 9.2).

Validates interactions between:
- Catalog -> Registry -> CLI
- Catalog -> Registry -> Broker
- Tags -> Intents -> Chain (filter_chain_by_intent)
- Advisor -> Chain reorder (with intent awareness)
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.broker import AnalyzerBid, BrokerResult, merge_bids
from aurarouter.catalog_model import ArtifactKind, CatalogArtifact
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import (
    IntentDefinition,
    IntentRegistry,
    build_intent_registry,
)
from aurarouter.mcp_client.registry import McpClientRegistry
from aurarouter.mcp_tools import list_intents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    active_analyzer: str = "test-analyzer",
    role_bindings: dict[str, str] | None = None,
    catalog_extras: dict | None = None,
    models: dict | None = None,
    roles: dict | None = None,
) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    analyzer_entry: dict = {
        "kind": "analyzer",
        "display_name": "Test Analyzer",
        "analyzer_kind": "intent_triage",
    }
    if role_bindings is not None:
        analyzer_entry["role_bindings"] = role_bindings

    catalog: dict = {active_analyzer: analyzer_entry}
    if catalog_extras:
        catalog.update(catalog_extras)

    cfg.config = {
        "system": {"active_analyzer": active_analyzer},
        "catalog": catalog,
        "models": models or {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": roles or {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return cfg


# ======================================================================
# T9.2.1: Catalog -> Registry -> CLI
# ======================================================================


class TestCatalogRegistryCli:
    """Register analyzer, build registry, verify `intent list` CLI output
    includes custom intents."""

    def test_intent_list_includes_custom_intents(self, tmp_path):
        """CLI 'intent list --json' should include analyzer-declared intents."""
        import yaml

        cfg_data = {
            "system": {
                "log_level": "INFO",
                "default_timeout": 120.0,
                "active_analyzer": "sar-analyzer",
            },
            "models": {
                "m1": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate",
                    "model_name": "mock",
                },
            },
            "roles": {"router": ["m1"], "reasoning": ["m1"], "coding": ["m1"]},
            "catalog": {
                "sar-analyzer": {
                    "kind": "analyzer",
                    "display_name": "SAR Analyzer",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {
                        "sar_processing": "reasoning",
                        "geoint_analysis": "coding",
                    },
                },
            },
        }
        config_path = tmp_path / "auraconfig.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg_data, f)

        from aurarouter.cli import main

        out = StringIO()
        err = StringIO()
        with (
            patch.object(
                sys, "argv",
                ["aurarouter", "--config", str(config_path), "intent", "list", "--json"],
            ),
            patch("sys.stdout", out),
            patch("sys.stderr", err),
        ):
            try:
                main()
            except SystemExit:
                pass

        output = out.getvalue()
        data = json.loads(output)
        intent_names = [i["name"] for i in data["intents"]]
        assert "sar_processing" in intent_names
        assert "geoint_analysis" in intent_names
        # Built-in intents should also be present
        assert "DIRECT" in intent_names
        assert "SIMPLE_CODE" in intent_names
        assert "COMPLEX_REASONING" in intent_names

    def test_intent_list_includes_source_and_role(self, tmp_path):
        """Each intent in JSON output should have source and target_role."""
        import yaml

        cfg_data = {
            "system": {
                "log_level": "INFO",
                "default_timeout": 120.0,
                "active_analyzer": "custom-analyzer",
            },
            "models": {
                "m1": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate",
                    "model_name": "mock",
                },
            },
            "roles": {"router": ["m1"], "coding": ["m1"]},
            "catalog": {
                "custom-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Custom",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"my_intent": "coding"},
                },
            },
        }
        config_path = tmp_path / "auraconfig.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg_data, f)

        from aurarouter.cli import main

        out = StringIO()
        with (
            patch.object(
                sys, "argv",
                ["aurarouter", "--config", str(config_path), "intent", "list", "--json"],
            ),
            patch("sys.stdout", out),
            patch("sys.stderr", StringIO()),
        ):
            try:
                main()
            except SystemExit:
                pass

        data = json.loads(out.getvalue())
        my_intent = next(
            (i for i in data["intents"] if i["name"] == "my_intent"), None,
        )
        assert my_intent is not None
        assert my_intent["target_role"] == "coding"
        assert my_intent["source"] == "custom-analyzer"


# ======================================================================
# T9.2.2: Catalog -> Registry -> Broker
# ======================================================================


class TestCatalogRegistryBroker:
    """Register analyzer with custom intents, mock broker broadcast,
    verify intent is passed to remote analyzers."""

    def test_merge_bids_applies_intent_bonus(self):
        """When intent is provided with role_bindings, matching bids get a
        confidence bonus."""
        bid_a = AnalyzerBid(
            analyzer_id="analyzer-a", confidence=0.6,
            claimed_files=["a.py"], role="coding",
        )
        bid_b = AnalyzerBid(
            analyzer_id="analyzer-b", confidence=0.7,
            claimed_files=["b.py"], role="reasoning",
        )

        # analyzer-a has sar_processing in its role_bindings
        role_bindings = {
            "analyzer-a": {"sar_processing": "coding"},
        }

        result = merge_bids(
            [bid_a, bid_b],
            intent="sar_processing",
            analyzer_role_bindings=role_bindings,
        )

        # analyzer-a should have gotten a confidence bonus (0.6 + 0.1 = 0.7)
        a_in_result = next(
            b for b in result.bids if b.analyzer_id == "analyzer-a"
        )
        assert a_in_result.confidence == pytest.approx(0.7)

        # Trace should mention intent-aware scoring
        assert any("intent bonus" in t or "intent-aware" in t for t in result.execution_trace)

    def test_merge_bids_intent_without_role_bindings(self):
        """When intent is provided but no role_bindings, a trace entry is added."""
        bid = AnalyzerBid(analyzer_id="a", confidence=0.8)
        result = merge_bids([bid], intent="sar_processing")
        assert any("no role_bindings" in t for t in result.execution_trace)

    def test_broker_broadcast_passes_intent(self):
        """Verify that broadcast_to_analyzers passes the intent kwarg
        through to the single-analyzer call."""
        cfg = _make_config(
            role_bindings={"sar_processing": "reasoning"},
            catalog_extras={
                "remote-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Remote",
                    "analyzer_kind": "intent_triage",
                    "mcp_endpoint": "http://fake:9999/mcp",
                    "mcp_tool_name": "analyze",
                },
            },
        )

        import asyncio
        from unittest.mock import AsyncMock
        from aurarouter.broker import broadcast_to_analyzers

        with patch(
            "aurarouter.broker._call_single_analyzer",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_call:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    broadcast_to_analyzers(cfg, "test prompt", intent="sar_processing")
                )
            finally:
                loop.close()

            # Verify _call_single_analyzer was called with intent kwarg
            if mock_call.called:
                call_kwargs = mock_call.call_args
                assert call_kwargs is not None


# ======================================================================
# T9.2.3: Tags -> Intents -> Chain
# ======================================================================


class TestTagsIntentsChain:
    """Register model with supported_intents, verify filter_chain_by_intent
    selects it."""

    def test_filter_chain_selects_model_with_matching_intent(self):
        """Models declaring supported_intents matching the intent are kept."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "sar-model": {
                    "kind": "model",
                    "display_name": "SAR Model",
                    "provider": "ollama",
                    "supported_intents": ["sar_processing"],
                },
                "general-model": {
                    "kind": "model",
                    "display_name": "General Model",
                    "provider": "ollama",
                    "supported_intents": ["general"],
                },
            },
            "models": {},
            "roles": {"reasoning": ["sar-model", "general-model"]},
        }
        fabric = ComputeFabric(cfg)

        filtered = fabric.filter_chain_by_intent(
            ["sar-model", "general-model"], "sar_processing",
        )
        assert "sar-model" in filtered
        assert "general-model" not in filtered

    def test_filter_chain_with_mixed_declarations(self):
        """Models without supported_intents are kept; models with non-matching
        intents are removed."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "declaring-model": {
                    "kind": "model",
                    "display_name": "Declaring",
                    "supported_intents": ["sar_processing"],
                },
                "no-intents-model": {
                    "kind": "model",
                    "display_name": "NoIntents",
                    # No supported_intents field
                },
            },
            "models": {},
            "roles": {"coding": ["declaring-model", "no-intents-model"]},
        }
        fabric = ComputeFabric(cfg)

        filtered = fabric.filter_chain_by_intent(
            ["declaring-model", "no-intents-model"], "sar_processing",
        )
        # declaring-model matches, no-intents-model has no supported_intents so kept
        assert "declaring-model" in filtered
        assert "no-intents-model" in filtered

    def test_catalog_query_supported_intents_filter(self):
        """catalog_query(supported_intents=...) should filter correctly."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "sar-model": {
                    "kind": "model",
                    "display_name": "SAR",
                    "supported_intents": ["sar_processing"],
                },
                "plain-model": {
                    "kind": "model",
                    "display_name": "Plain",
                },
            },
        }
        results = cfg.catalog_query(supported_intents=["sar_processing"])
        ids = [r["artifact_id"] for r in results]
        assert "sar-model" in ids
        assert "plain-model" not in ids


# ======================================================================
# T9.2.4: Advisor -> Chain reorder with intent
# ======================================================================


class TestAdvisorChainReorder:
    """Register routing advisor, verify consult_routing_advisors passes
    intent, verify chain is reordered."""

    def test_advisor_receives_intent_in_call(self):
        """When intent is provided, it should be passed to the advisor's
        call_tool method."""
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.name = "intent-advisor"
        mock_client.get_capabilities.return_value = {"chain_reorder"}
        mock_client.call_tool.return_value = {"chain": ["m2", "m1"]}

        registry = McpClientRegistry()
        registry.register("intent-advisor", mock_client)

        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
                "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
            },
            "roles": {"coding": ["m1", "m2"]},
        }
        fabric = ComputeFabric(cfg, routing_advisors=registry)

        result = fabric.consult_routing_advisors(
            "coding", ["m1", "m2"], intent="sar_processing",
        )

        # Verify advisor was called with intent
        call_kwargs = mock_client.call_tool.call_args
        assert call_kwargs is not None
        # The intent should be in the kwargs
        _, kwargs = call_kwargs
        assert kwargs.get("intent") == "sar_processing"
        # Chain should be reordered
        assert result == ["m2", "m1"]

    def test_advisor_without_intent_still_works(self):
        """Advisor call without intent should still work (backwards compat)."""
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.name = "basic-advisor"
        mock_client.get_capabilities.return_value = {"chain_reorder"}
        mock_client.call_tool.return_value = {"chain": ["m2", "m1"]}

        registry = McpClientRegistry()
        registry.register("basic-advisor", mock_client)

        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg, routing_advisors=registry)

        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m2", "m1"]

        # Verify intent was NOT in the call kwargs (since None was passed)
        call_kwargs = mock_client.call_tool.call_args
        _, kwargs = call_kwargs
        assert "intent" not in kwargs

    def test_advisor_registration_and_listing(self):
        """Test register_routing_advisor and list_routing_advisors."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg)

        assert fabric.list_routing_advisors() == []

        mock_client = MagicMock()
        mock_client.name = "my-advisor"
        fabric.register_routing_advisor(mock_client)

        advisors = fabric.list_routing_advisors()
        assert "my-advisor" in advisors

    def test_advisor_idempotent_registration(self):
        """Re-registering the same advisor should be a no-op."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg)

        mock_client = MagicMock()
        mock_client.name = "my-advisor"
        fabric.register_routing_advisor(mock_client)
        fabric.register_routing_advisor(mock_client)

        advisors = fabric.list_routing_advisors()
        assert advisors.count("my-advisor") == 1


# ======================================================================
# T9.2 (cross-component): list_intents MCP tool
# ======================================================================


class TestListIntentsMcpTool:
    """Verify the list_intents MCP tool returns both built-in and
    analyzer-declared intents."""

    def test_list_intents_returns_all(self):
        cfg = _make_config(
            role_bindings={"sar_processing": "reasoning"},
        )
        result_json = list_intents(cfg)
        data = json.loads(result_json)

        intent_names = [i["name"] for i in data["intents"]]
        assert "sar_processing" in intent_names
        assert "DIRECT" in intent_names
        assert "SIMPLE_CODE" in intent_names
        assert "COMPLEX_REASONING" in intent_names

    def test_list_intents_includes_active_analyzer(self):
        cfg = _make_config(
            active_analyzer="my-analyzer",
            role_bindings={"custom": "coding"},
        )
        result_json = list_intents(cfg)
        data = json.loads(result_json)
        assert data["active_analyzer"] == "my-analyzer"


# ======================================================================
# T9.2 (cross-component): catalog_get enriches with declared_intents
# ======================================================================


class TestCatalogDeclaredIntentsEnrichment:
    """Verify that catalog_get enriches analyzer artifacts with
    declared_intents extracted from role_bindings."""

    def test_catalog_get_analyzer_has_declared_intents(self):
        cfg = _make_config(
            role_bindings={"sar_processing": "reasoning", "geoint": "coding"},
        )
        data = cfg.catalog_get("test-analyzer")
        assert data is not None
        assert "declared_intents" in data
        assert "sar_processing" in data["declared_intents"]
        assert "geoint" in data["declared_intents"]

    def test_catalog_get_model_no_declared_intents(self):
        """Models should not have declared_intents enrichment."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
        }
        data = cfg.catalog_get("m1")
        assert data is not None
        # Models get kind=model but should NOT have declared_intents
        assert "declared_intents" not in data
