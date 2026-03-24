"""Tests for the AuraRouter unified Python API (api.py).

Uses mocking to avoid real provider calls, file I/O, and database access.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.api import (
    APIConfig,
    AuraRouterAPI,
    CatalogEntry,
    HealthReport,
    LocalAsset,
    MCPToolStatus,
    ModelInfo,
    PrivacySummary,
    RoleChain,
    StorageInfo,
    TaskResult,
    TrafficSummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config_file(tmp_path: Path) -> Path:
    """Write a minimal auraconfig.yaml and return its path."""
    cfg_path = tmp_path / "auraconfig.yaml"
    cfg_path.write_text(
        "models:\n"
        "  test-model:\n"
        "    provider: ollama\n"
        "    model_name: test\n"
        "  cloud-model:\n"
        "    provider: openapi\n"
        "    model_name: gpt-4\n"
        "    tags: [remote]\n"
        "roles:\n"
        "  coding:\n"
        "    - test-model\n"
        "  reasoning:\n"
        "    - test-model\n"
        "  router:\n"
        "    - test-model\n"
        "  reviewer:\n"
        "    - test-model\n"
        "savings:\n"
        "  budget:\n"
        "    enabled: false\n"
        "  triage:\n"
        "    rules:\n"
        "      - max_complexity: 3\n"
        "        preferred_role: coding_lite\n"
        "        description: simple tasks\n"
        "    default_role: coding\n"
        "mcp_tools:\n"
        "  provider.generate:\n"
        "    enabled: true\n"
        "    description: Text generation\n"
        "  provider.list_models:\n"
        "    enabled: false\n"
    )
    return cfg_path


@pytest.fixture
def tmp_config(tmp_path):
    """Return the path to a minimal config file."""
    return _make_config_file(tmp_path)


@pytest.fixture
def api(tmp_config, tmp_path):
    """Return an AuraRouterAPI wired to a temp config."""
    cfg = APIConfig(
        config_path=str(tmp_config),
        enable_savings=False,
        enable_privacy=False,
        models_dir=str(tmp_path / "models"),
    )
    a = AuraRouterAPI(cfg)
    yield a
    a.close()


@pytest.fixture
def api_full(tmp_config, tmp_path):
    """Return an AuraRouterAPI with savings & privacy enabled."""
    cfg = APIConfig(
        config_path=str(tmp_config),
        enable_savings=True,
        enable_privacy=True,
        models_dir=str(tmp_path / "models"),
    )
    a = AuraRouterAPI(cfg)
    yield a
    a.close()


# ---------------------------------------------------------------------------
# 1. Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_init_default_config(self, tmp_config):
        """API can be initialized with an APIConfig."""
        cfg = APIConfig(
            config_path=str(tmp_config),
            enable_savings=False,
            enable_privacy=False,
        )
        api = AuraRouterAPI(cfg)
        assert api is not None
        api.close()

    def test_context_manager(self, tmp_config):
        """API works as a context manager."""
        cfg = APIConfig(
            config_path=str(tmp_config),
            enable_savings=False,
            enable_privacy=False,
        )
        with AuraRouterAPI(cfg) as a:
            assert a is not None
        # close() should be called automatically
        assert a._closed is True

    def test_double_close_is_safe(self, api):
        """Calling close() twice does not raise."""
        api.close()
        api.close()  # Should be a no-op


# ---------------------------------------------------------------------------
# 2. Model CRUD
# ---------------------------------------------------------------------------

class TestModelCRUD:
    def test_list_models(self, api):
        """list_models returns ModelInfo dataclasses."""
        models = api.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(isinstance(m, ModelInfo) for m in models)
        ids = [m.model_id for m in models]
        assert "test-model" in ids

    def test_get_model_found(self, api):
        """get_model returns ModelInfo for a known model."""
        m = api.get_model("test-model")
        assert m is not None
        assert m.model_id == "test-model"
        assert m.provider == "ollama"

    def test_get_model_not_found(self, api):
        """get_model returns None for unknown models."""
        assert api.get_model("nonexistent") is None

    def test_add_model(self, api):
        """add_model registers a new model."""
        info = api.add_model("new-model", {"provider": "ollama", "model_name": "x"})
        assert isinstance(info, ModelInfo)
        assert info.model_id == "new-model"
        assert api.get_model("new-model") is not None

    def test_update_model(self, api):
        """update_model modifies an existing model."""
        api.update_model("test-model", {"provider": "openapi", "model_name": "y"})
        m = api.get_model("test-model")
        assert m is not None
        assert m.provider == "openapi"

    def test_remove_model(self, api):
        """remove_model deletes a model and returns True."""
        assert api.remove_model("test-model") is True
        assert api.get_model("test-model") is None

    def test_remove_model_not_found(self, api):
        """remove_model returns False for unknown models."""
        assert api.remove_model("nonexistent") is False


# ---------------------------------------------------------------------------
# 3. Role chains
# ---------------------------------------------------------------------------

class TestRoleChains:
    def test_list_roles(self, api):
        """list_roles returns RoleChain dataclasses."""
        roles = api.list_roles()
        assert isinstance(roles, list)
        assert all(isinstance(r, RoleChain) for r in roles)
        names = [r.role for r in roles]
        assert "coding" in names

    def test_get_role_chain(self, api):
        """get_role_chain returns the chain for a known role."""
        rc = api.get_role_chain("coding")
        assert rc is not None
        assert rc.role == "coding"
        assert "test-model" in rc.chain

    def test_get_role_chain_missing(self, api):
        """get_role_chain returns None for unknown roles."""
        assert api.get_role_chain("nonexistent") is None

    def test_set_role_chain(self, api):
        """set_role_chain creates or replaces a role."""
        rc = api.set_role_chain("new-role", ["model-a", "model-b"])
        assert isinstance(rc, RoleChain)
        assert rc.chain == ["model-a", "model-b"]
        assert api.get_role_chain("new-role") is not None

    def test_remove_role(self, api):
        """remove_role deletes a role and returns True."""
        assert api.remove_role("coding") is True
        assert api.get_role_chain("coding") is None

    def test_remove_role_not_found(self, api):
        """remove_role returns False for unknown roles."""
        assert api.remove_role("nonexistent") is False


# ---------------------------------------------------------------------------
# 4. Task execution (mocked fabric)
# ---------------------------------------------------------------------------

class TestTaskExecution:
    @patch("aurarouter.api.time.monotonic", side_effect=[0.0, 1.0])
    def test_execute_direct(self, _mock_time, api):
        """execute_direct delegates to fabric and returns GenerateResult."""
        api._fabric.execute = MagicMock(return_value="Hello World")
        result = api.execute_direct("coding", "Write hello world")
        assert result.text == "Hello World"
        api._fabric.execute.assert_called_once()

    def test_execute_direct_none_result(self, api):
        """execute_direct handles None from fabric gracefully."""
        api._fabric.execute = MagicMock(return_value=None)
        result = api.execute_direct("coding", "fail")
        assert result.text == ""

    @patch("aurarouter.routing.analyze_intent")
    @patch("aurarouter.routing.generate_plan")
    @patch("aurarouter.routing.review_output")
    def test_execute_task(self, mock_review, mock_plan, mock_intent, api):
        """execute_task runs the full loop and returns TaskResult."""
        from aurarouter.routing import TriageResult, ReviewResult

        mock_intent.return_value = TriageResult(intent="SIMPLE_CODE", complexity=3)
        mock_plan.return_value = ["Step 1", "Step 2"]
        mock_review.return_value = ReviewResult(
            verdict="PASS", feedback="Looks good", correction_hints=[],
        )
        api._fabric.execute = MagicMock(return_value="output text")

        result = api.execute_task("Build a calculator", context="Python")
        assert isinstance(result, TaskResult)
        assert result.intent == "SIMPLE_CODE"
        assert result.complexity == 3
        assert result.plan == ["Step 1", "Step 2"]
        assert result.steps_executed == 2
        assert result.review_verdict == "PASS"
        assert "output text" in result.output

    @patch("aurarouter.routing.analyze_intent")
    @patch("aurarouter.routing.generate_plan")
    @patch("aurarouter.routing.review_output")
    def test_execute_task_callbacks(self, mock_review, mock_plan, mock_intent, api):
        """execute_task fires all callbacks."""
        from aurarouter.routing import TriageResult, ReviewResult

        mock_intent.return_value = TriageResult(intent="CODE", complexity=2)
        mock_plan.return_value = ["Do it"]
        mock_review.return_value = ReviewResult(
            verdict="PASS", feedback="ok", correction_hints=[],
        )
        api._fabric.execute = MagicMock(return_value="done")

        called = {"intent": False, "plan": False, "step": False, "review": False}

        def on_intent(i, c):
            called["intent"] = True

        def on_plan(p):
            called["plan"] = True

        def on_step(idx, desc):
            called["step"] = True

        def on_review(v, f):
            called["review"] = True

        api.execute_task(
            "test", on_intent=on_intent, on_plan=on_plan,
            on_step=on_step, on_review=on_review,
        )
        assert all(called.values()), f"Some callbacks not fired: {called}"


# ---------------------------------------------------------------------------
# 5. Traffic / privacy queries (mocked stores)
# ---------------------------------------------------------------------------

class TestMonitoring:
    def test_get_traffic_savings_disabled(self, api):
        """get_traffic returns empty summary when savings is off."""
        summary = api.get_traffic()
        assert isinstance(summary, TrafficSummary)
        assert summary.total_tokens == 0

    def test_get_traffic_with_stores(self, api_full):
        """get_traffic aggregates from usage store and cost engine."""
        api_full._usage_store.total_tokens = MagicMock(
            return_value={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        )
        api_full._usage_store.aggregate_tokens = MagicMock(return_value=[])
        api_full._cost_engine.total_spend = MagicMock(return_value=0.05)
        api_full._cost_engine.spend_by_provider = MagicMock(return_value={"ollama": 0.0})
        api_full._cost_engine.monthly_projection = MagicMock(return_value={})

        summary = api_full.get_traffic()
        assert summary.total_tokens == 150
        assert summary.total_spend == 0.05

    def test_get_privacy_events_disabled(self, api):
        """get_privacy_events returns empty when privacy is off."""
        summary = api.get_privacy_events()
        assert isinstance(summary, PrivacySummary)
        assert summary.total_events == 0

    def test_get_privacy_events_with_store(self, api_full):
        """get_privacy_events aggregates from privacy store."""
        api_full._privacy_store.query = MagicMock(return_value=[
            {"id": 1, "timestamp": "2025-01-01", "match_count": 2,
             "severities": ["high"], "pattern_names": ["API Key"]},
        ])
        api_full._privacy_store.summary = MagicMock(return_value={
            "total_events": 1,
            "by_severity": {"high": 1},
            "by_pattern": {"API Key": 1},
        })

        result = api_full.get_privacy_events()
        assert result.total_events == 1
        assert result.by_severity == {"high": 1}

    def test_get_budget_status_disabled(self, api):
        """get_budget_status returns None when savings is off."""
        assert api.get_budget_status() is None

    def test_get_budget_status_enabled(self, api_full):
        """get_budget_status returns dict when savings is on."""
        status = api_full.get_budget_status()
        assert isinstance(status, dict)
        assert "allowed" in status


# ---------------------------------------------------------------------------
# 6. Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_get_config_yaml(self, api):
        """get_config_yaml returns a YAML string."""
        yaml_str = api.get_config_yaml()
        assert isinstance(yaml_str, str)
        assert "models" in yaml_str

    def test_save_config(self, api, tmp_path):
        """save_config writes to disk and returns path."""
        out = tmp_path / "saved.yaml"
        result = api.save_config(str(out))
        assert result == str(out)
        assert out.is_file()

    def test_reload_config(self, api):
        """reload_config re-reads from disk without error."""
        api.reload_config()
        # Verify models still accessible after reload
        assert len(api.list_models()) >= 1

    def test_get_mcp_tools(self, api):
        """get_mcp_tools returns MCPToolStatus list."""
        tools = api.get_mcp_tools()
        assert isinstance(tools, list)
        assert all(isinstance(t, MCPToolStatus) for t in tools)
        names = [t.name for t in tools]
        assert "provider.generate" in names

    def test_set_mcp_tool(self, api):
        """set_mcp_tool toggles a tool's enabled state."""
        result = api.set_mcp_tool("provider.generate", False)
        assert isinstance(result, MCPToolStatus)
        assert result.enabled is False

    def test_get_system_settings(self, api):
        """get_system_settings returns a dict with known keys."""
        settings = api.get_system_settings()
        assert "environment" in settings
        assert settings["environment"] == "local"

    def test_get_environment(self, api):
        """get_environment returns the configured environment."""
        assert api.get_environment() == "local"

    def test_config_affects_other_nodes_local(self, api):
        """config_affects_other_nodes returns False for local."""
        assert api.config_affects_other_nodes() is False


# ---------------------------------------------------------------------------
# 7. Provider catalog
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_list_catalog(self, api):
        """list_catalog returns CatalogEntry dataclasses."""
        entries = api.list_catalog()
        assert isinstance(entries, list)
        assert all(isinstance(e, CatalogEntry) for e in entries)
        # Should include built-in providers
        names = [e.name for e in entries]
        assert "ollama" in names

    def test_add_remove_catalog_provider(self, api):
        """add/remove catalog provider round-trips correctly."""
        entry = api.add_catalog_provider("test-prov", "http://localhost:9999")
        assert isinstance(entry, CatalogEntry)
        assert entry.name == "test-prov"
        assert entry.source == "manual"

        assert api.remove_catalog_provider("test-prov") is True
        assert api.remove_catalog_provider("test-prov") is False


# ---------------------------------------------------------------------------
# 8. Semantic verbs / routing helpers
# ---------------------------------------------------------------------------

class TestRoutingHelpers:
    def test_resolve_role_synonym(self, api):
        """resolve_role_synonym maps synonyms to canonical names."""
        assert api.resolve_role_synonym("programming") == "coding"
        assert api.resolve_role_synonym("coding") == "coding"

    def test_get_missing_required_roles(self, api):
        """get_missing_required_roles detects unconfigured required roles."""
        # The fixture config has router, reasoning, coding -- all required
        missing = api.get_missing_required_roles()
        assert isinstance(missing, list)
        # All required roles are configured in fixture
        assert len(missing) == 0

    def test_get_missing_required_roles_after_removal(self, api):
        """get_missing_required_roles reports removed required roles."""
        api.remove_role("router")
        missing = api.get_missing_required_roles()
        assert "router" in missing

    def test_get_triage_rules_disabled(self, api):
        """get_triage_rules returns empty list when savings is off."""
        assert api.get_triage_rules() == []

    def test_get_triage_rules_enabled(self, api_full):
        """get_triage_rules returns rules when savings is on."""
        rules = api_full.get_triage_rules()
        assert isinstance(rules, list)
        assert len(rules) >= 1
        assert rules[0]["preferred_role"] == "coding_lite"


# ---------------------------------------------------------------------------
# 9. Storage helpers
# ---------------------------------------------------------------------------

class TestStorage:
    def test_list_local_assets_empty(self, api):
        """list_local_assets returns empty list for fresh storage."""
        assets = api.list_local_assets()
        assert isinstance(assets, list)
        assert all(isinstance(a, LocalAsset) for a in assets)

    def test_get_storage_info(self, api):
        """get_storage_info returns StorageInfo dataclass."""
        info = api.get_storage_info()
        assert isinstance(info, StorageInfo)
        assert info.total_files == 0

    def test_list_grid_models(self, api):
        """list_grid_models returns models with remote/grid tags."""
        # cloud-model has tags: [remote] in the fixture config
        grid = api.list_grid_models()
        ids = [m.model_id for m in grid]
        assert "cloud-model" in ids


# ---------------------------------------------------------------------------
# 10. Dataclass contracts
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_task_result_defaults(self):
        """TaskResult has sensible defaults."""
        r = TaskResult(output="hello")
        assert r.output == "hello"
        assert r.plan == []
        assert r.total_elapsed == 0.0

    def test_model_info_defaults(self):
        """ModelInfo has sensible defaults."""
        m = ModelInfo(model_id="x")
        assert m.provider == ""
        assert m.config == {}

    def test_api_config_defaults(self):
        """APIConfig defaults are correct."""
        c = APIConfig()
        assert c.environment == "local"
        assert c.enable_savings is True
        assert c.enable_privacy is True
        assert c.config_path is None
