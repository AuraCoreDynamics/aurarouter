"""Cross-stack validation tests.

End-to-end tests verifying that catalog CRUD, analyzer selection, and
kind-based queries work together coherently through real code paths.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.mcp_tools import (
    catalog_get_artifact,
    catalog_list_artifacts,
    catalog_register_artifact,
    catalog_remove_artifact,
    get_active_analyzer,
    route_task,
    set_active_analyzer,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_config(**extra) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
            "reviewer": ["m1"],
        },
        "catalog": {},
        "system": {},
        **extra,
    }
    return cfg


# ------------------------------------------------------------------
# Test: register all three kinds, then query per kind
# ------------------------------------------------------------------


class TestRegisterAndQueryByKind:
    def test_register_all_kinds_then_count_per_kind(self):
        """Register model, service, analyzer via catalog_register_artifact,
        then verify catalog_list_artifacts returns correct counts per kind."""
        cfg = _make_config()

        # Register one of each kind
        for kind, name in [("model", "k-model"), ("service", "k-svc"), ("analyzer", "k-analyzer")]:
            result = json.loads(catalog_register_artifact(
                cfg, artifact_id=name, kind=kind, display_name=f"Test {kind}",
            ))
            assert result["success"] is True

        # Query by kind and verify counts
        models = json.loads(catalog_list_artifacts(cfg, kind="model"))
        model_ids = [m["artifact_id"] for m in models]
        assert "k-model" in model_ids
        assert "m1" in model_ids  # legacy model
        assert "k-svc" not in model_ids
        assert "k-analyzer" not in model_ids

        services = json.loads(catalog_list_artifacts(cfg, kind="service"))
        service_ids = [s["artifact_id"] for s in services]
        assert service_ids == ["k-svc"]

        analyzers = json.loads(catalog_list_artifacts(cfg, kind="analyzer"))
        analyzer_ids = [a["artifact_id"] for a in analyzers]
        assert analyzer_ids == ["k-analyzer"]

        # All together
        all_items = json.loads(catalog_list_artifacts(cfg))
        assert len(all_items) == 4  # 3 registered + 1 legacy


# ------------------------------------------------------------------
# Test: set active analyzer, route_task attempts analyzer path
# ------------------------------------------------------------------


class TestActiveAnalyzerRouteTask:
    def test_set_analyzer_then_route_task_attempts_analyzer(self):
        """Set active analyzer to a remote endpoint, verify route_task
        attempts the remote analyzer path (mocked fabric)."""
        cfg = _make_config()

        # Register a remote analyzer
        cfg.catalog_set("remote-test", {
            "kind": "analyzer",
            "display_name": "Remote Test",
            "mcp_endpoint": "http://analyzer:9090/mcp",
            "mcp_tool_name": "classify",
        })
        set_active_analyzer(cfg, analyzer_id="remote-test")

        # Confirm it's active
        active = json.loads(get_active_analyzer(cfg))
        assert active["active_analyzer"] == "remote-test"

        # Mock fabric and remote analyzer
        from aurarouter.savings.models import GenerateResult

        mock_fabric = MagicMock()
        mock_fabric.config = cfg
        mock_fabric.execute.return_value = GenerateResult(text="analyzer routed output")
        mock_fabric.get_max_review_iterations.return_value = 0

        remote_result = {"ranked_models": ["m1"], "role": "coding"}

        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = remote_result

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            output = route_task(mock_fabric, None, task="test task", config=cfg)
            assert output == "analyzer routed output"
            # fabric.execute should have been called with the role from remote result
            mock_fabric.execute.assert_called_once_with("coding", "TASK: test task")


# ------------------------------------------------------------------
# Full CRUD round-trip
# ------------------------------------------------------------------


class TestCatalogCrudRoundTrip:
    def test_register_query_remove_lifecycle(self):
        """Register all three kinds, query, remove."""
        cfg = _make_config()

        # Register
        for kind, aid in [("model", "my-model"), ("service", "my-svc"), ("analyzer", "my-analyzer")]:
            result = json.loads(catalog_register_artifact(
                cfg, artifact_id=aid, kind=kind,
                display_name=f"My {kind.title()}", provider="test",
            ))
            assert result["success"] is True

        # All should be retrievable
        for aid in ("my-model", "my-svc", "my-analyzer"):
            item = json.loads(catalog_get_artifact(cfg, aid))
            assert item["artifact_id"] == aid
            assert "error" not in item

        # Remove all
        for aid in ("my-model", "my-svc", "my-analyzer"):
            result = json.loads(catalog_remove_artifact(cfg, aid))
            assert result["success"] is True

        # Only legacy m1 remains
        remaining = json.loads(catalog_list_artifacts(cfg))
        remaining_ids = [item["artifact_id"] for item in remaining]
        assert remaining_ids == ["m1"]
