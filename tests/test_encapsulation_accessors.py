"""Tests for public accessors added to fix encapsulation violations (Task Group 2).

Covers:
- ComputeFabric.config property
- ComputeFabric.get_max_review_iterations()
- ComputeFabric.get_local_chain()
- ComputeFabric.set_routing_advisors()
- SessionManager.save_session()
- SessionManager.auto_gist property
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.models import Session, TokenStats
from aurarouter.sessions.store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(models: dict, roles: dict, execution: dict | None = None) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    if execution:
        cfg.config["execution"] = execution
    return cfg


def _make_fabric(models: dict, roles: dict, execution: dict | None = None) -> ComputeFabric:
    return ComputeFabric(_make_config(models, roles, execution))


# ---------------------------------------------------------------------------
# ComputeFabric.config property
# ---------------------------------------------------------------------------

class TestFabricConfigProperty:
    def test_returns_config_loader(self):
        cfg = _make_config({}, {})
        fabric = ComputeFabric(cfg)
        assert fabric.config is cfg

    def test_config_reflects_update(self):
        cfg1 = _make_config({}, {})
        cfg2 = _make_config({"m1": {"provider": "ollama"}}, {})
        fabric = ComputeFabric(cfg1)
        fabric.update_config(cfg2)
        assert fabric.config is cfg2


# ---------------------------------------------------------------------------
# ComputeFabric.get_max_review_iterations
# ---------------------------------------------------------------------------

class TestGetMaxReviewIterations:
    def test_default_value(self):
        fabric = _make_fabric({}, {})
        assert fabric.get_max_review_iterations() == 3  # default

    def test_custom_value(self):
        fabric = _make_fabric({}, {}, execution={"max_review_iterations": 5})
        assert fabric.get_max_review_iterations() == 5

    def test_zero_disables_loop(self):
        fabric = _make_fabric({}, {}, execution={"max_review_iterations": 0})
        assert fabric.get_max_review_iterations() == 0


# ---------------------------------------------------------------------------
# ComputeFabric.get_local_chain
# ---------------------------------------------------------------------------

class TestGetLocalChain:
    def test_filters_out_cloud_models(self):
        fabric = _make_fabric(
            models={
                "local_ollama": {"provider": "ollama", "model_name": "l1", "endpoint": "http://x"},
                "cloud_google": {"provider": "google", "model_name": "gemini-pro", "api_key": "k"},
                "local_llamacpp": {"provider": "llamacpp", "model_path": "/m.gguf"},
            },
            roles={"coding": ["local_ollama", "cloud_google", "local_llamacpp"]},
        )
        local_chain = fabric.get_local_chain("coding")
        assert local_chain == ["local_ollama", "local_llamacpp"]
        assert "cloud_google" not in local_chain

    def test_returns_empty_when_all_cloud(self):
        fabric = _make_fabric(
            models={
                "g1": {"provider": "google", "model_name": "gemini-pro", "api_key": "k"},
                "c1": {"provider": "claude", "model_name": "opus", "api_key": "k"},
            },
            roles={"coding": ["g1", "c1"]},
        )
        assert fabric.get_local_chain("coding") == []

    def test_returns_all_when_all_local(self):
        fabric = _make_fabric(
            models={
                "m1": {"provider": "ollama", "model_name": "l1", "endpoint": "http://x"},
                "m2": {"provider": "llamacpp", "model_path": "/m.gguf"},
            },
            roles={"coding": ["m1", "m2"]},
        )
        assert fabric.get_local_chain("coding") == ["m1", "m2"]

    def test_empty_role_returns_empty(self):
        fabric = _make_fabric(models={}, roles={"coding": []})
        assert fabric.get_local_chain("coding") == []

    def test_undefined_role_returns_empty(self):
        fabric = _make_fabric(models={}, roles={})
        assert fabric.get_local_chain("nonexistent") == []

    def test_explicit_hosting_tier_overrides_provider_default(self):
        """An ollama model explicitly marked as 'cloud' should be filtered out."""
        fabric = _make_fabric(
            models={
                "m1": {
                    "provider": "ollama",
                    "model_name": "l1",
                    "endpoint": "http://x",
                    "hosting_tier": "cloud",
                },
                "m2": {"provider": "ollama", "model_name": "l2", "endpoint": "http://y"},
            },
            roles={"coding": ["m1", "m2"]},
        )
        assert fabric.get_local_chain("coding") == ["m2"]

    def test_cloud_model_with_on_prem_tier_included(self):
        """A google model explicitly marked as 'on-prem' should be included."""
        fabric = _make_fabric(
            models={
                "g1": {
                    "provider": "google",
                    "model_name": "gemini-pro",
                    "api_key": "k",
                    "hosting_tier": "on-prem",
                },
            },
            roles={"coding": ["g1"]},
        )
        assert fabric.get_local_chain("coding") == ["g1"]


# ---------------------------------------------------------------------------
# ComputeFabric.set_routing_advisors
# ---------------------------------------------------------------------------

class TestSetRoutingAdvisors:
    def test_sets_routing_advisors(self):
        fabric = _make_fabric({}, {})
        mock_registry = MagicMock()
        fabric.set_routing_advisors(mock_registry)
        # Verify it was set by checking the internal attribute
        assert fabric._routing_advisors is mock_registry

    def test_replaces_existing_advisors(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        fabric = _make_fabric({}, {})
        fabric.set_routing_advisors(mock1)
        assert fabric._routing_advisors is mock1
        fabric.set_routing_advisors(mock2)
        assert fabric._routing_advisors is mock2

    def test_set_none_clears_advisors(self):
        fabric = _make_fabric({}, {})
        fabric.set_routing_advisors(MagicMock())
        fabric.set_routing_advisors(None)
        assert fabric._routing_advisors is None


# ---------------------------------------------------------------------------
# SessionManager.save_session
# ---------------------------------------------------------------------------

class TestSessionManagerSaveSession:
    def test_save_session_persists(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        manager = SessionManager(store=store)
        session = manager.create_session(role="coding")

        # Modify the session externally
        from aurarouter.sessions.models import Message
        session.add_message(Message(role="user", content="hello"))

        # Save via the public accessor
        manager.save_session(session)

        # Verify it's persisted
        loaded = store.load(session.session_id)
        assert loaded is not None
        assert len(loaded.history) == 1
        assert loaded.history[0].content == "hello"


# ---------------------------------------------------------------------------
# SessionManager.auto_gist property
# ---------------------------------------------------------------------------

class TestSessionManagerAutoGist:
    def test_default_true(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        manager = SessionManager(store=store)
        assert manager.auto_gist is True

    def test_explicit_true(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        manager = SessionManager(store=store, auto_gist=True)
        assert manager.auto_gist is True

    def test_explicit_false(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        manager = SessionManager(store=store, auto_gist=False)
        assert manager.auto_gist is False

    def test_reflects_constructor_value(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        for value in (True, False):
            manager = SessionManager(store=store, auto_gist=value)
            assert manager.auto_gist is value
