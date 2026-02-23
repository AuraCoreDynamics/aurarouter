"""Integration tests for session-aware fabric, config, and MCP tools."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult
from aurarouter.sessions.models import Session, Message
from aurarouter.sessions.store import SessionStore
from aurarouter.sessions.manager import SessionManager


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfigSessions:
    def test_get_sessions_config_default(self):
        cfg = ConfigLoader(allow_missing=True)
        assert cfg.get_sessions_config() == {}

    def test_get_sessions_config_enabled(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config["sessions"] = {
            "enabled": True,
            "condensation_threshold": 0.7,
            "auto_gist": False,
        }
        result = cfg.get_sessions_config()
        assert result["enabled"] is True
        assert result["condensation_threshold"] == 0.7
        assert result["auto_gist"] is False


# ---------------------------------------------------------------------------
# Fabric execute_session tests
# ---------------------------------------------------------------------------


def _make_fabric(model_response="test response", context_limit=0):
    """Create a ComputeFabric with a mock provider."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "test_model": {
                "provider": "ollama",
                "model_name": "test",
                "endpoint": "http://localhost:11434/api/generate",
                "context_limit": context_limit,
            },
        },
        "roles": {
            "coding": ["test_model"],
        },
    }
    fabric = ComputeFabric(cfg)
    return fabric, cfg


class TestExecuteSession:
    def test_basic(self):
        fabric, cfg = _make_fabric()
        session = Session()

        mock_result = GenerateResult(
            text="Generated response",
            input_tokens=20,
            output_tokens=10,
            model_id="test_model",
            provider="ollama",
        )

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = fabric.execute_session(
                role="coding",
                session=session,
                message="Write a function",
            )

        assert result.text == "Generated response"
        # Session should have user message + assistant message
        assert len(session.history) == 2
        assert session.history[0].role == "user"
        assert session.history[0].content == "Write a function"
        assert session.history[1].role == "assistant"

    def test_gist_extraction(self):
        fabric, cfg = _make_fabric()
        session = Session()

        mock_result = GenerateResult(
            text="Here is the code.\n---GIST---\nProvided fibonacci implementation.",
            input_tokens=20,
            output_tokens=10,
        )

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = fabric.execute_session(
                role="coding",
                session=session,
                message="Write fibonacci",
            )

        assert result.text == "Here is the code."
        assert result.gist == "Provided fibonacci implementation."
        assert len(session.shared_context) == 1
        assert session.shared_context[0].summary == "Provided fibonacci implementation."

    def test_fallback_on_first_model_failure(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "model_a": {
                    "provider": "ollama",
                    "model_name": "a",
                    "endpoint": "http://localhost:11434/api/generate",
                },
                "model_b": {
                    "provider": "ollama",
                    "model_name": "b",
                    "endpoint": "http://localhost:11434/api/generate",
                },
            },
            "roles": {"coding": ["model_a", "model_b"]},
        }
        fabric = ComputeFabric(cfg)
        session = Session()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Model A failed")
            return GenerateResult(text="Model B response", input_tokens=5, output_tokens=3)

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", side_effect=side_effect):
            result = fabric.execute_session(
                role="coding",
                session=session,
                message="test",
            )

        assert result.text == "Model B response"

    def test_execute_unchanged(self):
        """Verify existing execute() still works exactly as before."""
        fabric, cfg = _make_fabric()

        mock_result = GenerateResult(text="stateless response", input_tokens=10, output_tokens=5)
        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_usage", return_value=mock_result):
            result = fabric.execute("coding", "Hello world")

        assert result == "stateless response"


# ---------------------------------------------------------------------------
# SessionManager + Fabric end-to-end
# ---------------------------------------------------------------------------


class TestEndToEndSessionLifecycle:
    def test_full_lifecycle(self, tmp_path):
        """create -> message -> message -> status -> delete"""
        store = SessionStore(db_path=tmp_path / "sessions.db")

        # Mock generate_fn for condensation
        def mock_generate(role, prompt):
            return "Summary of conversation."

        manager = SessionManager(
            store=store,
            auto_gist=True,
            generate_fn=mock_generate,
        )

        # Create session
        session = manager.create_session(role="coding", context_limit=10000)
        assert session.session_id

        # Add messages
        session = manager.add_user_message(session, "Write fibonacci", tokens=5)
        assert len(session.history) == 1

        session = manager.add_assistant_message(
            session,
            "Here is fibonacci.\n---GIST---\nProvided fibonacci function.",
            model_id="test",
            tokens=15,
        )
        assert len(session.history) == 2
        assert len(session.shared_context) == 1

        session = manager.add_user_message(session, "Add memoization", tokens=5)
        assert len(session.history) == 3

        # Check status
        assert not manager.check_pressure(session)

        # List sessions
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session.session_id

        # Delete
        assert manager.delete_session(session.session_id) is True
        assert manager.get_session(session.session_id) is None

    def test_condensation_flow(self, tmp_path):
        """Test that condensation works when pressure is high."""
        store = SessionStore(db_path=tmp_path / "sessions.db")

        def mock_generate(role, prompt):
            return "Condensed summary."

        manager = SessionManager(
            store=store,
            condensation_threshold=0.8,
            generate_fn=mock_generate,
        )

        session = manager.create_session(context_limit=1000)
        # Simulate high token usage
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(Message(role=role, content=f"msg {i}", tokens=150))
        session.token_stats.input_tokens = 850  # High pressure
        store.save(session)

        assert manager.check_pressure(session)  # 0.85 >= 0.8
        session = manager.condense(session)
        assert len(session.history) == 2  # Kept last 2
        assert len(session.shared_context) == 1  # Condensation gist
