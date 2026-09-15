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
    """Tests for the refactored execute_session() that takes messages, not Session."""

    def test_basic(self):
        """execute_session accepts messages list and returns GenerateResult."""
        fabric, cfg = _make_fabric()

        mock_result = GenerateResult(
            text="Generated response",
            input_tokens=20,
            output_tokens=10,
            model_id="test_model",
            provider="ollama",
        )

        messages = [{"role": "user", "content": "Write a function"}]

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = fabric.execute_session(
                role="coding",
                messages=messages,
            )

        assert result.text == "Generated response"
        assert result.model_id == "test_model"
        assert result.output_tokens == 10

    def test_gist_extraction(self):
        """execute_session extracts gist from response without touching session."""
        fabric, cfg = _make_fabric()

        mock_result = GenerateResult(
            text="Here is the code.\n---GIST---\nProvided fibonacci implementation.",
            input_tokens=20,
            output_tokens=10,
        )

        messages = [{"role": "user", "content": "Write fibonacci"}]

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = fabric.execute_session(
                role="coding",
                messages=messages,
            )

        assert result.text == "Here is the code."
        assert result.gist == "Provided fibonacci implementation."

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

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Model A failed")
            return GenerateResult(text="Model B response", input_tokens=5, output_tokens=3)

        messages = [{"role": "user", "content": "test"}]

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", side_effect=side_effect):
            result = fabric.execute_session(
                role="coding",
                messages=messages,
            )

        assert result.text == "Model B response"

    def test_does_not_mutate_session(self):
        """execute_session must not reference or mutate Session objects."""
        fabric, cfg = _make_fabric()
        session = Session()
        session.add_message(Message(role="user", content="Hello"))

        mock_result = GenerateResult(
            text="Response\n---GIST---\nGist text.",
            input_tokens=10,
            output_tokens=5,
        )

        # Pass messages from session, but session should not be mutated
        messages = session.get_messages_as_dicts()
        history_len_before = len(session.history)
        gist_count_before = len(session.shared_context)

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = fabric.execute_session(role="coding", messages=messages)

        assert result.text == "Response"
        assert result.gist == "Gist text."
        # Session should be UNCHANGED -- fabric no longer mutates it
        assert len(session.history) == history_len_before
        assert len(session.shared_context) == gist_count_before

    def test_execute_unchanged(self):
        """Verify existing execute() still works exactly as before."""
        fabric, cfg = _make_fabric()

        mock_result = GenerateResult(text="stateless response", input_tokens=10, output_tokens=5)
        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_usage", return_value=mock_result):
            result = fabric.execute("coding", "Hello world")

        assert result is not None
        assert result.text == "stateless response"


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


# ---------------------------------------------------------------------------
# T5.5: send_message() full integration tests
# ---------------------------------------------------------------------------


class TestSendMessage:
    """Tests for SessionManager.send_message() — the sole session entry point."""

    def _make_manager_and_fabric(self, tmp_path, auto_gist=True, context_limit=10000):
        """Helper: create a fabric + session manager wired together."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "test_model": {
                    "provider": "ollama",
                    "model_name": "test",
                    "endpoint": "http://localhost:11434/api/generate",
                },
            },
            "roles": {
                "coding": ["test_model"],
                "summarizer": ["test_model"],
            },
        }
        fabric = ComputeFabric(cfg)
        store = SessionStore(db_path=tmp_path / "sessions.db")
        manager = SessionManager(
            store=store,
            auto_gist=auto_gist,
            condensation_threshold=0.8,
            generate_fn=lambda role, prompt: GenerateResult(
                text="Fallback gist summary.", output_tokens=5,
            ),
        )
        session = manager.create_session(role="coding", context_limit=context_limit)
        return fabric, manager, session

    def test_send_message_adds_user_and_assistant(self, tmp_path):
        """send_message adds both user and assistant messages to history."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=False)

        mock_result = GenerateResult(
            text="Hello, I can help.",
            input_tokens=10,
            output_tokens=8,
            model_id="test_model",
            provider="ollama",
        )

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = manager.send_message(session, "Hello", fabric)

        assert result.text == "Hello, I can help."
        assert len(session.history) == 2
        assert session.history[0].role == "user"
        assert session.history[0].content == "Hello"
        assert session.history[1].role == "assistant"
        assert session.history[1].content == "Hello, I can help."
        assert session.history[1].model_id == "test_model"
        assert session.history[1].tokens == 8

    def test_send_message_extracts_gist(self, tmp_path):
        """send_message stores gist from fabric result in shared_context."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=True)

        mock_result = GenerateResult(
            text="Code here.\n---GIST---\nProvided fibonacci.",
            input_tokens=10,
            output_tokens=8,
        )

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = manager.send_message(session, "Write fibonacci", fabric)

        assert result.text == "Code here."
        assert result.gist == "Provided fibonacci."
        assert len(session.shared_context) == 1
        assert session.shared_context[0].summary == "Provided fibonacci."

    def test_send_message_fallback_gist(self, tmp_path):
        """When model provides no gist and auto_gist is on, fallback gist is generated."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=True)

        mock_result = GenerateResult(
            text="Response without gist.",
            input_tokens=10,
            output_tokens=8,
            gist=None,
        )

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            result = manager.send_message(session, "Do something", fabric)

        # Fallback gist should have been generated (our generate_fn returns "Fallback gist summary.")
        assert len(session.shared_context) == 1
        assert session.shared_context[0].summary == "Fallback gist summary."

    def test_send_message_updates_token_stats(self, tmp_path):
        """send_message updates output_tokens and context_limit on session."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=False, context_limit=5000)

        mock_result = GenerateResult(
            text="Response.",
            input_tokens=10,
            output_tokens=25,
            context_limit=8000,
        )

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            manager.send_message(session, "Question", fabric)

        assert session.token_stats.output_tokens == 25
        assert session.token_stats.context_limit == 8000

    def test_send_message_persists_session(self, tmp_path):
        """send_message persists the session to store after mutation."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=False)

        mock_result = GenerateResult(text="Persisted.", input_tokens=5, output_tokens=3)

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result):
            manager.send_message(session, "Save me", fabric)

        # Reload from store and verify
        reloaded = manager.get_session(session.session_id)
        assert reloaded is not None
        assert len(reloaded.history) == 2
        assert reloaded.history[1].content == "Persisted."

    def test_send_message_uses_role_override(self, tmp_path):
        """send_message respects a role override parameter."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=False)

        mock_result = GenerateResult(text="Summary result.", input_tokens=5, output_tokens=3)

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_result) as mock_gen:
            manager.send_message(session, "Summarize this", fabric, role="summarizer")

        # Verify generate_with_history was called (the role drives chain selection)
        mock_gen.assert_called_once()

    def test_send_message_no_gist_injection_when_disabled(self, tmp_path):
        """When auto_gist=False, gist instruction is not injected."""
        fabric, manager, session = self._make_manager_and_fabric(tmp_path, auto_gist=False)

        mock_result = GenerateResult(text="Clean response.", input_tokens=5, output_tokens=3)

        captured_messages = []

        def capture_gen(messages, **kwargs):
            captured_messages.extend(messages)
            return mock_result

        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", side_effect=capture_gen):
            manager.send_message(session, "Plain question", fabric)

        # The user message should NOT contain gist instruction
        user_msgs = [m for m in captured_messages if m["role"] == "user"]
        assert user_msgs
        assert "---GIST---" not in user_msgs[-1]["content"]
        assert "2-sentence factual summary" not in user_msgs[-1]["content"]


class TestSendMessageFullFlow:
    """Full end-to-end flow: create -> send -> verify -> pressure -> condense -> send."""

    def test_complete_session_flow(self, tmp_path):
        """Full session lifecycle through send_message."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "test_model": {
                    "provider": "ollama",
                    "model_name": "test",
                    "endpoint": "http://localhost:11434/api/generate",
                },
            },
            "roles": {
                "coding": ["test_model"],
                "summarizer": ["test_model"],
            },
        }
        fabric = ComputeFabric(cfg)
        store = SessionStore(db_path=tmp_path / "sessions.db")

        condense_result = GenerateResult(
            text="Condensed: discussed fibonacci and memoization.",
            output_tokens=12,
        )

        manager = SessionManager(
            store=store,
            auto_gist=True,
            condensation_threshold=0.8,
            generate_fn=lambda role, prompt: condense_result,
        )

        # 1. Create session
        session = manager.create_session(role="coding", context_limit=1000)
        assert session.session_id
        assert session.token_stats.context_limit == 1000

        # 2. Send first message
        mock_r1 = GenerateResult(
            text="Here is fibonacci.\n---GIST---\nImplemented fibonacci function.",
            input_tokens=50, output_tokens=30, context_limit=1000,
        )
        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_r1):
            r1 = manager.send_message(session, "Write fibonacci", fabric)

        assert r1.text == "Here is fibonacci."
        assert len(session.history) == 2
        assert len(session.shared_context) == 1
        assert session.shared_context[0].summary == "Implemented fibonacci function."

        # 3. Send second message
        mock_r2 = GenerateResult(
            text="Added memoization.\n---GIST---\nAdded memoization to fibonacci.",
            input_tokens=80, output_tokens=40, context_limit=1000,
        )
        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_r2):
            r2 = manager.send_message(session, "Add memoization", fabric)

        assert r2.text == "Added memoization."
        assert len(session.history) == 4  # user, assistant, user, assistant
        assert len(session.shared_context) == 2

        # 4. Simulate high pressure and condense
        session.token_stats.input_tokens = 850
        session.token_stats.output_tokens = 0
        assert manager.check_pressure(session)  # 850/1000 = 0.85 >= 0.8

        session = manager.condense(session)
        assert len(session.history) == 2  # Kept last 2
        assert len(session.shared_context) == 3  # 2 gists + 1 condensation gist
        condensation_gist = session.shared_context[-1]
        assert condensation_gist.source_role == "summarizer"
        assert condensation_gist.replaces_count == 2  # Replaced 2 old messages

        # 5. Send another message after condensation
        mock_r3 = GenerateResult(
            text="Added error handling.",
            input_tokens=30, output_tokens=15,
        )
        with patch("aurarouter.providers.ollama.OllamaProvider.generate_with_history", return_value=mock_r3):
            r3 = manager.send_message(session, "Add error handling", fabric)

        assert r3.text == "Added error handling."
        assert len(session.history) == 4  # 2 kept + 2 new
        # Verify session was persisted after final message
        reloaded = manager.get_session(session.session_id)
        assert reloaded is not None
        assert len(reloaded.history) == 4

    def test_condensation_uses_actual_tokens(self, tmp_path):
        """Condensation uses output_tokens from GenerateResult, not char heuristic."""
        store = SessionStore(db_path=tmp_path / "sessions.db")

        condense_result = GenerateResult(
            text="Short summary.",  # Only 14 chars -> ~3 tokens by heuristic
            output_tokens=42,       # But actual token count is 42
        )

        manager = SessionManager(
            store=store,
            condensation_threshold=0.8,
            generate_fn=lambda role, prompt: condense_result,
        )

        session = manager.create_session(context_limit=1000)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(Message(role=role, content=f"msg {i}", tokens=100))
        session.token_stats.input_tokens = 850
        store.save(session)

        old_token_sum = sum(m.tokens for m in session.history[:-2])  # 400

        session = manager.condense(session)

        # With heuristic: summary_tokens = max(1, 14//4) = 3
        # With actual: summary_tokens = 42
        # expected: 850 - 400 + 42 = 492
        assert session.token_stats.input_tokens == 850 - old_token_sum + 42

    def test_condensation_falls_back_to_heuristic_for_str(self, tmp_path):
        """When generate_fn returns plain str, condensation uses char heuristic."""
        store = SessionStore(db_path=tmp_path / "sessions.db")

        # generate_fn returns plain string (no output_tokens)
        manager = SessionManager(
            store=store,
            condensation_threshold=0.8,
            generate_fn=lambda role, prompt: "A short condensed summary text.",
        )

        session = manager.create_session(context_limit=1000)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(Message(role=role, content=f"msg {i}", tokens=100))
        session.token_stats.input_tokens = 850
        store.save(session)

        old_token_sum = sum(m.tokens for m in session.history[:-2])  # 400
        summary_text = "A short condensed summary text."
        heuristic_tokens = max(1, len(summary_text.strip()) // 4)

        session = manager.condense(session)

        # Should use heuristic since generate_fn returns str
        assert session.token_stats.input_tokens == 850 - old_token_sum + heuristic_tokens
