"""Tests for SessionManager lifecycle, context pressure, and gisting."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aurarouter.sessions.models import Session, Message, Gist, TokenStats
from aurarouter.sessions.store import SessionStore
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.gisting import GIST_MARKER


@pytest.fixture
def store(tmp_path):
    return SessionStore(db_path=tmp_path / "test_sessions.db")


@pytest.fixture
def manager(store):
    return SessionManager(store=store)


# --- Session creation ---

def test_create_session(manager, store):
    session = manager.create_session()
    assert session.session_id
    loaded = store.load(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id


def test_create_session_with_role(manager):
    session = manager.create_session(role="coding")
    assert session.metadata["active_role"] == "coding"


def test_create_session_with_context_limit(manager):
    session = manager.create_session(context_limit=32768)
    assert session.token_stats.context_limit == 32768


# --- Session retrieval ---

def test_get_session(manager):
    session = manager.create_session()
    loaded = manager.get_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id


def test_get_session_missing(manager):
    assert manager.get_session("nonexistent-id") is None


# --- Message management ---

def test_add_user_message(manager, store):
    session = manager.create_session()
    session = manager.add_user_message(session, "Hello!", tokens=5)
    assert len(session.history) == 1
    assert session.history[0].role == "user"
    assert session.history[0].content == "Hello!"
    # Verify persisted
    loaded = store.load(session.session_id)
    assert len(loaded.history) == 1


def test_add_assistant_message_with_gist(manager):
    session = manager.create_session(role="coding")
    raw_response = f"Here is the code.\n{GIST_MARKER}\nProvided a fibonacci implementation."
    session = manager.add_assistant_message(
        session, raw_response, model_id="test-model", tokens=20
    )
    # Content should be cleaned
    assert session.history[0].content == "Here is the code."
    # Gist should be stored
    assert len(session.shared_context) == 1
    assert session.shared_context[0].summary == "Provided a fibonacci implementation."
    assert session.shared_context[0].source_role == "coding"


def test_add_assistant_message_no_gist(manager):
    session = manager.create_session()
    session = manager.add_assistant_message(session, "Plain response", model_id="m1")
    assert session.history[0].content == "Plain response"
    assert len(session.shared_context) == 0


# --- Message preparation ---

def test_prepare_messages_basic(manager):
    session = manager.create_session()
    session.add_message(Message(role="user", content="hello"))
    session.add_message(Message(role="assistant", content="hi"))
    messages = manager.prepare_messages(session)
    # With auto_gist=True, no user message at end to inject into
    # (last message is assistant), so messages should just have the two
    assert any(m["role"] == "user" for m in messages)
    assert any(m["role"] == "assistant" for m in messages)


def test_prepare_messages_with_gist_injection(manager):
    session = manager.create_session()
    session.add_message(Message(role="user", content="hello"))
    messages = manager.prepare_messages(session)
    # auto_gist=True, last user message should have gist instruction
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert "---GIST---" in user_msgs[0]["content"]


def test_prepare_messages_with_shared_context(manager):
    session = manager.create_session()
    session.add_gist(Gist("coding", "m1", "Prior summary"))
    session.add_message(Message(role="user", content="continue"))
    messages = manager.prepare_messages(session)
    # Should have a system message with context prefix
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Prior summary" in system_msgs[0]["content"]


def test_prepare_messages_no_auto_gist(store):
    manager = SessionManager(store=store, auto_gist=False)
    session = manager.create_session()
    session.add_message(Message(role="user", content="hello"))
    messages = manager.prepare_messages(session)
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert "---GIST---" not in user_msgs[0]["content"]


# --- Context pressure ---

def test_check_pressure_below_threshold(manager):
    session = manager.create_session(context_limit=10000)
    session.token_stats.input_tokens = 3000
    session.token_stats.output_tokens = 3000
    assert not manager.check_pressure(session)  # 0.6 < 0.8


def test_check_pressure_above_threshold(manager):
    session = manager.create_session(context_limit=10000)
    session.token_stats.input_tokens = 4000
    session.token_stats.output_tokens = 4500
    assert manager.check_pressure(session)  # 0.85 >= 0.8


# --- Condensation ---

def test_condense_basic(store):
    mock_fn = MagicMock(return_value="Condensed summary of the conversation.")
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session()
    # Add 6 messages
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        session.add_message(Message(role=role, content=f"Message {i}", tokens=100))
    store.save(session)

    session = manager.condense(session)
    # Should keep only last 2 messages
    assert len(session.history) == 2
    assert session.history[0].content == "Message 4"
    assert session.history[1].content == "Message 5"
    # Should have a gist from condensation
    assert len(session.shared_context) == 1
    assert session.shared_context[0].summary == "Condensed summary of the conversation."
    assert session.shared_context[0].replaces_count == 4
    mock_fn.assert_called_once()


def test_condense_too_few_messages(store):
    mock_fn = MagicMock()
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session()
    session.add_message(Message(role="user", content="hello"))
    session.add_message(Message(role="assistant", content="hi"))
    session = manager.condense(session)
    assert len(session.history) == 2  # Unchanged
    mock_fn.assert_not_called()


def test_condense_no_generate_fn(manager):
    session = manager.create_session()
    for i in range(6):
        session.add_message(Message(role="user", content=f"msg {i}"))
    session = manager.condense(session)
    assert len(session.history) == 6  # Unchanged — no generate_fn


def test_condense_generate_fn_fails(store):
    mock_fn = MagicMock(side_effect=RuntimeError("Model unavailable"))
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session()
    for i in range(6):
        session.add_message(Message(role="user", content=f"msg {i}"))
    session = manager.condense(session)
    assert len(session.history) == 6  # Unchanged on failure


# --- Fallback gisting ---

def test_generate_fallback_gist(store):
    mock_fn = MagicMock(return_value="A 2-sentence summary.")
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session(role="coding")
    session = manager.generate_fallback_gist(session, "Long response text", model_id="m1")
    assert len(session.shared_context) == 1
    assert session.shared_context[0].summary == "A 2-sentence summary."
    assert session.shared_context[0].source_role == "coding"


def test_generate_fallback_gist_no_fn(manager):
    session = manager.create_session()
    session = manager.generate_fallback_gist(session, "text")
    assert len(session.shared_context) == 0  # No change


# --- Delete / List ---

def test_delete_session(manager):
    session = manager.create_session()
    assert manager.delete_session(session.session_id) is True
    assert manager.get_session(session.session_id) is None


def test_list_sessions(manager):
    for _ in range(3):
        manager.create_session()
    sessions = manager.list_sessions()
    assert len(sessions) == 3
    assert all("session_id" in s for s in sessions)


# --- TG-C Task 3.4: Condensation hardening tests ---


def test_condense_none_generate_fn_logs_warning(manager, caplog):
    """Condensation with None generate_fn logs warning."""
    session = manager.create_session()
    for i in range(5):
        session.add_message(Message(role="user", content=f"msg {i}"))

    with caplog.at_level(logging.WARNING, logger="AuraRouter.Sessions"):
        session = manager.condense(session)

    assert len(session.history) == 5  # Unchanged
    assert "generate_fn not bound" in caplog.text


def test_condense_empty_summary_logs_warning(store, caplog):
    """Condensation with empty summary logs warning."""
    mock_fn = MagicMock(return_value="")
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session()
    for i in range(6):
        session.add_message(Message(role="user", content=f"msg {i}"))

    with caplog.at_level(logging.WARNING, logger="AuraRouter.Sessions"):
        session = manager.condense(session)

    assert len(session.history) == 6  # Unchanged
    assert "summarizer returned empty response" in caplog.text


def test_condense_exception_logs_warning(store, caplog):
    """Condensation with exception logs warning with exception info."""
    mock_fn = MagicMock(side_effect=RuntimeError("Model unavailable"))
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session()
    for i in range(6):
        session.add_message(Message(role="user", content=f"msg {i}"))

    with caplog.at_level(logging.WARNING, logger="AuraRouter.Sessions"):
        session = manager.condense(session)

    assert len(session.history) == 6  # Unchanged
    assert "Model unavailable" in caplog.text


def test_configurable_condensation_threshold(store):
    """Custom threshold triggers pressure at the configured ratio."""
    manager = SessionManager(store=store, condensation_threshold=0.5)
    session = manager.create_session(context_limit=10000)
    session.token_stats.input_tokens = 3000
    session.token_stats.output_tokens = 2500
    # Pressure = 5500/10000 = 0.55 >= 0.5 → should trigger
    assert manager.check_pressure(session)

    # With default 0.8, same state should NOT trigger
    default_manager = SessionManager(store=store)
    assert not default_manager.check_pressure(session)  # 0.55 < 0.8


def test_summary_token_accounting(store):
    """After condensation, token stats reflect both removed and added tokens."""
    summary_text = "A concise summary of the conversation."  # 40 chars → ~10 tokens
    mock_fn = MagicMock(return_value=summary_text)
    manager = SessionManager(store=store, generate_fn=mock_fn)
    session = manager.create_session()
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        session.add_message(Message(role=role, content=f"Message {i}", tokens=100))
    session.token_stats.input_tokens = 600
    store.save(session)

    session = manager.condense(session)

    # 4 old messages removed (400 tokens), summary added (~10 tokens)
    expected_summary_tokens = max(1, len(summary_text.strip()) // 4)
    expected_input = max(0, 600 - 400 + expected_summary_tokens)
    assert session.token_stats.input_tokens == expected_input
    assert expected_summary_tokens > 0  # Verify heuristic produced non-zero
