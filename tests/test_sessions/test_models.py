"""Comprehensive tests for aurarouter.sessions.models."""

import uuid
from datetime import datetime, timezone

import pytest

from aurarouter.sessions.models import Message, Gist, TokenStats, Session


# -- Message ---------------------------------------------------------------


class TestMessage:
    def test_message_defaults(self):
        """Message gets auto-timestamp, empty model_id, 0 tokens."""
        msg = Message("user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp != ""
        # Verify it is a valid ISO 8601 timestamp
        datetime.fromisoformat(msg.timestamp)
        assert msg.model_id == ""
        assert msg.tokens == 0

    def test_message_roundtrip(self):
        """Message.from_dict(msg.to_dict()) preserves all fields."""
        msg = Message(
            role="assistant",
            content="world",
            timestamp="2025-01-01T00:00:00+00:00",
            model_id="gemini-pro",
            tokens=42,
        )
        restored = Message.from_dict(msg.to_dict())
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.timestamp == msg.timestamp
        assert restored.model_id == msg.model_id
        assert restored.tokens == msg.tokens

    def test_from_dict_ignores_unknown_keys(self):
        """Message.from_dict with extra keys does not raise."""
        msg = Message.from_dict({"role": "user", "content": "hi", "bogus": 1})
        assert msg.role == "user"
        assert msg.content == "hi"


# -- Gist ------------------------------------------------------------------


class TestGist:
    def test_gist_defaults(self):
        """Gist gets auto-timestamp when none provided."""
        gist = Gist("coding", "model1", "summary text")
        assert gist.source_role == "coding"
        assert gist.source_model_id == "model1"
        assert gist.summary == "summary text"
        assert gist.timestamp != ""
        datetime.fromisoformat(gist.timestamp)
        assert gist.replaces_count == 0

    def test_gist_roundtrip(self):
        """Gist.from_dict(gist.to_dict()) preserves all fields."""
        gist = Gist(
            source_role="assistant",
            source_model_id="claude-3",
            summary="A summary of the conversation",
            timestamp="2025-06-01T12:00:00+00:00",
            replaces_count=5,
        )
        restored = Gist.from_dict(gist.to_dict())
        assert restored.source_role == gist.source_role
        assert restored.source_model_id == gist.source_model_id
        assert restored.summary == gist.summary
        assert restored.timestamp == gist.timestamp
        assert restored.replaces_count == gist.replaces_count


# -- TokenStats ------------------------------------------------------------


class TestTokenStats:
    def test_token_stats_pressure_zero_limit(self):
        """context_limit=0 means pressure == 0.0."""
        stats = TokenStats(input_tokens=100, output_tokens=200, context_limit=0)
        assert stats.pressure == 0.0

    def test_token_stats_pressure_normal(self):
        """input=400, output=400, limit=1000 gives pressure == 0.8."""
        stats = TokenStats(input_tokens=400, output_tokens=400, context_limit=1000)
        assert stats.pressure == pytest.approx(0.8)

    def test_token_stats_pressure_capped(self):
        """Pressure never exceeds 1.0."""
        stats = TokenStats(input_tokens=800, output_tokens=800, context_limit=1000)
        assert stats.pressure == 1.0

    def test_token_stats_remaining(self):
        """limit=1000, input=400, output=200 gives remaining == 400."""
        stats = TokenStats(input_tokens=400, output_tokens=200, context_limit=1000)
        assert stats.remaining == 400

    def test_token_stats_remaining_zero_limit(self):
        """remaining is 0 when context_limit is 0."""
        stats = TokenStats(input_tokens=100, output_tokens=100, context_limit=0)
        assert stats.remaining == 0

    def test_token_stats_total_used(self):
        """total_used is sum of input and output."""
        stats = TokenStats(input_tokens=300, output_tokens=200)
        assert stats.total_used == 500

    def test_token_stats_roundtrip(self):
        """TokenStats.from_dict(stats.to_dict()) preserves all fields."""
        stats = TokenStats(input_tokens=10, output_tokens=20, context_limit=100)
        restored = TokenStats.from_dict(stats.to_dict())
        assert restored.input_tokens == stats.input_tokens
        assert restored.output_tokens == stats.output_tokens
        assert restored.context_limit == stats.context_limit


# -- Session ---------------------------------------------------------------


class TestSession:
    def test_session_auto_id(self):
        """Session() generates a UUID4 session_id."""
        session = Session()
        parsed = uuid.UUID(session.session_id, version=4)
        assert str(parsed) == session.session_id

    def test_session_add_message(self):
        """add_message appends to history and increments iteration_count."""
        session = Session()
        msg = Message("user", "hello", tokens=10)
        session.add_message(msg)

        assert len(session.history) == 1
        assert session.history[0] is msg
        assert session.metadata["iteration_count"] == 1
        assert session.token_stats.input_tokens == 10

        msg2 = Message("assistant", "hi back", tokens=5)
        session.add_message(msg2)
        assert len(session.history) == 2
        assert session.metadata["iteration_count"] == 2
        assert session.token_stats.input_tokens == 15

    def test_session_add_gist(self):
        """add_gist appends to shared_context."""
        session = Session()
        gist = Gist("assistant", "model1", "A summary")
        session.add_gist(gist)

        assert len(session.shared_context) == 1
        assert session.shared_context[0] is gist

    def test_session_get_messages_as_dicts(self):
        """get_messages_as_dicts returns [{role, content}] format."""
        session = Session()
        session.add_message(Message("user", "hello"))
        session.add_message(Message("assistant", "world"))

        result = session.get_messages_as_dicts()
        assert result == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_session_get_context_prefix(self):
        """get_context_prefix returns formatted gist prefix."""
        session = Session()
        assert session.get_context_prefix() == ""

        session.add_gist(Gist("assistant", "model1", "First summary"))
        session.add_gist(Gist("assistant", "model2", "Second summary"))

        prefix = session.get_context_prefix()
        assert "[Prior Context]" in prefix
        assert "- First summary" in prefix
        assert "- Second summary" in prefix
        assert "[End Prior Context]" in prefix

    def test_session_roundtrip(self):
        """Session.from_dict(session.to_dict()) preserves all fields."""
        session = Session()
        session.add_message(Message("user", "hello", tokens=10))
        session.add_message(
            Message("assistant", "hi", model_id="gemini", tokens=5)
        )
        session.add_gist(Gist("assistant", "gemini", "Greeted user", replaces_count=2))

        data = session.to_dict()
        restored = Session.from_dict(data)

        assert restored.session_id == session.session_id
        assert restored.created_at == session.created_at
        assert restored.updated_at == session.updated_at
        assert len(restored.history) == 2
        assert restored.history[0].role == "user"
        assert restored.history[0].content == "hello"
        assert restored.history[1].model_id == "gemini"
        assert len(restored.shared_context) == 1
        assert restored.shared_context[0].summary == "Greeted user"
        assert restored.shared_context[0].replaces_count == 2
        assert restored.token_stats.input_tokens == session.token_stats.input_tokens
        assert restored.metadata == session.metadata

    def test_session_timestamps_auto_set(self):
        """created_at and updated_at are auto-set on construction."""
        session = Session()
        assert session.created_at != ""
        assert session.updated_at != ""
        datetime.fromisoformat(session.created_at)
        datetime.fromisoformat(session.updated_at)
