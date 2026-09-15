"""Comprehensive tests for aurarouter.sessions.store."""

import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aurarouter.sessions.models import Message, Gist, Session
from aurarouter.sessions.store import SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    """Create a SessionStore backed by a temp directory."""
    return SessionStore(db_path=tmp_path / "test_sessions.db")


def _make_session(**kwargs) -> Session:
    """Helper to create a Session with optional overrides."""
    return Session(**kwargs)


# -- Core CRUD -------------------------------------------------------------


class TestSessionStoreCRUD:
    def test_save_and_load(self, store: SessionStore):
        """Save a session, load by ID, verify equal."""
        session = _make_session()
        session.add_message(Message("user", "hello", tokens=10))
        store.save(session)

        loaded = store.load(session.session_id)
        assert loaded is not None
        assert loaded.session_id == session.session_id
        assert len(loaded.history) == 1
        assert loaded.history[0].content == "hello"
        assert loaded.token_stats.input_tokens == 10

    def test_load_missing(self, store: SessionStore):
        """Load nonexistent ID returns None."""
        result = store.load("nonexistent-id")
        assert result is None

    def test_save_overwrites(self, store: SessionStore):
        """Save same session_id twice, load returns latest."""
        session = _make_session()
        session.add_message(Message("user", "first"))
        store.save(session)

        session.add_message(Message("user", "second"))
        store.save(session)

        loaded = store.load(session.session_id)
        assert loaded is not None
        assert len(loaded.history) == 2
        assert loaded.history[1].content == "second"


# -- Listing ---------------------------------------------------------------


class TestSessionStoreListing:
    def test_list_sessions(self, store: SessionStore):
        """Save 3 sessions, list_sessions() returns all 3 with metadata."""
        ids = []
        for i in range(3):
            s = _make_session()
            s.add_message(Message("user", f"msg-{i}"))
            store.save(s)
            ids.append(s.session_id)

        listing = store.list_sessions()
        assert len(listing) == 3
        listed_ids = {item["session_id"] for item in listing}
        for sid in ids:
            assert sid in listed_ids

        for item in listing:
            assert "session_id" in item
            assert "created_at" in item
            assert "updated_at" in item

    def test_list_sessions_limit_offset(self, store: SessionStore):
        """Verify pagination works with limit and offset."""
        for i in range(5):
            s = _make_session()
            ts = datetime(2025, 1, 1, 0, 0, i, tzinfo=timezone.utc).isoformat()
            s.updated_at = ts
            store.save(s)

        page1 = store.list_sessions(limit=2, offset=0)
        assert len(page1) == 2

        page2 = store.list_sessions(limit=2, offset=2)
        assert len(page2) == 2

        page3 = store.list_sessions(limit=2, offset=4)
        assert len(page3) == 1

        all_ids = [item["session_id"] for item in page1 + page2 + page3]
        assert len(set(all_ids)) == 5


# -- Delete ----------------------------------------------------------------


class TestSessionStoreDelete:
    def test_delete(self, store: SessionStore):
        """Save then delete, verify load returns None, delete returns True."""
        session = _make_session()
        store.save(session)

        result = store.delete(session.session_id)
        assert result is True

        loaded = store.load(session.session_id)
        assert loaded is None

    def test_delete_missing(self, store: SessionStore):
        """Delete nonexistent returns False."""
        result = store.delete("nonexistent-id")
        assert result is False


# -- Purge -----------------------------------------------------------------


class TestSessionStorePurge:
    def test_purge_before(self, store: SessionStore):
        """Save 3 sessions with different timestamps, purge old ones."""
        old_ts = "2024-01-01T00:00:00+00:00"
        mid_ts = "2025-06-01T00:00:00+00:00"
        new_ts = "2026-01-01T00:00:00+00:00"

        s1 = _make_session()
        s1.updated_at = old_ts
        s1.created_at = old_ts
        store.save(s1)

        s2 = _make_session()
        s2.updated_at = mid_ts
        s2.created_at = mid_ts
        store.save(s2)

        s3 = _make_session()
        s3.updated_at = new_ts
        s3.created_at = new_ts
        store.save(s3)

        deleted = store.purge_before(mid_ts)
        assert deleted == 1

        assert store.load(s1.session_id) is None
        assert store.load(s2.session_id) is not None
        assert store.load(s3.session_id) is not None


# -- Thread Safety ---------------------------------------------------------


class TestSessionStoreThreadSafety:
    def test_thread_safety(self, store: SessionStore):
        """Spawn 10 threads each saving a session concurrently, verify all saved."""
        sessions = [_make_session() for _ in range(10)]
        errors = []

        def save_session(s: Session):
            try:
                s.add_message(Message("user", f"thread-msg-{s.session_id[:8]}"))
                store.save(s)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=save_session, args=(s,)) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"

        listing = store.list_sessions(limit=100)
        saved_ids = {item["session_id"] for item in listing}
        for s in sessions:
            assert s.session_id in saved_ids

        for s in sessions:
            loaded = store.load(s.session_id)
            assert loaded is not None
            assert len(loaded.history) == 1
