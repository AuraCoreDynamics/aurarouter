"""Encapsulation audit tests — verifies that production code respects
public API boundaries established in TG1-TG6.

Also includes:
- T7.3: High-concurrency thread safety stress test
- T7.4: End-to-end session flow integration test
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.models import Session, Message, Gist
from aurarouter.sessions.store import SessionStore


# ===================================================================
# T7.2 — Encapsulation meta-tests (source-level verification)
# ===================================================================


_SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "aurarouter"


def test_no_private_config_access_in_mcp_tools():
    """Verify mcp_tools.py doesn't access fabric._config directly."""
    source = (_SRC_DIR / "mcp_tools.py").read_text()
    assert "fabric._config" not in source


def test_no_private_session_access_in_mcp_tools():
    """Verify mcp_tools.py doesn't access session_manager._store or _auto_gist."""
    source = (_SRC_DIR / "mcp_tools.py").read_text()
    assert "session_manager._store" not in source
    assert "session_manager._auto_gist" not in source


def test_no_direct_routing_advisors_in_server():
    """Verify server.py doesn't directly set fabric._routing_advisors."""
    source = (_SRC_DIR / "server.py").read_text()
    # Should use set_routing_advisors(), not direct assignment
    lines = [l for l in source.splitlines() if "fabric._routing_advisors" in l and "=" in l]
    assert len(lines) == 0


def test_no_private_get_provider_in_mcp_tools():
    """Verify mcp_tools.py doesn't call fabric._get_provider directly."""
    source = (_SRC_DIR / "mcp_tools.py").read_text()
    assert "fabric._get_provider" not in source


def test_no_private_store_access_in_mcp_tools():
    """Verify mcp_tools.py doesn't access session_manager._store directly."""
    source = (_SRC_DIR / "mcp_tools.py").read_text()
    assert "._store" not in source


def test_no_private_provider_cache_in_mcp_tools():
    """Verify mcp_tools.py doesn't access fabric._provider_cache directly."""
    source = (_SRC_DIR / "mcp_tools.py").read_text()
    assert "fabric._provider_cache" not in source


def test_server_uses_set_routing_advisors():
    """Verify server.py uses set_routing_advisors() (public API) to configure advisors."""
    source = (_SRC_DIR / "server.py").read_text()
    # If routing_advisors are configured at all, it should be via the public method
    if "routing_advisors" in source:
        assert "set_routing_advisors" in source or "fabric._routing_advisors" not in source


# ===================================================================
# T7.3 — High-concurrency thread safety test
# ===================================================================


def test_config_thread_safety_high_concurrency(tmp_path):
    """Verify ConfigLoader under heavy concurrent load.

    100 threads, each doing 50 set_model + get_model_config cycles.
    Verify no exceptions and no corrupted reads.
    """
    import yaml

    config_content = {
        "models": {},
        "roles": {"coding": []},
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))

    num_threads = 100
    iterations_per_thread = 50
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(iterations_per_thread):
                model_id = f"model_{thread_id}_{i}"
                cfg.set_model(model_id, {"provider": "ollama", "model_name": f"m{i}"})
        except Exception as exc:
            errors.append(exc)

    def reader(thread_id: int) -> None:
        try:
            for _ in range(iterations_per_thread):
                # Read all model IDs and get config for each
                for mid in cfg.get_all_model_ids():
                    result = cfg.get_model_config(mid)
                    # Verify non-corrupted: result should be a dict
                    assert isinstance(result, dict), f"Corrupted read: {result!r}"
        except Exception as exc:
            errors.append(exc)

    def role_mutator(thread_id: int) -> None:
        try:
            for i in range(iterations_per_thread):
                chain = cfg.get_role_chain("coding")
                new_model = f"rm_{thread_id}_{i}"
                cfg.set_role_chain("coding", chain + [new_model])
        except Exception as exc:
            errors.append(exc)

    threads = []
    # 50 writer threads
    for tid in range(num_threads // 2):
        threads.append(threading.Thread(target=writer, args=(tid,), daemon=True))
    # 40 reader threads
    for tid in range(num_threads * 2 // 5):
        threads.append(threading.Thread(target=reader, args=(tid,), daemon=True))
    # 10 role mutator threads
    for tid in range(num_threads // 10):
        threads.append(threading.Thread(target=role_mutator, args=(tid,), daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"Concurrent access errors ({len(errors)}): {errors[:5]}"

    # All models written by writers should be present
    all_ids = cfg.get_all_model_ids()
    for tid in range(num_threads // 2):
        for i in range(iterations_per_thread):
            assert f"model_{tid}_{i}" in all_ids, f"Missing model_{tid}_{i}"

    # Role chain should be a valid list
    chain = cfg.get_role_chain("coding")
    assert isinstance(chain, list)
    # Should have at least some entries from the role mutators
    assert len(chain) > 0


def test_config_thread_safety_1000_ops_per_thread(tmp_path):
    """Verify ConfigLoader with 100 threads and 1000 operations per thread.

    Mixed reads and writes at maximum throughput. No exceptions allowed.
    """
    import yaml

    config_content = {
        "models": {"seed": {"provider": "ollama", "model_name": "seed"}},
        "roles": {"coding": ["seed"]},
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))

    num_threads = 100
    ops_per_thread = 1000
    errors: list[Exception] = []

    def mixed_ops(thread_id: int) -> None:
        try:
            for i in range(ops_per_thread):
                if i % 2 == 0:
                    # Write
                    cfg.set_model(
                        f"t{thread_id}_m{i}",
                        {"provider": "ollama", "model_name": f"model_{i}"},
                    )
                else:
                    # Read
                    cfg.get_model_config("seed")
                    cfg.get_all_model_ids()
                    cfg.get_role_chain("coding")
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=mixed_ops, args=(tid,), daemon=True)
        for tid in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"Thread safety errors ({len(errors)}): {errors[:5]}"

    # Verify at least some models were written
    all_ids = cfg.get_all_model_ids()
    assert "seed" in all_ids
    # Each thread writes 500 models (even indices 0..998)
    # Verify at least one thread's writes are complete
    assert len(all_ids) > 100  # Conservative — we expect ~50000


# ===================================================================
# T7.4 — End-to-end session flow test
# ===================================================================


class TestEndToEndSessionFlow:
    """Comprehensive end-to-end test covering the full session lifecycle."""

    def _make_fabric_and_manager(self, tmp_path, context_limit=10000):
        """Create wired fabric + session manager."""
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
            text="Summary: discussed fibonacci, memoization, and error handling.",
            output_tokens=20,
        )

        manager = SessionManager(
            store=store,
            auto_gist=True,
            condensation_threshold=0.8,
            generate_fn=lambda role, prompt: condense_result,
        )
        return fabric, manager, store

    def test_full_session_lifecycle_5_messages(self, tmp_path):
        """E2E: create -> 5 messages -> verify stats -> condense -> 2 more -> verify gists -> delete."""
        fabric, manager, store = self._make_fabric_and_manager(tmp_path, context_limit=2000)

        # 1. Create session
        session = manager.create_session(role="coding", context_limit=2000)
        assert session.session_id
        assert session.token_stats.context_limit == 2000

        # 2. Send 5 messages via send_message()
        responses = []
        for i in range(5):
            mock_result = GenerateResult(
                text=f"Response {i}.\n---GIST---\nHandled step {i}.",
                input_tokens=50 + i * 10,
                output_tokens=30 + i * 5,
                model_id="test_model",
                provider="ollama",
                context_limit=2000,
            )
            with patch(
                "aurarouter.providers.ollama.OllamaProvider.generate_with_history",
                return_value=mock_result,
            ):
                result = manager.send_message(session, f"Do step {i}", fabric)
            responses.append(result)

        # 3. Verify token stats accumulate
        assert session.token_stats.output_tokens > 0
        total_output = sum(r.output_tokens for r in responses)
        assert session.token_stats.output_tokens == total_output
        assert session.token_stats.context_limit == 2000

        # Verify history: 5 user + 5 assistant = 10 messages
        assert len(session.history) == 10

        # Verify gists exist from each response
        # 5 gists from model responses + 0 fallback gists (model provided gists)
        assert len(session.shared_context) >= 5

        # 4. Trigger condensation by artificially raising pressure
        session.token_stats.input_tokens = 1700  # 1700/2000 = 0.85 >= 0.8
        assert manager.check_pressure(session)

        session = manager.condense(session)

        # After condensation: kept last 2 messages, old 8 removed
        assert len(session.history) == 2
        # Gists: 5 original + condensation gist = at least 6
        assert len(session.shared_context) >= 6

        # 5. Send 2 more messages after condensation
        for i in range(5, 7):
            mock_result = GenerateResult(
                text=f"Post-condense response {i}.\n---GIST---\nPost step {i}.",
                input_tokens=40,
                output_tokens=20,
                model_id="test_model",
                provider="ollama",
            )
            with patch(
                "aurarouter.providers.ollama.OllamaProvider.generate_with_history",
                return_value=mock_result,
            ):
                manager.send_message(session, f"Do step {i}", fabric)

        # 6. Verify gists exist in shared context
        # Pre-condense: >=5, condensation: +1, post-condense: +2 = >=8
        assert len(session.shared_context) >= 8

        # History: 2 kept + 2 user + 2 assistant = 6
        assert len(session.history) == 6

        # Verify session is persisted
        reloaded = manager.get_session(session.session_id)
        assert reloaded is not None
        assert len(reloaded.history) == 6
        assert len(reloaded.shared_context) >= 8

        # 7. Delete session
        assert manager.delete_session(session.session_id) is True
        assert manager.get_session(session.session_id) is None

    def test_session_gists_accumulate_correctly(self, tmp_path):
        """Verify that gists from multiple messages accumulate in shared_context."""
        fabric, manager, store = self._make_fabric_and_manager(tmp_path)
        session = manager.create_session(role="coding", context_limit=50000)

        gist_texts = []
        for i in range(5):
            gist_text = f"Implemented feature {i}."
            gist_texts.append(gist_text)
            mock_result = GenerateResult(
                text=f"Code for feature {i}.\n---GIST---\n{gist_text}",
                input_tokens=20,
                output_tokens=15,
                model_id="test_model",
            )
            with patch(
                "aurarouter.providers.ollama.OllamaProvider.generate_with_history",
                return_value=mock_result,
            ):
                manager.send_message(session, f"Build feature {i}", fabric)

        # All 5 gists should be present
        assert len(session.shared_context) == 5
        for i, gist in enumerate(session.shared_context):
            assert gist.summary == gist_texts[i]
            assert gist.source_role == "coding"

    def test_session_token_stats_reflect_all_messages(self, tmp_path):
        """Token stats accurately reflect cumulative usage across all messages."""
        fabric, manager, store = self._make_fabric_and_manager(tmp_path)
        session = manager.create_session(role="coding", context_limit=100000)

        expected_output_tokens = 0
        for i in range(5):
            output_tokens = 10 * (i + 1)
            expected_output_tokens += output_tokens
            mock_result = GenerateResult(
                text=f"Response {i}.",
                input_tokens=20,
                output_tokens=output_tokens,
                model_id="test_model",
            )
            with patch(
                "aurarouter.providers.ollama.OllamaProvider.generate_with_history",
                return_value=mock_result,
            ):
                manager.send_message(session, f"Message {i}", fabric)

        assert session.token_stats.output_tokens == expected_output_tokens
        # output_tokens: 10 + 20 + 30 + 40 + 50 = 150
        assert session.token_stats.output_tokens == 150

    def test_session_survives_persistence_roundtrip(self, tmp_path):
        """Full session state survives save/reload from SQLite store."""
        fabric, manager, store = self._make_fabric_and_manager(tmp_path)
        session = manager.create_session(role="coding", context_limit=5000)

        mock_result = GenerateResult(
            text="Result.\n---GIST---\nDid the thing.",
            input_tokens=20,
            output_tokens=15,
            model_id="test_model",
            provider="ollama",
            context_limit=5000,
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_history",
            return_value=mock_result,
        ):
            manager.send_message(session, "Do the thing", fabric)

        # Reload from store
        reloaded = store.load(session.session_id)
        assert reloaded is not None
        assert reloaded.session_id == session.session_id
        assert len(reloaded.history) == 2
        assert reloaded.history[0].role == "user"
        assert reloaded.history[0].content == "Do the thing"
        assert reloaded.history[1].role == "assistant"
        assert reloaded.history[1].content == "Result."
        assert reloaded.history[1].model_id == "test_model"
        assert len(reloaded.shared_context) >= 1
        assert reloaded.shared_context[0].summary == "Did the thing."
        assert reloaded.token_stats.context_limit == 5000
