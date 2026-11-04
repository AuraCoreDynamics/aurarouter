"""Phase 3 tests: Zero-LLM Ephemeral SessionProjector (v2.1 §5)."""

from __future__ import annotations

import copy
import time

import pytest

from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.models import Gist, Message, Session
from aurarouter.sessions.projector import (
    ProjectedPayload,
    SessionProjector,
    TargetContextBudget,
)
from aurarouter.sessions.store import SessionStore


class TestSessionProjector:
    def _build_128k_session(self) -> Session:
        """Build a 40-turn (~128k-token) session with pre-computed gists."""
        session = Session()
        chunk = "def compute_step(x: int) -> int:\n    return (x * 31 + 17) % 1000003\n"
        # ~3,200 tokens per turn * 40 turns = ~128,000 tokens
        big_turn_body = chunk * 170
        for i in range(20):
            session.add_message(
                Message(
                    role="user",
                    content=f"Turn {i} user request:\n{big_turn_body}",
                    tokens=3200,
                    tombstoned=(i == 2),  # Turn 2 user tool dump is tombstoned
                )
            )
            gist_str = f"Turn {i}: implemented compute_step variant #{i} with modular arithmetic."
            session.add_message(
                Message(
                    role="assistant",
                    content=f"Turn {i} assistant implementation:\n{big_turn_body}",
                    model_id="claude-sonnet-4",
                    tokens=3200,
                    gist=gist_str,
                    gist_tokens=16,
                )
            )
            session.add_gist(
                Gist(source_role="coding", source_model_id="claude-sonnet-4", summary=gist_str)
            )

        # Final active user question (~120 tokens)
        session.add_message(
            Message(
                role="user",
                content="Now write unit tests for Turn 19's modular arithmetic edge cases.",
                tokens=25,
            )
        )
        return session

    def test_128k_to_8k_projection_zero_llm_sub_millisecond_and_immutable(self):
        session = self._build_128k_session()
        history_snapshot = copy.deepcopy(session.to_dict())
        raw_messages = session.get_messages_as_dicts()
        raw_messages_copy = copy.deepcopy(raw_messages)

        projector = SessionProjector()
        budget = TargetContextBudget(
            model_id="local_qwen_8k",
            total_context_limit=8192,
            reserved_output_tokens=2048,
            safety_margin_ratio=0.08,
        )
        assert budget.max_usable_input_tokens == int(8192 * 0.92) - 2048  # 5488

        # Warm call + timed run
        projector.project(
            messages=raw_messages,
            system_prompt="You are a senior Python engineer.",
            budget=budget,
            session=session,
            failover_from="claude-sonnet-4",
            failover_reason="HTTP 429 rate_limit_exceeded",
        )

        t0 = time.perf_counter()
        projected = projector.project(
            messages=raw_messages,
            system_prompt="You are a senior Python engineer.",
            budget=budget,
            session=session,
            failover_from="claude-sonnet-4",
            failover_reason="HTTP 429 rate_limit_exceeded",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Sub-millisecond O(N) CPU execution (generous 5ms ceiling for OS scheduling jitter)
        assert elapsed_ms < 5.0, f"Projection took {elapsed_ms:.3f}ms"
        assert projected.was_projected is True
        assert projected.active_turn_truncated is False
        assert projected.estimated_input_tokens <= budget.max_usable_input_tokens
        assert projected.turns_kept_full >= 2  # Active user turn + Turn 19 assistant (3200 + 25 <= 5488)
        assert projected.turns_gisted >= 10
        assert "[FAILOVER CONTEXT:" in projected.system_prompt
        assert "claude-sonnet-4" in projected.system_prompt

        # First projected message contains condensed gists, last is active user prompt
        assert projected.messages[0]["role"] == "system"
        assert "[Prior Context]" in projected.messages[0]["content"]
        assert projected.messages[-1]["role"] == "user"
        assert "Turn 19's modular arithmetic edge cases" in projected.messages[-1]["content"]

        # Canonical Session and caller messages list are 100% UNMUTATED
        assert session.to_dict() == history_snapshot
        assert raw_messages == raw_messages_copy

    def test_40k_single_turn_paste_deterministic_middle_out_elision(self):
        lines = [f"LINE_{i:04d}: stack_frame_or_log_entry_payload_data_{i}" for i in range(3000)]
        giant_paste = "\n".join(lines)  # ~150k chars (~42k tokens)
        messages = [{"role": "user", "content": giant_paste}]

        projector = SessionProjector()
        budget = TargetContextBudget(
            model_id="local_llama_8k",
            total_context_limit=8192,
            reserved_output_tokens=2048,
        )

        projected = projector.project(
            messages=messages,
            system_prompt="Analyze this crash log.",
            budget=budget,
        )

        assert projected.was_projected is True
        assert projected.active_turn_truncated is True
        assert projected.estimated_input_tokens <= budget.max_usable_input_tokens

        elided_user = projected.messages[-1]["content"]
        # Preserves head (LINE_0000) and tail (LINE_2999), elides middle with marker
        assert "LINE_0000:" in elided_user
        assert "LINE_2999:" in elided_user
        assert "tokens omitted during context projection]" in elided_user
        assert "LINE_1500:" not in elided_user

    def test_fast_path_no_op_when_within_budget_or_unlimited(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        projector = SessionProjector()

        p1 = projector.project(
            messages=messages,
            system_prompt="Be brief.",
            budget=TargetContextBudget(model_id="local_8k", total_context_limit=8192),
        )
        assert p1.was_projected is False
        assert p1.messages == messages

        p2 = projector.project(
            messages=messages,
            budget=TargetContextBudget(model_id="unknown_ctx", total_context_limit=0),
        )
        assert p2.was_projected is False
        assert p2.messages == messages

    def test_message_gist_persistence_and_cache_affinity_metadata(self, tmp_path):
        # Legacy dict without gist fields loads cleanly
        legacy = Message.from_dict({"role": "assistant", "content": "hi", "tokens": 2})
        assert legacy.gist is None
        assert legacy.gist_tokens == 0

        store = SessionStore(db_path=tmp_path / "sessions.db")
        mgr = SessionManager(
            store=store,
            auto_gist=True,
            generate_fn=lambda role, prompt: "Fallback 2-sentence gist.",
        )
        session = mgr.create_session(role="coding", context_limit=32000)
        mgr.add_user_message(session, "Implement LRU cache")
        mgr.add_assistant_message(
            session,
            "class LRUCache: pass\n---GIST---\nImplemented LRUCache skeleton.",
            model_id="claude-sonnet-4",
            tokens=12,
        )

        reloaded = mgr.get_session(session.session_id)
        assert reloaded is not None
        assert reloaded.history[-1].gist == "Implemented LRUCache skeleton."
        assert reloaded.history[-1].gist_tokens > 0
        assert reloaded.metadata["last_model_id"] == "claude-sonnet-4"
        assert reloaded.metadata["last_routed_at"] > 0

