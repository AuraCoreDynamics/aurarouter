"""Session lifecycle management with context pressure and condensation."""

from __future__ import annotations

import json
from typing import Callable, Optional, TYPE_CHECKING

from aurarouter._logging import get_logger

from aurarouter.sessions.models import (
    Session,
    Message,
    Gist,
    TokenStats,
)
from aurarouter.tokens import count_tokens
from aurarouter.sessions.store import SessionStore
from aurarouter.sessions.gisting import (
    inject_gist_instruction,
    extract_gist,
    build_condensation_prompt,
    build_fallback_gist_prompt,
)

if TYPE_CHECKING:
    from aurarouter.fabric import ComputeFabric
    from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.Sessions")


class CompactionStrategy:
    """Interface for session history compaction (Task 4.1)."""

    def condense(self, session: Session, generate_fn: Callable) -> Session:
        """Condense session history and return updated session."""
        raise NotImplementedError


class ResumeContextBuilder:
    """Interface for dynamic session resume context (Task 4.2)."""

    def build_resume_context(self, session: Session) -> str | None:
        """Build a summary of the session state for resume reminders."""
        raise NotImplementedError


class DefaultCompactionStrategy(CompactionStrategy):
    """Default tombstoning + gisting strategy."""

    def condense(self, session: Session, generate_fn: Callable) -> Session:
        if len(session.history) <= 2:
            return session

        # Split: old messages to potentially condense, recent messages to keep
        old_messages = session.history[:-2]
        recent_messages = session.history[-2:]
        
        to_gist = []
        messages_to_keep_as_tombstones = []
        
        # Identify messages for tombstoning vs gisting
        for m in old_messages:
            is_tool_output = m.role == "tool" or (m.role == "user" and (m.content.strip().startswith("{") or m.content.strip().startswith("[")))
            if is_tool_output and len(m.content) > 500:
                m.tombstoned = True
                messages_to_keep_as_tombstones.append(m)
            else:
                to_gist.append(m)

        if not to_gist:
            session.history = messages_to_keep_as_tombstones + recent_messages
            return session

        old_dicts = [{"role": m.role, "content": m.content} for m in to_gist]
        prompt = build_condensation_prompt(old_dicts)

        try:
            raw_result = generate_fn("summarizer", prompt)
            from aurarouter.savings.models import GenerateResult as GR
            summary = raw_result.text if isinstance(raw_result, GR) else str(raw_result)
            if not summary or not summary.strip():
                logger.warning("Condensation failed: summarizer returned empty response")
                return session
        except Exception as exc:
            logger.warning("Condensation failed: %s", exc)
            return session

        # Create gist from condensation
        gist = Gist(
            source_role="summarizer",
            source_model_id="",
            summary=summary.strip(),
            replaces_count=len(to_gist),
        )
        session.add_gist(gist)

        # Update history: tombstones + recent
        old_tokens_removed = sum(m.tokens for m in to_gist)
        actual_tokens = raw_result.output_tokens if isinstance(raw_result, GR) else 0
        summary_tokens = actual_tokens if actual_tokens > 0 else count_tokens(summary.strip())
        
        session.history = messages_to_keep_as_tombstones + recent_messages
        session.token_stats.input_tokens = max(
            0, session.token_stats.input_tokens - old_tokens_removed + summary_tokens
        )
        return session


class SessionManager:
    """Manages session lifecycle, context pressure, and gisting.

    The manager is the sole owner of session state mutations.  ComputeFabric
    handles routing/generation and returns a ``GenerateResult``; this class
    handles all session bookkeeping (adding messages, gists, token stats,
    persistence).

    Args:
        store: SessionStore for persistence.
        condensation_threshold: Context pressure ratio (0.0-1.0) that
            triggers automatic condensation. Default: 0.8.
        auto_gist: Whether to inject gist instructions into prompts.
            Default: True.
        generate_fn: Optional callable(role, prompt) -> GenerateResult | str
            used for condensation and fallback gisting. Typically
            bound to ComputeFabric.execute() at integration time.
        compaction_strategy: Optional custom compaction strategy (Task 4.1).
    """

    def __init__(
        self,
        store: SessionStore,
        condensation_threshold: float = 0.8,
        auto_gist: bool = True,
        generate_fn: Optional[Callable[[str, str], str]] = None,
        compaction_strategy: CompactionStrategy | None = None,
        resume_context_builder: ResumeContextBuilder | None = None,
    ):
        self._store = store
        self._threshold = condensation_threshold
        self._auto_gist = auto_gist
        self._generate_fn = generate_fn
        self._compaction_strategy = compaction_strategy or DefaultCompactionStrategy()
        self._resume_context_builder = resume_context_builder

    def set_compaction_strategy(self, strategy: CompactionStrategy) -> None:
        """Set a custom compaction strategy (Task 4.1)."""
        self._compaction_strategy = strategy

    def set_resume_context_builder(self, builder: ResumeContextBuilder) -> None:
        """Set a custom resume context builder (Task 4.2)."""
        self._resume_context_builder = builder

    def bind_generator(self, generate_fn: Callable) -> None:
        """Bind a generation function for condensation/gisting."""
        self._generate_fn = generate_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_text(result) -> str:
        """Extract plain text from a generate_fn result.

        The generate_fn may return a GenerateResult or a plain str
        (for backwards compatibility).
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Assume GenerateResult (has .text attribute)
        return getattr(result, "text", str(result))

    @staticmethod
    def _result_output_tokens(result) -> int:
        """Extract output_tokens from a generate_fn result, or 0."""
        return getattr(result, "output_tokens", 0)

    # ------------------------------------------------------------------
    # Public accessors (avoid reaching into _store / _auto_gist)
    # ------------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        """Persist the current session state."""
        self._store.save(session)

    @property
    def auto_gist(self) -> bool:
        """Whether automatic gist injection is enabled."""
        return self._auto_gist

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, role: str = "", context_limit: int = 0) -> Session:
        """Create a new session and persist it.

        Args:
            role: The active role for this session (e.g., "coding").
            context_limit: The context window limit of the target model.

        Returns:
            A new Session with a generated UUID.
        """
        session = Session(
            token_stats=TokenStats(context_limit=context_limit),
            metadata={"active_role": role},
        )
        self._store.save(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        return self._store.load(session_id)

    def add_user_message(self, session: Session, content: str, tokens: int = 0) -> Session:
        """Add a user message to the session.

        Args:
            session: The session to update.
            content: The user's message text.
            tokens: Token count for this message (0 if unknown).

        Returns:
            The updated session (also persisted).
        """
        if tokens == 0:
            tokens = count_tokens(content)
        msg = Message(role="user", content=content, tokens=tokens)
        session.add_message(msg)
        self._store.save(session)
        return session

    def add_assistant_message(
        self,
        session: Session,
        content: str,
        model_id: str = "",
        tokens: int = 0,
    ) -> Session:
        """Add an assistant response to the session.

        If auto_gist is enabled and the response contains a gist marker,
        the gist is extracted and stored in shared_context.

        Args:
            session: The session to update.
            content: The raw model response (may contain gist marker).
            model_id: The model that generated this response.
            tokens: Token count for this message.

        Returns:
            The updated session (also persisted).
        """
        clean_content, gist_text = extract_gist(content)

        msg = Message(
            role="assistant",
            content=clean_content,
            model_id=model_id,
            tokens=tokens if tokens > 0 else count_tokens(clean_content),
        )
        session.add_message(msg)

        if gist_text:
            gist = Gist(
                source_role=session.metadata.get("active_role", ""),
                source_model_id=model_id,
                summary=gist_text,
                replaces_count=0,
            )
            session.add_gist(gist)

        self._store.save(session)
        return session

    def prepare_messages(self, session: Session) -> list[dict]:
        """Prepare the message list for sending to a provider.

        If auto_gist is enabled, injects the gist instruction into the
        last user message. Prepends shared context as a system message
        if gists exist.
        
        Detects stale sessions and injects resume reminders (Task 1.4).

        Returns:
            List of {"role": ..., "content": ...} dicts ready for
            provider.generate_with_history().
        """
        from datetime import datetime, timezone, timedelta
        
        messages = session.get_messages_as_dicts()
        
        # Build context prefix from shared gists
        context_prefix = session.get_context_prefix()
        
        # Check for staleness (Task 1.4)
        now = datetime.now(timezone.utc)
        try:
            updated_at = datetime.fromisoformat(session.updated_at)
            is_stale = (now - updated_at) > timedelta(minutes=20)
        except (ValueError, TypeError):
            is_stale = False
            
        if is_stale and messages:
            staleness_mins = (now - updated_at).seconds // 60
            resume_block = f"\n[SESSION RESUME: This conversation was paused {staleness_mins}m ago.]"
            
            # Include reasoning trace summary if present in metadata
            if "monologue_trace" in session.metadata:
                trace = session.metadata["monologue_trace"]
                resume_block += f"\n[REASONING STATE: {len(trace)} monologue steps archived.]"
            if "speculative_trace" in session.metadata:
                trace = session.metadata["speculative_trace"]
                resume_block += f"\n[SPECULATIVE STATE: {len(trace)} speculative steps archived.]"
                
            context_prefix = context_prefix + resume_block if context_prefix else resume_block

        # Prepend shared context as a system message
        if context_prefix:
            messages = [{"role": "system", "content": context_prefix}] + messages

        # Inject gist instruction into the last user message
        if self._auto_gist and messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i] = {
                        "role": "user",
                        "content": inject_gist_instruction(messages[i]["content"]),
                    }
                    break

        return messages

    def send_message(
        self,
        session: Session,
        message: str,
        fabric: ComputeFabric,
        role: str = "",
        inject_gist: bool = True,
        permissions: dict | None = None,
    ) -> GenerateResult:
        """Send a message in a session.  Single entry point for session-aware execution.

        Handles all session state management: adding user/assistant messages,
        extracting gists, updating token stats, persisting, and triggering
        fallback gist generation.

        Args:
            session: The session to send the message in.
            message: The user's message text.
            fabric: ComputeFabric for routing/generation.
            role: Override role (empty = use session's active_role).
            inject_gist: Whether to inject gist instructions.
            permissions: Optional local execution permissions (Task 2.2).

        Returns:
            GenerateResult from the fabric.
        """
        from aurarouter.savings.models import GenerateResult as GR

        # 1. Add user message
        tokens = count_tokens(message)
        session.add_message(Message(role="user", content=message, tokens=tokens))

        # 2. Prepare messages (inject gist instruction, prepend context)
        messages = self.prepare_messages(session)

        # 3. Build system prompt from context prefix
        #    (prepare_messages already prepends context as a system message,
        #     so we pass an empty system_prompt to avoid duplication)
        system_prompt = ""

        # 4. Route through fabric (Task 3.1: support monologue/speculative)
        active_role = role or session.metadata.get("active_role", "coding")
        execution_mode = session.metadata.get("execution_mode", "standard")
        
        if execution_mode == "monologue":
            import asyncio
            # Convert messages to task/context
            task = messages[-1]["content"] if messages else ""
            context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
            
            # Execute monologue
            try:
                monologue_result = asyncio.run(fabric.execute_monologue(
                    task=task, context=context, permissions=permissions
                ))
                result = GR(
                    text=monologue_result.final_output,
                    model_id=monologue_result.reasoning_trace[-1].model_id if monologue_result.reasoning_trace else "",
                    provider="monologue",
                )
                # Archive trace (Task 1.4)
                session.metadata["monologue_trace"] = [s.to_dict() for s in monologue_result.reasoning_trace]
            except Exception as e:
                logger.error(f"Monologue execution failed: {e}")
                result = GR(text=f"ERROR: Monologue failed: {e}")
                
        elif execution_mode == "speculative":
            import asyncio
            task = messages[-1]["content"] if messages else ""
            context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
            
            try:
                spec_result = asyncio.run(fabric.execute_speculative(
                    task=task, context=context, permissions=permissions
                ))
                result = GR(
                    text=spec_result.get("content", ""),
                    model_id=spec_result.get("model_id", ""),
                    provider="speculative",
                )
                # Archive trace (Task 1.4)
                if "session_id" in spec_result:
                    session.metadata["speculative_trace"] = spec_result
            except Exception as e:
                logger.error(f"Speculative execution failed: {e}")
                result = GR(text=f"ERROR: Speculative failed: {e}")
                
        else:
            result = fabric.execute_session(
                role=active_role,
                messages=messages,
                system_prompt=system_prompt,
                json_mode=False,
            )

        # 5. Post-process: add assistant message, update token stats
        #    result.text is already gist-extracted by fabric's execute_session
        msg = Message(
            role="assistant",
            content=result.text,
            model_id=result.model_id,
            tokens=result.output_tokens,
        )
        session.add_message(msg)
        session.token_stats.output_tokens += result.output_tokens

        # 6. Update context limit if provider reported it
        if result.context_limit > 0:
            session.token_stats.context_limit = result.context_limit

        # 7. Store gist if fabric extracted one
        if result.gist:
            gist = Gist(
                source_role=active_role,
                source_model_id=result.model_id,
                summary=result.gist,
                replaces_count=0,
            )
            session.add_gist(gist)

        # 8. Persist
        self.save_session(session)

        # 9. Fallback gist if needed
        if self._auto_gist and result.gist is None:
            self.generate_fallback_gist(session, result.text, result.model_id)

        return result

    def check_pressure(self, session: Session) -> bool:
        """Check if the session's context pressure exceeds the threshold.

        Returns True if condensation is needed.
        """
        return session.token_stats.pressure >= self._threshold

    def condense(self, session: Session) -> Session:
        """Condense the session's history using the active strategy.

        Returns:
            The updated session with condensed history.
        """
        if self._generate_fn is None:
            logger.warning("Condensation skipped: generate_fn not bound")
            return session

        session = self._compaction_strategy.condense(session, self._generate_fn)
        self._store.save(session)
        return session

    def generate_fallback_gist(
        self, session: Session, response_text: str, model_id: str = ""
    ) -> Session:
        """Generate a gist using the summarizer when the model didn't provide one.

        Requires generate_fn to be set.

        Returns:
            The updated session with the fallback gist added.
        """
        if self._generate_fn is None:
            logger.debug("Fallback gist skipped: generate_fn not bound")
            return session

        prompt = build_fallback_gist_prompt(response_text)

        try:
            raw_result = self._generate_fn("summarizer", prompt)
            summary = self._result_text(raw_result)
            if summary and summary.strip():
                gist = Gist(
                    source_role=session.metadata.get("active_role", ""),
                    source_model_id=model_id,
                    summary=summary.strip(),
                    replaces_count=0,
                )
                session.add_gist(gist)
                self._store.save(session)
        except Exception:
            logger.debug("Fallback gist generation failed", exc_info=True)

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from the store."""
        return self._store.delete(session_id)

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List sessions (metadata only)."""
        return self._store.list_sessions(limit=limit, offset=offset)
