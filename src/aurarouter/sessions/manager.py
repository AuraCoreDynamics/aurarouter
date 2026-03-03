"""Session lifecycle management with context pressure and condensation."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from aurarouter._logging import get_logger

from aurarouter.sessions.models import (
    Session,
    Message,
    Gist,
    TokenStats,
)
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
    """

    def __init__(
        self,
        store: SessionStore,
        condensation_threshold: float = 0.8,
        auto_gist: bool = True,
        generate_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self._store = store
        self._threshold = condensation_threshold
        self._auto_gist = auto_gist
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
            tokens=tokens,
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

        Returns:
            List of {"role": ..., "content": ...} dicts ready for
            provider.generate_with_history().
        """
        messages = session.get_messages_as_dicts()

        # Prepend shared context as a system message
        context_prefix = session.get_context_prefix()
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

        Returns:
            GenerateResult from the fabric.
        """
        from aurarouter.savings.models import GenerateResult as GR

        # 1. Add user message
        session.add_message(Message(role="user", content=message))

        # 2. Prepare messages (inject gist instruction, prepend context)
        messages = self.prepare_messages(session)

        # 3. Build system prompt from context prefix
        #    (prepare_messages already prepends context as a system message,
        #     so we pass an empty system_prompt to avoid duplication)
        system_prompt = ""

        # 4. Route through fabric (fabric no longer touches session)
        active_role = role or session.metadata.get("active_role", "coding")
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
        """Condense the session's history by summarizing old messages.

        Keeps the most recent 2 messages intact. Summarizes all older
        messages into a Gist and removes the originals.

        Requires generate_fn to be set. If not set or if condensation
        fails, returns the session unchanged.

        Returns:
            The updated session with condensed history.
        """
        if self._generate_fn is None:
            logger.warning("Condensation skipped: generate_fn not bound")
            return session

        if len(session.history) <= 2:
            return session

        # Split: old messages to condense, recent messages to keep
        old_messages = session.history[:-2]
        recent_messages = session.history[-2:]

        old_dicts = [{"role": m.role, "content": m.content} for m in old_messages]
        prompt = build_condensation_prompt(old_dicts)

        try:
            raw_result = self._generate_fn("summarizer", prompt)
            summary = self._result_text(raw_result)
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
            replaces_count=len(old_messages),
        )
        session.add_gist(gist)

        # Replace history with only recent messages
        old_tokens = sum(m.tokens for m in old_messages)
        # Use actual output_tokens if available, else heuristic (1 token ~ 4 chars)
        actual_tokens = self._result_output_tokens(raw_result)
        summary_tokens = actual_tokens if actual_tokens > 0 else max(1, len(summary.strip()) // 4)
        session.history = recent_messages
        session.token_stats.input_tokens = max(
            0, session.token_stats.input_tokens - old_tokens + summary_tokens
        )

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
