"""Session lifecycle management with context pressure and condensation."""

from __future__ import annotations

from typing import Callable, Optional

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


class SessionManager:
    """Manages session lifecycle, context pressure, and gisting.

    The manager is decoupled from ComputeFabric — it receives a
    generate_fn callable for condensation and fallback gisting to
    avoid circular imports.

    Args:
        store: SessionStore for persistence.
        condensation_threshold: Context pressure ratio (0.0-1.0) that
            triggers automatic condensation. Default: 0.8.
        auto_gist: Whether to inject gist instructions into prompts.
            Default: True.
        generate_fn: Optional callable(role: str, prompt: str) -> str
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
            return session

        if len(session.history) <= 2:
            return session

        # Split: old messages to condense, recent messages to keep
        old_messages = session.history[:-2]
        recent_messages = session.history[-2:]

        old_dicts = [{"role": m.role, "content": m.content} for m in old_messages]
        prompt = build_condensation_prompt(old_dicts)

        try:
            summary = self._generate_fn("summarizer", prompt)
            if not summary or not summary.strip():
                return session
        except Exception:
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
        session.history = recent_messages
        session.token_stats.input_tokens = max(
            0, session.token_stats.input_tokens - old_tokens
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
            return session

        prompt = build_fallback_gist_prompt(response_text)

        try:
            summary = self._generate_fn("summarizer", prompt)
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
            pass

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from the store."""
        return self._store.delete(session_id)

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List sessions (metadata only)."""
        return self._store.list_sessions(limit=limit, offset=offset)
