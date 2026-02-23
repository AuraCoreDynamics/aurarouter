from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Message:
    """A single message in a session's conversation history."""
    role: str              # "user", "assistant", "system"
    content: str
    timestamp: str = ""    # ISO 8601, auto-set if empty
    model_id: str = ""     # Which model produced this (empty for user/system)
    tokens: int = 0        # Token count for this message

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class Gist:
    """A condensed summary that replaces raw message history."""
    source_role: str         # Role that generated the original content
    source_model_id: str     # Model that generated the original content
    summary: str             # The condensed summary text
    timestamp: str = ""      # ISO 8601, auto-set if empty
    replaces_count: int = 0  # Number of raw messages this gist replaces

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Gist:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class TokenStats:
    """Real-time token usage tracking for a session."""
    input_tokens: int = 0
    output_tokens: int = 0
    context_limit: int = 0   # Model's total context window (0 = unknown)

    @property
    def total_used(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int:
        if self.context_limit == 0:
            return 0
        return max(0, self.context_limit - self.total_used)

    @property
    def pressure(self) -> float:
        """Context pressure as a ratio 0.0-1.0. Returns 0.0 if limit is unknown."""
        if self.context_limit == 0:
            return 0.0
        return min(1.0, self.total_used / self.context_limit)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_limit": self.context_limit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TokenStats:
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            context_limit=data.get("context_limit", 0),
        )


@dataclass
class Session:
    """A stateful conversation session with message history and shared context."""
    session_id: str = ""
    history: list[Message] = field(default_factory=list)
    shared_context: list[Gist] = field(default_factory=list)
    token_stats: TokenStats = field(default_factory=TokenStats)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def add_message(self, message: Message) -> None:
        """Append a message and update token stats and timestamp."""
        self.history.append(message)
        self.token_stats.input_tokens += message.tokens
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if "iteration_count" not in self.metadata:
            self.metadata["iteration_count"] = 0
        self.metadata["iteration_count"] += 1

    def add_gist(self, gist: Gist) -> None:
        """Add a gist to the shared context."""
        self.shared_context.append(gist)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_messages_as_dicts(self) -> list[dict]:
        """Return history as list of {"role": ..., "content": ...} dicts."""
        return [{"role": m.role, "content": m.content} for m in self.history]

    def get_context_prefix(self) -> str:
        """Build a context prefix from shared gists for injection into prompts."""
        if not self.shared_context:
            return ""
        parts = ["[Prior Context]"]
        for gist in self.shared_context:
            parts.append(f"- {gist.summary}")
        parts.append("[End Prior Context]\n")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "history": [m.to_dict() for m in self.history],
            "shared_context": [g.to_dict() for g in self.shared_context],
            "token_stats": self.token_stats.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Session:
        return cls(
            session_id=data.get("session_id", ""),
            history=[Message.from_dict(m) for m in data.get("history", [])],
            shared_context=[Gist.from_dict(g) for g in data.get("shared_context", [])],
            token_stats=TokenStats.from_dict(data.get("token_stats", {})),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
