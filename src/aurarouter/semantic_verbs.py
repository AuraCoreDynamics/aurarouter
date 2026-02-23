"""Semantic verb registry for role-name resolution.

Maps synonyms to canonical role names so the intent classifier's output
is normalised before routing.  E.g. "programming" -> "coding".

Built-in verbs can be extended via ``semantic_verbs`` in auraconfig.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SemanticVerb:
    """A canonical role with its known synonyms."""

    role: str
    description: str
    synonyms: list[str] = field(default_factory=list)
    required: bool = False


# ------------------------------------------------------------------
# Built-in verb table
# ------------------------------------------------------------------

BUILTIN_VERBS: list[SemanticVerb] = [
    SemanticVerb(
        role="router",
        description="Intent classification and task triage",
        synonyms=["classifier", "triage", "intent"],
        required=True,
    ),
    SemanticVerb(
        role="reasoning",
        description="Multi-step planning and architectural reasoning",
        synonyms=["planner", "architect", "planning"],
        required=True,
    ),
    SemanticVerb(
        role="coding",
        description="Code generation and implementation",
        synonyms=["code generation", "programming", "developer"],
        required=True,
    ),
    SemanticVerb(
        role="summarization",
        description="Text summarization and digest generation",
        synonyms=["summarize", "tldr", "digest"],
    ),
    SemanticVerb(
        role="analysis",
        description="Data analysis and evaluation",
        synonyms=["analyze", "evaluate", "assess"],
    ),
]

# Quick lookup: synonym -> canonical role
_BUILTIN_INDEX: dict[str, str] = {}
for _v in BUILTIN_VERBS:
    _BUILTIN_INDEX[_v.role.lower()] = _v.role
    for _s in _v.synonyms:
        _BUILTIN_INDEX[_s.lower()] = _v.role


def resolve_synonym(
    verb: str,
    custom_verbs: dict[str, list[str]] | None = None,
) -> str:
    """Map a verb (possibly a synonym) to its canonical role name.

    Parameters
    ----------
    verb:
        The role name or synonym returned by the classifier.
    custom_verbs:
        Optional user-defined mapping ``{role: [synonym, ...]}``.

    Returns
    -------
    The canonical role name, or *verb* unchanged if no mapping exists.
    """
    key = verb.strip().lower()

    # Check custom verbs first (user overrides).
    if custom_verbs:
        for role, synonyms in custom_verbs.items():
            if key == role.lower():
                return role
            for syn in synonyms:
                if key == syn.lower():
                    return role

    # Fall back to built-in index.
    return _BUILTIN_INDEX.get(key, verb)


def get_known_roles() -> list[str]:
    """Return all known canonical role names (built-in)."""
    return [v.role for v in BUILTIN_VERBS]


def get_required_roles() -> list[str]:
    """Return the role names that must be configured for routing to work."""
    return [v.role for v in BUILTIN_VERBS if v.required]
