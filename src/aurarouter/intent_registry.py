"""Intent Registry for AuraRouter.

Central registry of all known intents (built-in + analyzer-declared).
The registry is the single source of truth for intent classification
choices and intent-to-role resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader

logger = get_logger("AuraRouter.IntentRegistry")


@dataclass
class IntentDefinition:
    """A single intent the system can classify."""

    name: str  # e.g., "SIMPLE_CODE", "sar_processing"
    description: str  # One-line description for the classifier prompt
    target_role: str  # Role to route to (e.g., "coding", "reasoning")
    source: str  # "builtin" | analyzer_id that declared it
    priority: int = 0  # Higher = preferred on conflict (builtin=0, analyzer=10)


class IntentRegistry:
    """Central registry of all known intents (built-in + analyzer-declared)."""

    BUILTIN_INTENTS: ClassVar[list[IntentDefinition]] = [
        IntentDefinition(
            name="DIRECT",
            description="Simple questions, jokes, or single-turn tasks that don't require code or multi-step reasoning",
            target_role="coding",
            source="builtin",
            priority=0,
        ),
        IntentDefinition(
            name="SIMPLE_CODE",
            description="Straightforward code generation or implementation tasks",
            target_role="coding",
            source="builtin",
            priority=0,
        ),
        IntentDefinition(
            name="COMPLEX_REASONING",
            description="Multi-step reasoning, architectural design, or complex analysis tasks",
            target_role="reasoning",
            source="builtin",
            priority=0,
        ),
    ]

    def __init__(self) -> None:
        self._intents: dict[str, IntentDefinition] = {}
        # Register built-in intents.
        for intent in self.BUILTIN_INTENTS:
            self._intents[intent.name] = intent

    def register(self, intent: IntentDefinition) -> None:
        """Register a single intent definition.

        If an intent with the same name already exists, the one with
        higher priority wins.  On equal priority the new registration
        replaces the old one.
        """
        existing = self._intents.get(intent.name)
        if existing is not None and existing.priority > intent.priority:
            logger.debug(
                "Skipping registration of intent '%s' from '%s' "
                "(existing from '%s' has higher priority %d > %d)",
                intent.name,
                intent.source,
                existing.source,
                existing.priority,
                intent.priority,
            )
            return
        self._intents[intent.name] = intent

    def unregister_by_source(self, source: str) -> None:
        """Remove all intents registered by a given source.

        Built-in intents (source ``"builtin"``) are never removed.
        """
        to_remove = [
            name
            for name, defn in self._intents.items()
            if defn.source == source and defn.source != "builtin"
        ]
        for name in to_remove:
            del self._intents[name]

    def register_from_role_bindings(
        self, analyzer_id: str, role_bindings: dict[str, str]
    ) -> None:
        """Convert an analyzer's ``role_bindings`` dict into intent definitions.

        Each key in *role_bindings* is treated as an intent name and its
        value as the target role.  The analyzer is recorded as the source
        with priority 10 (higher than built-in).

        To avoid duplicates on re-registration, ``unregister_by_source``
        is called first.
        """
        self.unregister_by_source(analyzer_id)
        for intent_name, target_role in role_bindings.items():
            self.register(
                IntentDefinition(
                    name=intent_name,
                    description=f"Intent '{intent_name}' declared by analyzer '{analyzer_id}'",
                    target_role=target_role,
                    source=analyzer_id,
                    priority=10,
                )
            )

    def get_all(self) -> list[IntentDefinition]:
        """Return all registered intent definitions."""
        return list(self._intents.values())

    def get_by_name(self, name: str) -> IntentDefinition | None:
        """Look up an intent by name (case-sensitive)."""
        return self._intents.get(name)

    def get_intent_names(self) -> list[str]:
        """Return all registered intent names."""
        return list(self._intents.keys())

    def resolve_role(self, intent_name: str) -> str | None:
        """Return the target role for an intent, or None if unknown."""
        defn = self._intents.get(intent_name)
        return defn.target_role if defn is not None else None

    def build_classifier_choices(self) -> str:
        """Build a formatted string of intent choices for the classifier prompt.

        Each intent is listed with its description so the LLM understands
        what each classification means.
        """
        lines: list[str] = []
        for defn in self._intents.values():
            lines.append(f'- {defn.name}: {defn.description}')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_intent_registry(config: ConfigLoader) -> IntentRegistry:
    """Build an IntentRegistry populated with built-in and analyzer intents.

    1. Creates a registry with built-in intents.
    2. Looks up the active analyzer via ``config.get_active_analyzer()``
       and ``config.catalog_get()``.
    3. If the analyzer has ``role_bindings`` in its spec, calls
       ``register_from_role_bindings()``.
    4. Returns the populated registry.
    """
    from aurarouter.catalog_model import CatalogArtifact

    registry = IntentRegistry()

    active_id = config.get_active_analyzer()
    if not active_id:
        return registry

    artifact_data = config.catalog_get(active_id)
    if artifact_data is None:
        return registry

    artifact = CatalogArtifact.from_dict(active_id, artifact_data)
    role_bindings = artifact.spec.get("role_bindings")
    if role_bindings and isinstance(role_bindings, dict):
        registry.register_from_role_bindings(active_id, role_bindings)

    return registry
