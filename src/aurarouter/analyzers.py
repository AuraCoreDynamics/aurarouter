"""Built-in analyzers for AuraRouter.

Provides factory functions for creating default analyzer artifacts.
"""

from __future__ import annotations

from aurarouter.catalog_model import ArtifactKind, CatalogArtifact


def create_default_analyzer() -> CatalogArtifact:
    """Built-in analyzer wrapping existing intent -> triage -> execute logic."""
    return CatalogArtifact(
        artifact_id="aurarouter-default",
        kind=ArtifactKind.ANALYZER,
        display_name="AuraRouter Default",
        description="Intent classification with complexity-based triage routing",
        provider="aurarouter",
        version="1.0",
        capabilities=["code", "reasoning", "review", "planning"],
        spec={
            "analyzer_kind": "intent_triage",
            "role_bindings": {
                "simple_code": "coding",
                "complex_reasoning": "reasoning",
                "review": "reviewer",
            },
        },
    )
