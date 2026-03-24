"""AuraRouter Example Provider -- external provider package template.

This module defines the entry point that AuraRouter's ProviderCatalog
calls to discover this provider.
"""

from __future__ import annotations

from aurarouter.providers.protocol import ProviderMetadata


def get_provider_metadata() -> ProviderMetadata:
    """Return metadata for this external provider.

    This function is referenced by the ``aurarouter.providers`` entry
    point in ``pyproject.toml``.
    """
    return ProviderMetadata(
        name="example",
        provider_type="mcp",
        version="0.1.0",
        description="Example AuraRouter external provider",
        command=["python", "-m", "aurarouter_example.server"],
        requires_config=[],
        homepage="",
    )
