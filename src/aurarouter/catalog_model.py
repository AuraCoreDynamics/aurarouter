"""Unified artifact catalog domain model.

Supports three artifact kinds: model, service, and analyzer.
Provides typed data structures for the catalog subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArtifactKind(str, Enum):
    MODEL = "model"
    SERVICE = "service"
    ANALYZER = "analyzer"


@dataclass
class CatalogArtifact:
    """A single artifact known to the catalog.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        kind: The artifact kind (model, service, or analyzer).
        display_name: Human-readable name.
        description: Optional longer description.
        provider: Provider or origin identifier.
        version: Version string.
        tags: Freeform tags for filtering.
        capabilities: Declared capabilities for query matching.
        status: Lifecycle status (default: ``"registered"``).
        spec: Kind-specific configuration fields.
    """

    artifact_id: str
    kind: ArtifactKind
    display_name: str
    description: str = ""
    provider: str = ""
    version: str = ""
    tags: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    supported_intents: list[str] = field(default_factory=list)
    status: str = "registered"
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def is_remote(self) -> bool:
        """True if this artifact has an MCP endpoint in its spec."""
        return self.spec.get("mcp_endpoint") is not None

    def to_dict(self) -> dict:
        """Serialize. Kind-specific spec fields merge at top level."""
        d: dict[str, Any] = {
            "kind": self.kind.value,
            "display_name": self.display_name,
        }
        if self.description:
            d["description"] = self.description
        if self.provider:
            d["provider"] = self.provider
        if self.version:
            d["version"] = self.version
        if self.tags:
            d["tags"] = self.tags
        if self.capabilities:
            d["capabilities"] = self.capabilities
        if self.supported_intents:
            d["supported_intents"] = self.supported_intents
        if self.status != "registered":
            d["status"] = self.status
        if self.spec:
            d.update(self.spec)
        return d

    @classmethod
    def from_dict(cls, artifact_id: str, data: dict) -> CatalogArtifact:
        """Deserialize from a flat dict (e.g. from YAML config)."""
        kind = ArtifactKind(data.get("kind", "model"))
        known = {
            "kind", "display_name", "description", "provider", "version",
            "tags", "capabilities", "supported_intents", "status",
        }
        spec = {k: v for k, v in data.items() if k not in known}
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            display_name=data.get("display_name", artifact_id),
            description=data.get("description", ""),
            provider=data.get("provider", ""),
            version=data.get("version", ""),
            tags=data.get("tags", []),
            capabilities=data.get("capabilities", []),
            supported_intents=data.get("supported_intents", []),
            status=data.get("status", "registered"),
            spec=spec,
        )
