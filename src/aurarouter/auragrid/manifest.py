"""
AuraGrid manifest management for AuraRouter.

Validates and builds AuraGrid manifests for deploying AuraRouter as a MAS.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class ManifestBuilder:
    """Build and validate AuraGrid manifests for AuraRouter."""

    def __init__(
        self,
        app_id: str = "aurarouter-v2",
        name: str = "AuraRouter",
        version: str = "0.2.0",
    ):
        """
        Initialize manifest builder.

        Args:
            app_id: Unique application ID for AuraGrid
            name: Human-readable name
            version: Semantic version
        """
        self.app_id = app_id
        self.name = name
        self.version = version
        self.services: List[Dict[str, Any]] = []

    def add_service(
        self,
        service_id: str,
        service_name: str,
        description: str,
        execution_mode: str = "Distributed",
    ) -> "ManifestBuilder":
        """
        Add a service to the manifest.

        Args:
            service_id: Unique service identifier
            service_name: Service class name
            description: Human-readable description
            execution_mode: "Distributed" or "CellSingleton"

        Returns:
            Self for chaining
        """
        self.services.append(
            {
                "id": service_id,
                "name": service_name,
                "description": description,
                "executionMode": execution_mode,
            }
        )
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the manifest dictionary.

        Returns:
            Manifest ready for serialization
        """
        return {
            "appid": self.app_id,
            "name": self.name,
            "version": self.version,
            "description": "Multi-model routing fabric for local and cloud LLMs",
            "startup": {
                "executable": "python",
                "args": [
                    "-m",
                    "aurarouter.auragrid.mas_host",
                    "startup",
                ],
            },
            "shutdown": {
                "executable": "python",
                "args": [
                    "-m",
                    "aurarouter.auragrid.mas_host",
                    "shutdown",
                ],
            },
            "services": self.services,
            "guiEntryPoints": [],
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize manifest to JSON.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.build(), indent=indent)

    def save(self, path: Path) -> None:
        """
        Save manifest to file.

        Args:
            path: File path to save to
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


def create_default_manifest() -> Dict[str, Any]:
    """
    Create the default AuraRouter manifest for AuraGrid.

    Returns:
        Manifest dictionary
    """
    builder = ManifestBuilder()

    # Add all four services
    builder.add_service(
        "router-service",
        "RouterService",
        "Intent classification routing",
        "Distributed",
    )
    builder.add_service(
        "reasoning-service",
        "ReasoningService",
        "Architectural planning",
        "Distributed",
    )
    builder.add_service(
        "coding-service",
        "CodingService",
        "Code generation",
        "Distributed",
    )
    builder.add_service(
        "unified-service",
        "UnifiedRouterService",
        "Unified intelligent code generation",
        "Distributed",
    )

    return builder.build()
