"""
AuraGrid manifest management for AuraRouter.

Builds AppManifest-compatible dicts matching C# schema:
  AppManifest → MasDefinition → PythonMasConfig

Field names use PascalCase to match C# [JsonPropertyName] defaults.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_version() -> str:
    """Read version from package metadata, fallback to pyproject.toml, then default."""
    try:
        from importlib.metadata import version
        return version("aurarouter")
    except Exception as ex:  # noqa: F841
        logger.debug("auragrid.manifest._get_version_error", exc_info=True)
    try:
        import tomllib
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return data.get("project", {}).get("version", "0.5.5")
    except Exception as ex:  # noqa: F841
        logger.debug("auragrid.manifest._get_version_error", exc_info=True)
    return "0.5.5"


class ManifestBuilder:
    """Build AuraGrid AppManifest dicts for AuraRouter.

    Output matches C# AppManifest record (PascalCase field names):
    - AppId, Name, Version, Description
    - Services: list of MasDefinition (MasId, DisplayName, Mode, Runtime, PythonConfig)
    """

    def __init__(
        self,
        app_id: str = "aurarouter-v2",
        name: str = "AuraRouter",
        version: Optional[str] = None,
    ):
        self.app_id = app_id
        self.name = name
        self.version = version or _get_version()
        self.services: List[Dict[str, Any]] = []

    def add_service(
        self,
        mas_id: str,
        display_name: str,
        mode: str = "Distributed",
        script_path: str = "src/aurarouter/__main__.py",
        arguments: str = "--config auraconfig.yaml",
        managed_venv_name: str = "aurarouter",
        requirements_file: str = "requirements.txt",
    ) -> "ManifestBuilder":
        """Add a MAS service definition."""
        self.services.append({
            "MasId": mas_id,
            "DisplayName": display_name,
            "Mode": mode,
            "Runtime": "Python",
            "PythonConfig": {
                "ScriptPath": script_path,
                "Arguments": arguments,
                "WorkingDirectory": ".",
                "ManagedVenvName": managed_venv_name,
                "RequirementsFile": requirements_file,
            },
        })
        return self

    def build(self) -> Dict[str, Any]:
        """Build the AppManifest dictionary (PascalCase, C#-compatible)."""
        return {
            "AppId": self.app_id,
            "Name": self.name,
            "Version": self.version,
            "Description": "Multi-model routing fabric for local and cloud LLMs",
            "IsDefault": True,
            "Services": self.services,
            "GuiEntryPoints": [],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON."""
        return json.dumps(self.build(), indent=indent)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


def create_default_manifest() -> Dict[str, Any]:
    """
    Create the default AuraRouter manifest for AuraGrid.

    Returns:
        AppManifest dictionary matching C# schema.
    """
    builder = ManifestBuilder()
    builder.add_service(
        mas_id="aurarouter-node",
        display_name="AuraRouter Inference Node",
        mode="Distributed",
    )
    return builder.build()
