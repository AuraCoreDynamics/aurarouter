"""Config migration: old format -> new format with catalog section."""

from typing import Any

import copy

import yaml


def migrate_config(config_data: dict) -> tuple[dict, list[str]]:
    """Migrate an old-format config dict to include catalog entries.

    Args:
        config_data: The loaded config dict (from YAML).

    Returns:
        A tuple of (migrated_config, report_lines).
        The migrated config is a NEW dict (input is not modified).
        report_lines describe what was changed.

    Migration rules:
    1. If "catalog" section missing -> create it
    2. If "grid_services.endpoints" exists -> create catalog service entries
       for each endpoint (artifact_id = endpoint name, kind = service)
    3. If "system.active_analyzer" missing -> set to "aurarouter-default"
    4. Do NOT remove or modify "models" or "grid_services" sections
    """
    migrated = copy.deepcopy(config_data)
    report: list[str] = []

    # Ensure catalog section
    if "catalog" not in migrated:
        migrated["catalog"] = {}
        report.append("Added empty 'catalog' section")

    # Migrate grid_services endpoints to catalog service entries
    grid_cfg = migrated.get("grid_services", {})
    endpoints = grid_cfg.get("endpoints", [])
    for ep in endpoints:
        name = ep.get("name", "")
        url = ep.get("url", "")
        if name and name not in migrated["catalog"]:
            migrated["catalog"][name] = {
                "kind": "service",
                "display_name": name,
                "description": "MCP service (migrated from grid_services)",
                "provider": name,
                "endpoint": url,
                "protocol": "mcp",
                "auto_sync_models": grid_cfg.get("auto_sync_models", False),
                "health_check": True,
            }
            report.append(f"Migrated grid service '{name}' ({url}) to catalog")

    # Ensure system.active_analyzer
    system = migrated.setdefault("system", {})
    if "active_analyzer" not in system:
        system["active_analyzer"] = "aurarouter-default"
        report.append("Set default active_analyzer to 'aurarouter-default'")

    if not report:
        report.append("No migration needed — config is already current")

    return migrated, report


def migrate_config_file(config_path: str, dry_run: bool = False) -> list[str]:
    """Migrate a YAML config file in-place (or dry-run).

    Returns report lines.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    migrated, report = migrate_config(data)

    if not dry_run and report[0] != "No migration needed — config is already current":
        with open(config_path, "w") as f:
            yaml.dump(migrated, f, default_flow_style=False, sort_keys=False)
        report.append(f"Wrote migrated config to {config_path}")
    elif dry_run:
        report.append("(dry-run — no changes written)")

    return report
