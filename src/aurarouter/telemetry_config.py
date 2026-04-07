"""Telemetry opt-in/opt-out configuration reader for AuraCore Python projects.

Global config file: ~/.auracore/telemetry.json
Default: ALL external telemetry OFF (opt-out by default).
Local usage tracking (budget/stats for the user) is independent of telemetry flag.

This file is intentionally duplicated across Python projects — no shared package.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TELEMETRY_CONFIG_PATH = Path.home() / ".auracore" / "telemetry.json"

_TELEMETRY_ENV_VAR = "AURACORE_TELEMETRY_ENABLED"
_TRUTHY = {"1", "true", "yes"}
_FALSY = {"0", "false", "no"}


def _get_env_override() -> bool | None:
    val = os.environ.get(_TELEMETRY_ENV_VAR, "").strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return None  # not set or unrecognised — fall through

_DEFAULTS: dict = {
    "telemetry_version": 1,
    "enabled": False,
    "anonymous_usage_stats": False,
    "crash_reports": False,
    "local_usage_tracking": True,
    "opted_in_at": None,
    "opted_out_at": None,
}


def _load() -> dict:
    try:
        if _TELEMETRY_CONFIG_PATH.exists():
            data = json.loads(_TELEMETRY_CONFIG_PATH.read_text(encoding="utf-8"))
            return {**_DEFAULTS, **data}
    except Exception as exc:
        logger.debug("Could not read telemetry config from %s: %s", _TELEMETRY_CONFIG_PATH, exc)
    return dict(_DEFAULTS)


def is_external_telemetry_enabled() -> bool:
    """Returns True only if the user has explicitly enabled external telemetry."""
    override = _get_env_override()
    if override is not None:
        logger.info("Telemetry: %s via %s", "enabled" if override else "disabled", _TELEMETRY_ENV_VAR)
        return override
    return bool(_load().get("enabled", False))


def is_local_usage_tracking_enabled() -> bool:
    """Returns True if local usage tracking (budget/stats) is enabled. Default: True."""
    return bool(_load().get("local_usage_tracking", True))


def get_config() -> dict:
    """Return the full telemetry config dict."""
    return _load()
