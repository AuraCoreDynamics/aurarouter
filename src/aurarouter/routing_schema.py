"""Schema validation and normalization for AuraRouter v2.1/v2.2 routing config sections.

Mirrors ``analyzer_schema.py``: validation is warn-only and coerces/clamps
malformed values to safe defaults so invalid YAML inputs never crash hot-path
casts inside ``ComputeFabric``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.RoutingSchema")

VALID_ROUTING_STACK_MODES = {"off", "shadow", "active"}


@dataclass
class RoutingConfigValidation:
    """Aggregated validation result for v2 routing config sections."""

    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _coerce_bool(val: Any, default: bool, field_path: str, warnings: list[str]) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    warnings.append(f"{field_path}={val!r} is not a bool; using default {default}")
    return default


def _coerce_float(
    val: Any,
    default: float,
    field_path: str,
    warnings: list[str],
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    strict_positive: bool = False,
) -> float:
    if isinstance(val, bool) or val is None:
        if isinstance(val, bool):
            warnings.append(f"{field_path}={val!r} is bool, expected float; using default {default}")
        return default
    try:
        num = float(val)
    except (TypeError, ValueError):
        warnings.append(f"{field_path}={val!r} is not numeric; using default {default}")
        return default

    if strict_positive and num <= 0.0:
        warnings.append(f"{field_path}={num} must be > 0; using default {default}")
        return default
    if min_val is not None and num < min_val:
        warnings.append(f"{field_path}={num} < {min_val}; clamping to {min_val}")
        num = min_val
    if max_val is not None and num > max_val:
        warnings.append(f"{field_path}={num} > {max_val}; clamping to {max_val}")
        num = max_val
    return num


def _coerce_int(
    val: Any,
    default: int | None,
    field_path: str,
    warnings: list[str],
    *,
    min_val: int = 0,
    strict_positive: bool = False,
) -> int | None:
    if val is None:
        return default
    if isinstance(val, bool):
        warnings.append(f"{field_path}={val!r} is bool, expected int; using default {default}")
        return default
    try:
        num = int(val)
    except (TypeError, ValueError):
        warnings.append(f"{field_path}={val!r} is not an integer; using default {default}")
        return default

    if strict_positive and num <= 0:
        warnings.append(f"{field_path}={num} must be > 0; ignoring invalid quota")
        return default
    if num < min_val:
        warnings.append(f"{field_path}={num} < {min_val}; clamping to {min_val}")
        return min_val
    return num


def _normalize_quota_map(
    raw_map: Any,
    default_warn: float,
    section_path: str,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    if raw_map is None:
        return {}
    if not isinstance(raw_map, dict):
        warnings.append(f"{section_path} must be a mapping; ignoring {type(raw_map).__name__}")
        return {}

    cleaned: dict[str, dict[str, Any]] = {}
    for name, q_raw in raw_map.items():
        if not isinstance(q_raw, dict):
            warnings.append(f"{section_path}.{name} must be a dict; ignoring")
            continue
        entry: dict[str, Any] = {}
        for q_key in ("rpm", "itpm", "otpm"):
            if q_key in q_raw and q_raw[q_key] is not None:
                val = _coerce_int(
                    q_raw[q_key],
                    None,
                    f"{section_path}.{name}.{q_key}",
                    warnings,
                    strict_positive=True,
                )
                if val is not None:
                    entry[q_key] = val
        if "warning_threshold" in q_raw and q_raw["warning_threshold"] is not None:
            entry["warning_threshold"] = _coerce_float(
                q_raw["warning_threshold"],
                default_warn,
                f"{section_path}.{name}.warning_threshold",
                warnings,
                min_val=0.0,
                max_val=1.0,
            )
        cleaned[str(name)] = entry
    return cleaned


def normalize_rate_limits_config(
    config_dict: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize ``rate_limits`` config section."""
    local_warnings: list[str] = [] if warnings is None else warnings
    raw = config_dict.get("rate_limits") if isinstance(config_dict, dict) else None
    if raw is not None and not isinstance(raw, dict):
        local_warnings.append("rate_limits must be a dict; falling back to defaults")
        raw = {}
    cfg = raw or {}

    default_warn = _coerce_float(
        cfg.get("default_warning_threshold"),
        0.80,
        "rate_limits.default_warning_threshold",
        local_warnings,
        min_val=0.0,
        max_val=1.0,
    )
    replica_count = _coerce_int(
        cfg.get("replica_count"),
        1,
        "rate_limits.replica_count",
        local_warnings,
        min_val=1,
        strict_positive=True,
    )
    result = {
        "enabled": _coerce_bool(cfg.get("enabled"), True, "rate_limits.enabled", local_warnings),
        "default_warning_threshold": default_warn,
        "replica_count": replica_count or 1,
        "emit_ietf_headers": _coerce_bool(
            cfg.get("emit_ietf_headers"), False, "rate_limits.emit_ietf_headers", local_warnings
        ),
        "providers": _normalize_quota_map(
            cfg.get("providers"), default_warn, "rate_limits.providers", local_warnings
        ),
        "models": _normalize_quota_map(
            cfg.get("models"), default_warn, "rate_limits.models", local_warnings
        ),
    }
    if warnings is None:
        for w in local_warnings:
            logger.warning("config schema warning: %s", w)
    return result


def normalize_routing_stack_config(
    config_dict: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize ``routing_stack`` config section."""
    local_warnings: list[str] = [] if warnings is None else warnings
    raw = config_dict.get("routing_stack") if isinstance(config_dict, dict) else None
    has_section = isinstance(raw, dict)
    if raw is not None and not has_section:
        local_warnings.append("routing_stack must be a dict; falling back to defaults")
        raw = {}
    cfg = raw or {}

    default_mode = "shadow" if has_section else "off"
    raw_mode = cfg.get("mode", default_mode)
    mode = str(raw_mode).lower() if isinstance(raw_mode, str) else ""
    if mode not in VALID_ROUTING_STACK_MODES:
        local_warnings.append(
            f"routing_stack.mode={raw_mode!r} is invalid (expected one of {sorted(VALID_ROUTING_STACK_MODES)}); "
            f"using {default_mode!r}"
        )
        mode = default_mode

    result = {
        "mode": mode,
        "bandit_enabled": _coerce_bool(
            cfg.get("bandit_enabled"), True, "routing_stack.bandit_enabled", local_warnings
        ),
        "cache_affinity_enabled": _coerce_bool(
            cfg.get("cache_affinity_enabled"),
            True,
            "routing_stack.cache_affinity_enabled",
            local_warnings,
        ),
        "cost_lambda_max": _coerce_float(
            cfg.get("cost_lambda_max"),
            0.35,
            "routing_stack.cost_lambda_max",
            local_warnings,
            min_val=0.0,
            max_val=10.0,
        ),
        "cost_elasticity_kappa": _coerce_float(
            cfg.get("cost_elasticity_kappa"),
            0.32,
            "routing_stack.cost_elasticity_kappa",
            local_warnings,
            min_val=0.0,
            max_val=5.0,
        ),
    }
    if warnings is None:
        for w in local_warnings:
            logger.warning("config schema warning: %s", w)
    return result


def normalize_failover_projection_config(
    config_dict: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize ``sessions.failover_projection`` config section."""
    local_warnings: list[str] = [] if warnings is None else warnings
    sessions_cfg = config_dict.get("sessions") if isinstance(config_dict, dict) else None
    raw = sessions_cfg.get("failover_projection") if isinstance(sessions_cfg, dict) else None
    if raw is not None and not isinstance(raw, dict):
        local_warnings.append("sessions.failover_projection must be a dict; falling back to defaults")
        raw = {}
    cfg = raw or {}

    head_ratio = _coerce_float(
        cfg.get("active_turn_head_keep_ratio"),
        0.35,
        "sessions.failover_projection.active_turn_head_keep_ratio",
        local_warnings,
        min_val=0.01,
        max_val=0.95,
    )
    tail_ratio = _coerce_float(
        cfg.get("active_turn_tail_keep_ratio"),
        0.45,
        "sessions.failover_projection.active_turn_tail_keep_ratio",
        local_warnings,
        min_val=0.01,
        max_val=0.95,
    )
    reserved_out = _coerce_int(
        cfg.get("reserved_output_tokens"),
        2048,
        "sessions.failover_projection.reserved_output_tokens",
        local_warnings,
        min_val=0,
    )

    result = {
        "enabled": _coerce_bool(
            cfg.get("enabled"), True, "sessions.failover_projection.enabled", local_warnings
        ),
        "safety_margin_ratio": _coerce_float(
            cfg.get("safety_margin_ratio"),
            0.08,
            "sessions.failover_projection.safety_margin_ratio",
            local_warnings,
            min_val=0.0,
            max_val=0.85,
        ),
        "reserved_output_tokens": 2048 if reserved_out is None else reserved_out,
        "active_turn_head_keep_ratio": head_ratio,
        "active_turn_tail_keep_ratio": tail_ratio,
        "inject_failover_header": _coerce_bool(
            cfg.get("inject_failover_header"),
            True,
            "sessions.failover_projection.inject_failover_header",
            local_warnings,
        ),
    }
    if warnings is None:
        for w in local_warnings:
            logger.warning("config schema warning: %s", w)
    return result


def normalize_streaming_config(
    config_dict: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize ``streaming`` config section."""
    local_warnings: list[str] = [] if warnings is None else warnings
    raw = config_dict.get("streaming") if isinstance(config_dict, dict) else None
    if raw is not None and not isinstance(raw, dict):
        local_warnings.append("streaming must be a dict; falling back to defaults")
        raw = {}
    cfg = raw or {}

    result = {
        "ttft_timeout_seconds": _coerce_float(
            cfg.get("ttft_timeout_seconds"),
            15.0,
            "streaming.ttft_timeout_seconds",
            local_warnings,
            strict_positive=True,
        ),
        "saturated_ttft_timeout_seconds": _coerce_float(
            cfg.get("saturated_ttft_timeout_seconds"),
            5.0,
            "streaming.saturated_ttft_timeout_seconds",
            local_warnings,
            strict_positive=True,
        ),
    }
    if warnings is None:
        for w in local_warnings:
            logger.warning("config schema warning: %s", w)
    return result


def validate_v2_routing_config(config_dict: dict[str, Any] | None) -> RoutingConfigValidation:
    """Validate all v2.1/v2.2 routing configuration sections (warn-only)."""
    warnings: list[str] = []
    errors: list[str] = []
    if config_dict is not None and not isinstance(config_dict, dict):
        errors.append("root configuration must be a dict")
        return RoutingConfigValidation(valid=False, warnings=warnings, errors=errors)

    normalize_rate_limits_config(config_dict, warnings=warnings)
    normalize_routing_stack_config(config_dict, warnings=warnings)
    normalize_failover_projection_config(config_dict, warnings=warnings)
    normalize_streaming_config(config_dict, warnings=warnings)
    return RoutingConfigValidation(
        valid=(len(errors) == 0 and len(warnings) == 0),
        warnings=warnings,
        errors=errors,
    )

