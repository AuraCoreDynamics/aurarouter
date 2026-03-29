"""Analyzer spec schema validation.

Provides validation functions for analyzer-specific spec fields
within the unified artifact catalog. Validation is warn-only for
backwards compatibility — callers decide whether to reject or proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.AnalyzerSchema")

REQUIRED_ANALYZER_FIELDS = {"analyzer_kind"}
OPTIONAL_ANALYZER_FIELDS = {
    "role_bindings",
    "mcp_endpoint",
    "mcp_tool_name",
    "capabilities",
    "description",
}


@dataclass
class AnalyzerSpecValidation:
    """Result of validating an analyzer's spec dict."""

    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    declared_intents: list[str] = field(default_factory=list)


def validate_analyzer_spec(
    spec: dict,
    available_roles: list[str] | None = None,
) -> AnalyzerSpecValidation:
    """Validate an analyzer's spec dict.

    Checks:
    - Required fields present (``analyzer_kind``).
    - ``role_bindings`` values reference known roles (if *available_roles*
      is provided).
    - ``role_bindings`` keys are valid identifier strings.
    - ``mcp_endpoint`` is a valid URL if present.
    - ``capabilities`` is a list of strings if present.

    Parameters
    ----------
    spec:
        The flat spec dict from a ``CatalogArtifact`` with ``kind=analyzer``.
    available_roles:
        Optional list of currently configured role names.  When provided,
        role binding targets are checked against this list.

    Returns
    -------
    AnalyzerSpecValidation
        Aggregated validation result with errors, warnings, and extracted
        declared intents.
    """
    errors: list[str] = []
    warnings: list[str] = []
    declared_intents: list[str] = []

    # --- Required fields ---------------------------------------------------
    for req in REQUIRED_ANALYZER_FIELDS:
        if req not in spec:
            errors.append(f"missing required field: {req}")

    # --- role_bindings -----------------------------------------------------
    role_bindings = spec.get("role_bindings")
    if role_bindings is not None:
        if not isinstance(role_bindings, dict):
            errors.append("role_bindings must be a dict")
        else:
            for key, target in role_bindings.items():
                # Keys must be valid identifiers
                if not isinstance(key, str) or not key.isidentifier():
                    warnings.append(
                        f"role_bindings key {key!r} is not a valid identifier"
                    )
                else:
                    declared_intents.append(key)

                # Values must be strings referencing roles
                if not isinstance(target, str):
                    warnings.append(
                        f"role_bindings[{key!r}] target must be a string"
                    )
                elif available_roles is not None and target not in available_roles:
                    warnings.append(
                        f"role_bindings[{key!r}] targets role {target!r} "
                        f"which is not in configured roles"
                    )

    # --- mcp_endpoint ------------------------------------------------------
    mcp_endpoint = spec.get("mcp_endpoint")
    if mcp_endpoint is not None:
        if not isinstance(mcp_endpoint, str):
            errors.append("mcp_endpoint must be a string")
        else:
            parsed = urlparse(mcp_endpoint)
            if not parsed.scheme and not parsed.netloc:
                errors.append(
                    f"mcp_endpoint {mcp_endpoint!r} is not a valid URL "
                    f"(missing both scheme and netloc)"
                )
            elif not parsed.scheme:
                errors.append(
                    f"mcp_endpoint {mcp_endpoint!r} is missing a URL scheme "
                    f"(e.g. http:// or https://)"
                )
            elif not parsed.netloc:
                warnings.append(
                    f"mcp_endpoint {mcp_endpoint!r} has no network location — "
                    f"this may be intentional for local transports"
                )

    # --- capabilities (in spec, not the top-level artifact field) ----------
    caps_in_spec = spec.get("capabilities")
    if caps_in_spec is not None:
        if not isinstance(caps_in_spec, list):
            warnings.append("capabilities must be a list")
        elif not all(isinstance(c, str) for c in caps_in_spec):
            warnings.append("all capabilities entries must be strings")

    valid = len(errors) == 0
    return AnalyzerSpecValidation(
        valid=valid,
        warnings=warnings,
        errors=errors,
        declared_intents=declared_intents,
    )


def extract_declared_intents(spec: dict) -> list[str]:
    """Extract intent names from an analyzer spec's role_bindings keys.

    Returns an empty list if role_bindings is missing or not a dict.
    """
    role_bindings = spec.get("role_bindings")
    if not isinstance(role_bindings, dict):
        return []
    return [k for k in role_bindings if isinstance(k, str)]
