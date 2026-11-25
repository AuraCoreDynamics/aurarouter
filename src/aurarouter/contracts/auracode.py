"""AuraCode domain analyzer contract.

Defines the intents, role mappings, capabilities, and deterministic BLACK
Task Group scoring-to-routing policy mapping that AuraCode and external
BLACK harnesses expect from their AuraRouter FMoE integration.
"""

from __future__ import annotations

from typing import Any

AURACODE_INTENTS: dict[str, str] = {
    "generate_code": "coding",
    "edit_code": "coding",
    "complete_code": "coding",
    "explain_code": "reasoning",
    "review": "reasoning",
    "chat": "reasoning",
    "plan": "reasoning",
    "refactor": "coding",
    "review_diff": "reasoning",
    "security_review": "reasoning",
    "generate_tests": "coding",
    "cross_file_edit": "coding",
    "architecture_trace": "reasoning",
    "ops_triage": "reasoning",
    "remediate": "coding",
}

AURACODE_CAPABILITIES: list[str] = [
    "code",
    "reasoning",
    "review",
    "planning",
    "diff_aware",
    "security_analysis",
    "test_aware",
    "cross_file",
    "ops_diagnostics",
]

# TG4: AuraCode must parse this block from GenerateResult.routing_context
# and surface it through ExecutionMetadata on EngineResponse.
AURACODE_ROUTING_CONTEXT_SCHEMA = {
    "strategy": "string — classification method used (vector/complexity/slm/federated)",
    "confidence_score": "float — pipeline confidence 0.0-1.0",
    "complexity_score": "int — prompt complexity 1-10",
    "selected_route": "string — role name routed to",
    "analyzer_chain": "list[str] — ordered analyzer IDs that ran",
    "intent": "string — final resolved intent",
    "hard_routed": "bool — whether cloud was bypassed",
    "simulated_cost_avoided": "float — estimated USD avoided by hard-routing; 0.0 if not hard-routed",
    "metadata": "dict — additional analyzer metadata",
}

_TIER_DEFAULT_SCORES: dict[str, int] = {
    "LOW": 2,
    "MEDIUM": 5,
    "HIGH": 8,
    "CRITICAL": 10,
}

_SECURITY_PROFILES = frozenset({
    "security",
    "secops",
    "devsecops",
    "security_review",
    "security-specialist",
    "sovereignty-specialist",
    "adversarial-qa-assessor",
})

_ARCHITECTURE_PROFILES = frozenset({
    "architecture",
    "architect",
    "contract-architect",
    "system-architect",
    "design",
    "plan",
    "planning",
})


def create_auracode_analyzer_spec() -> dict:
    """Return the canonical analyzer spec for an AuraCode-compatible analyzer."""
    return {
        "analyzer_kind": "intent_triage",
        "role_bindings": AURACODE_INTENTS,
        "capabilities": AURACODE_CAPABILITIES,
    }


def _resolve_target_entry(scoring_entry: dict[str, Any], tg_id: str | None = None) -> dict[str, Any]:
    """Extract a specific Task Group dict when given an overall scoring.json document."""
    if not isinstance(scoring_entry, dict):
        raise TypeError("scoring_entry must be a dict")

    groups = scoring_entry.get("per_task_group") or scoring_entry.get("task_groups")
    if tg_id and isinstance(groups, list):
        target_norm = tg_id.strip().lower()
        for grp in groups:
            if not isinstance(grp, dict):
                continue
            cand_id = str(grp.get("tg_id") or grp.get("group_id") or "").strip().lower()
            if cand_id == target_norm:
                merged: dict[str, Any] = {}
                for inherited_key in ("sovereignty", "security_local_only", "recommended_profile", "profile"):
                    if inherited_key in scoring_entry:
                        merged[inherited_key] = scoring_entry[inherited_key]
                merged.update(grp)
                return merged
    return scoring_entry


def _normalize_complexity(entry: dict[str, Any]) -> tuple[int, str]:
    """Deterministically normalize complexity_score (1..10) and complexity_tier."""
    raw_tier_val = (
        entry.get("complexity_tier")
        or entry.get("complexity")
        or entry.get("overall_complexity")
    )
    tier: str | None = None
    if isinstance(raw_tier_val, str):
        cand = raw_tier_val.strip().upper()
        if cand in _TIER_DEFAULT_SCORES:
            tier = cand

    raw_cs = entry.get("complexity_score")
    raw_ts = entry.get("total_score")

    score: int | None = None
    if raw_cs is not None:
        val = float(raw_cs)
        score = max(1, min(10, int(round(val))))
    elif raw_ts is not None:
        val = float(raw_ts)
        if val <= 10.0:
            score = max(1, min(10, int(round(val))))
        elif val <= 20.0:
            score = max(1, min(10, int(round(val / 2.0))))
        else:
            score = max(1, min(10, int(round(val / 10.0))))
        if tier == "CRITICAL":
            score = max(score, 9)
        elif tier == "HIGH":
            score = max(score, 7)

    if score is None:
        resolved_tier = tier or "MEDIUM"
        return _TIER_DEFAULT_SCORES[resolved_tier], resolved_tier

    if tier is None:
        if score <= 3:
            tier = "LOW"
        elif score <= 6:
            tier = "MEDIUM"
        elif score <= 8:
            tier = "HIGH"
        else:
            tier = "CRITICAL"

    return score, tier


def _count_boundaries(entry: dict[str, Any]) -> int:
    """Count crossed architectural/module boundaries from int or sequence."""
    bc = entry.get("boundaries_crossed")
    if isinstance(bc, (list, tuple, set)):
        return len(bc)
    if bc is not None and not isinstance(bc, bool):
        return max(0, int(bc))
    return 0


def map_black_scoring_to_routing_policy(
    scoring_entry: dict[str, Any],
    *,
    tg_id: str | None = None,
) -> dict[str, Any]:
    """Deterministically map a BLACK Task Group scoring.json entry to an FMoE routing policy.

    Accepts either a per-Task-Group scoring dict or an overall ``scoring.json`` dict.
    """
    entry = _resolve_target_entry(scoring_entry, tg_id=tg_id)

    resolved_tg_id = str(
        entry.get("tg_id")
        or entry.get("group_id")
        or entry.get("taskgroup_id")
        or tg_id
        or ""
    ).strip()

    complexity_score, complexity_tier = _normalize_complexity(entry)
    boundaries_count = _count_boundaries(entry)

    raw_profile = str(
        entry.get("profile")
        or entry.get("recommended_profile")
        or "default"
    ).strip()
    prof_norm = raw_profile.lower()

    is_security = prof_norm in _SECURITY_PROFILES or "security" in prof_norm
    is_architecture = not is_security and (
        prof_norm in _ARCHITECTURE_PROFILES or "architect" in prof_norm
    )

    if is_security:
        recommended_intent = "security_review"
    elif is_architecture:
        recommended_intent = "plan" if prof_norm in ("plan", "planning") else "architecture_trace"
    elif boundaries_count >= 2:
        recommended_intent = "cross_file_edit"
    else:
        recommended_intent = "generate_code"

    if complexity_score >= 7 or complexity_tier in ("HIGH", "CRITICAL") or is_security or is_architecture:
        recommended_role = "reasoning"
    else:
        recommended_role = "coding"

    caps: list[str] = []

    def _add_cap(tag: str) -> None:
        if tag not in caps:
            caps.append(tag)

    if is_security:
        _add_cap("reasoning")
        _add_cap("review")
        _add_cap("security_analysis")
    elif is_architecture:
        _add_cap("reasoning")
        _add_cap("planning")
    else:
        _add_cap("code")
        if recommended_role == "reasoning":
            _add_cap("reasoning")

    if boundaries_count >= 2:
        _add_cap("cross_file")
    if "test" in prof_norm:
        _add_cap("test_aware")
    if "diff" in prof_norm:
        _add_cap("diff_aware")
    if "ops" in prof_norm:
        _add_cap("ops_diagnostics")

    sovereignty = str(entry.get("sovereignty") or "auto").strip().lower()
    security_local_only = bool(entry.get("security_local_only", False))
    explicit_local_only = bool(entry.get("local_only", False))
    is_local_only = (
        sovereignty in ("local_only", "strict")
        or security_local_only
        or explicit_local_only
    )

    max_remediation_attempts = 2 if complexity_tier == "CRITICAL" else 3

    return {
        "tg_id": resolved_tg_id,
        "complexity_score": complexity_score,
        "complexity_tier": complexity_tier,
        "recommended_intent": recommended_intent,
        "recommended_role": recommended_role,
        "required_capabilities": caps,
        "boundaries_crossed": boundaries_count,
        "profile": raw_profile,
        "sovereignty": sovereignty,
        "hard_routed": is_local_only,
        "local_only": is_local_only,
        "max_remediation_attempts": max_remediation_attempts,
    }
