"""Parsers for multi-protocol rate limit headers, quotas, and error payloads.

AuraRouter v2.0 §3.1, §3.2
"""
from __future__ import annotations

import email.utils
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from aurarouter.ratelimit.models import RateLimitSnapshot

# Regex for Go-style duration strings: 1h2m3s, 2m59.5s, 250ms, 500us
_GO_DURATION_RE = re.compile(
    r"(?:(?P<hours>[\d.]+)h)?"
    r"(?:(?P<minutes>[\d.]+)m(?!s))?"
    r"(?:(?P<seconds>[\d.]+)s)?"
    r"(?:(?P<milliseconds>[\d.]+)ms)?"
    r"(?:(?P<microseconds>[\d.]+)(?:µs|us))?"
)


def parse_duration_string(val: str | None) -> float | None:
    """Parse Go-style durations ('2m59.5s', '250ms') or numeric seconds into float seconds."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None

    # Fast path: plain float/integer seconds
    try:
        return max(0.0, float(val_str))
    except ValueError:
        pass

    # Go duration pattern
    match = _GO_DURATION_RE.fullmatch(val_str)
    if not match or not any(match.groups()):
        return None

    parts = match.groupdict()
    total_seconds = 0.0
    if parts.get("hours"):
        total_seconds += float(parts["hours"]) * 3600.0
    if parts.get("minutes"):
        total_seconds += float(parts["minutes"]) * 60.0
    if parts.get("seconds"):
        total_seconds += float(parts["seconds"])
    if parts.get("milliseconds"):
        total_seconds += float(parts["milliseconds"]) / 1000.0
    if parts.get("microseconds"):
        total_seconds += float(parts["microseconds"]) / 1_000_000.0

    return max(0.0, total_seconds)


def parse_rfc3339_reset(val: str | None, now_utc: datetime | None = None) -> float | None:
    """Parse an RFC 3339 timestamp string into delta seconds relative to now."""
    if not val:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None

    # Handle Z suffix for fromisoformat
    if val_str.endswith("Z") or val_str.endswith("z"):
        val_str = val_str[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(val_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        curr = now_utc if now_utc is not None else datetime.now(timezone.utc)
        return max(0.0, (dt - curr).total_seconds())
    except (ValueError, TypeError):
        # Fallback: maybe it's delta seconds
        return parse_duration_string(val_str)


def parse_retry_after(val: str | None, now_utc: datetime | None = None) -> float | None:
    """Parse HTTP Retry-After header (either delta-seconds or HTTP-date)."""
    if not val:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None

    # Check numeric seconds first
    try:
        sec = float(val_str)
        return max(0.0, sec)
    except ValueError:
        pass

    # Try IMF-fixdate / RFC 1123 date
    try:
        dt = email.utils.parsedate_to_datetime(val_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        curr = now_utc if now_utc is not None else datetime.now(timezone.utc)
        return max(0.0, (dt - curr).total_seconds())
    except Exception:
        return None


def parse_ietf_ratelimit_header(rate_limit_val: str, policy_val: str | None = None) -> dict[str, Any]:
    """Parse IETF RateLimit and RateLimit-Policy structured fields (RFC 8941).

    Example RateLimit-Policy: "rpm";q=500;w=60, "tpm";q=200000;w=60
    Example RateLimit: "rpm";r=42;t=18, "tpm";r=15400;t=18
    """
    res: dict[str, Any] = {}

    def parse_params(item_str: str) -> tuple[str, dict[str, str]]:
        parts = [p.strip() for p in item_str.split(";")]
        name = parts[0].strip().strip('"')
        params = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k.strip()] = v.strip().strip('"')
        return name, params

    if policy_val:
        for item in policy_val.split(","):
            if not item.strip():
                continue
            name, params = parse_params(item)
            q_val = params.get("q")
            if q_val and q_val.isdigit():
                q_int = int(q_val)
                if "rpm" in name.lower() or "req" in name.lower():
                    res["rpm_limit"] = q_int
                elif "tok" in name.lower() or "tpm" in name.lower():
                    res["itpm_limit"] = q_int

    if rate_limit_val:
        for item in rate_limit_val.split(","):
            if not item.strip():
                continue
            name, params = parse_params(item)
            r_val = params.get("r")
            t_val = params.get("t")

            rem_int = int(r_val) if r_val and r_val.isdigit() else None
            reset_s = float(t_val) if t_val else None

            if "rpm" in name.lower() or "req" in name.lower():
                if rem_int is not None:
                    res["rpm_remaining"] = rem_int
                if reset_s is not None:
                    res["rpm_reset_s"] = reset_s
            elif "tok" in name.lower() or "tpm" in name.lower():
                if rem_int is not None:
                    res["itpm_remaining"] = rem_int
                if reset_s is not None:
                    res["itpm_reset_s"] = reset_s

    return res


def format_ietf_ratelimit_headers(
    rpm_limit: int | None = None,
    rpm_remaining: int | None = None,
    rpm_reset_s: float | None = None,
    itpm_limit: int | None = None,
    itpm_remaining: int | None = None,
    itpm_reset_s: float | None = None,
) -> dict[str, str]:
    """Emit RFC 8941 compliant RateLimit and RateLimit-Policy header strings."""
    policies = []
    ratelimits = []

    if rpm_limit is not None and rpm_limit > 0:
        policies.append(f'"rpm";q={rpm_limit};w=60')
    if itpm_limit is not None and itpm_limit > 0:
        policies.append(f'"tpm";q={itpm_limit};w=60')

    if rpm_remaining is not None or rpm_reset_s is not None:
        rem_str = f"r={max(0, rpm_remaining)}" if rpm_remaining is not None else "r=0"
        reset_str = f"t={int(math.ceil(rpm_reset_s))}" if rpm_reset_s is not None else "t=60"
        ratelimits.append(f'"rpm";{rem_str};{reset_str}')

    if itpm_remaining is not None or itpm_reset_s is not None:
        rem_str = f"r={max(0, itpm_remaining)}" if itpm_remaining is not None else "r=0"
        reset_str = f"t={int(math.ceil(itpm_reset_s))}" if itpm_reset_s is not None else "t=60"
        ratelimits.append(f'"tpm";{rem_str};{reset_str}')

    headers: dict[str, str] = {}
    if policies:
        headers["RateLimit-Policy"] = ", ".join(policies)
    if ratelimits:
        headers["RateLimit"] = ", ".join(ratelimits)
    return headers


def parse_gemini_error_details(body: dict | str | None) -> dict[str, Any]:
    """Extract RetryInfo and QuotaFailure details from Google Gemini / Vertex 429 JSON."""
    res: dict[str, Any] = {}
    if not body:
        return res

    data = body
    if isinstance(body, str):
        try:
            data = json.loads(body)
        except Exception:
            return res

    if not isinstance(data, dict):
        return res

    error = data.get("error", data)
    details = error.get("details", []) if isinstance(error, dict) else []

    for d in details:
        if not isinstance(d, dict):
            continue
        type_str = str(d.get("@type", ""))

        # Check RetryInfo
        if "RetryInfo" in type_str or "retryDelay" in d:
            delay_str = str(d.get("retryDelay", "")).strip()
            if delay_str:
                res["retry_after_s"] = parse_duration_string(delay_str)

        # Check QuotaFailure
        if "QuotaFailure" in type_str or "violations" in d:
            for v in d.get("violations", []):
                if not isinstance(v, dict):
                    continue
                quota_id = str(v.get("quotaId", "")).lower()
                desc = str(v.get("description", "")).lower()
                if "perday" in quota_id or "daily" in quota_id or "per day" in desc:
                    res["is_daily_quota"] = True

    return res


def parse_rate_limit_headers(
    headers: dict[str, Any] | None,
    error_body: dict | str | None = None,
    provider: str = "",
    now_utc: datetime | None = None,
) -> RateLimitSnapshot:
    """Normalize headers and error bodies across all supported provider conventions."""
    headers_clean: dict[str, str] = {}
    if headers:
        for k, v in headers.items():
            headers_clean[str(k).lower().strip()] = str(v).strip()

    # 1. Standard Retry-After
    retry_after_s: float | None = None
    if "retry-after" in headers_clean:
        retry_after_s = parse_retry_after(headers_clean["retry-after"], now_utc=now_utc)
    elif "retry-after-ms" in headers_clean:
        try:
            retry_after_s = max(0.0, float(headers_clean["retry-after-ms"]) / 1000.0)
        except ValueError:
            pass

    # 2. Check IETF HTTPAPI Standard headers
    ietf_data: dict[str, Any] = {}
    if "ratelimit" in headers_clean:
        ietf_data = parse_ietf_ratelimit_header(
            headers_clean["ratelimit"], headers_clean.get("ratelimit-policy")
        )

    # 3. Anthropic headers (anthropic-ratelimit-*)
    ant_req_limit = headers_clean.get("anthropic-ratelimit-requests-limit")
    ant_req_rem = headers_clean.get("anthropic-ratelimit-requests-remaining")
    ant_req_reset = headers_clean.get("anthropic-ratelimit-requests-reset")

    ant_in_limit = headers_clean.get("anthropic-ratelimit-input-tokens-limit")
    ant_in_rem = headers_clean.get("anthropic-ratelimit-input-tokens-remaining")
    ant_in_reset = headers_clean.get("anthropic-ratelimit-input-tokens-reset")

    ant_out_limit = headers_clean.get("anthropic-ratelimit-output-tokens-limit")
    ant_out_rem = headers_clean.get("anthropic-ratelimit-output-tokens-remaining")
    ant_out_reset = headers_clean.get("anthropic-ratelimit-output-tokens-reset")

    # Unified tokens fallback for Anthropic
    if not ant_in_limit and "anthropic-ratelimit-tokens-limit" in headers_clean:
        ant_in_limit = headers_clean.get("anthropic-ratelimit-tokens-limit")
        ant_in_rem = headers_clean.get("anthropic-ratelimit-tokens-remaining")
        ant_in_reset = headers_clean.get("anthropic-ratelimit-tokens-reset")

    # 4. OpenAI headers (x-ratelimit-*)
    oa_req_limit = headers_clean.get("x-ratelimit-limit-requests")
    oa_req_rem = headers_clean.get("x-ratelimit-remaining-requests")
    oa_req_reset = headers_clean.get("x-ratelimit-reset-requests")

    oa_tok_limit = headers_clean.get("x-ratelimit-limit-tokens")
    oa_tok_rem = headers_clean.get("x-ratelimit-remaining-tokens")
    oa_tok_reset = headers_clean.get("x-ratelimit-reset-tokens")

    # 5. Gemini / Vertex details
    gemini_data = parse_gemini_error_details(error_body)
    is_daily_quota = gemini_data.get("is_daily_quota", False)
    if retry_after_s is None and gemini_data.get("retry_after_s") is not None:
        retry_after_s = gemini_data["retry_after_s"]

    def _first(*vals: Any) -> Any:
        for v in vals:
            if v is not None:
                return v
        return None

    def _to_int(s: str | None) -> int | None:
        if s is None:
            return None
        s_clean = s.strip()
        return int(s_clean) if s_clean.isdigit() else None

    # Resolution precedence: IETF -> Anthropic -> OpenAI (preserving 0 values!)
    rpm_limit = _first(
        ietf_data.get("rpm_limit"),
        _to_int(ant_req_limit),
        _to_int(oa_req_limit),
    )
    rpm_remaining = _first(
        ietf_data.get("rpm_remaining"),
        _to_int(ant_req_rem),
        _to_int(oa_req_rem),
    )
    rpm_reset_s = _first(
        ietf_data.get("rpm_reset_s"),
        parse_rfc3339_reset(ant_req_reset, now_utc=now_utc),
        parse_duration_string(oa_req_reset),
    )

    itpm_limit = _first(
        ietf_data.get("itpm_limit"),
        _to_int(ant_in_limit),
        _to_int(oa_tok_limit),
    )
    itpm_remaining = _first(
        ietf_data.get("itpm_remaining"),
        _to_int(ant_in_rem),
        _to_int(oa_tok_rem),
    )
    itpm_reset_s = _first(
        ietf_data.get("itpm_reset_s"),
        parse_rfc3339_reset(ant_in_reset, now_utc=now_utc),
        parse_duration_string(oa_tok_reset),
    )

    otpm_limit = _to_int(ant_out_limit)
    otpm_remaining = _to_int(ant_out_rem)
    otpm_reset_s = parse_rfc3339_reset(ant_out_reset, now_utc=now_utc)

    # 6. Local Inference / Slot Telemetry
    slot_util = None
    kv_util = None
    reqs_waiting = None
    if isinstance(error_body, dict):
        if "slot_utilization" in error_body:
            slot_util = float(error_body["slot_utilization"])
        if "kv_cache_utilization" in error_body:
            kv_util = float(error_body["kv_cache_utilization"])
        if "requests_waiting" in error_body:
            reqs_waiting = int(error_body["requests_waiting"])

    return RateLimitSnapshot(
        rpm_limit=rpm_limit,
        rpm_remaining=rpm_remaining,
        rpm_reset_s=rpm_reset_s,
        itpm_limit=itpm_limit,
        itpm_remaining=itpm_remaining,
        itpm_reset_s=itpm_reset_s,
        otpm_limit=otpm_limit,
        otpm_remaining=otpm_remaining,
        otpm_reset_s=otpm_reset_s,
        retry_after_s=retry_after_s,
        is_daily_quota=is_daily_quota,
        slot_utilization=slot_util,
        kv_cache_utilization=kv_util,
        requests_waiting=reqs_waiting,
    )

