"""Unit tests for multi-protocol rate limit header and error parsers.

AuraRouter v2.0 §3.2
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from aurarouter.ratelimit.parsers import (
    parse_duration_string,
    parse_rfc3339_reset,
    parse_retry_after,
    parse_ietf_ratelimit_header,
    format_ietf_ratelimit_headers,
    parse_gemini_error_details,
    parse_rate_limit_headers,
)


class TestDurationParsing:
    def test_numeric_seconds(self):
        assert parse_duration_string("15") == 15.0
        assert parse_duration_string("0.5") == 0.5
        assert parse_duration_string(0) == 0.0

    def test_go_compound_durations(self):
        assert parse_duration_string("2m59.5s") == 179.5
        assert parse_duration_string("1h2m3s") == 3723.0
        assert parse_duration_string("250ms") == 0.25
        assert parse_duration_string("6ms") == 0.006
        assert parse_duration_string("500us") == 0.0005
        assert parse_duration_string("10s") == 10.0

    def test_invalid_and_empty(self):
        assert parse_duration_string(None) is None
        assert parse_duration_string("") is None
        assert parse_duration_string("invalid") is None


class TestTimestampParsing:
    def test_rfc3339_timestamp(self):
        now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
        target = "2026-11-03T12:00:30Z"
        delta = parse_rfc3339_reset(target, now_utc=now)
        assert delta == pytest.approx(30.0, rel=1e-3)

    def test_rfc3339_fractional_seconds(self):
        now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
        target = "2026-11-03T12:01:05.500+00:00"
        delta = parse_rfc3339_reset(target, now_utc=now)
        assert delta == pytest.approx(65.5, rel=1e-3)

    def test_past_timestamp_returns_zero(self):
        now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
        target = "2026-11-03T11:59:00Z"
        delta = parse_rfc3339_reset(target, now_utc=now)
        assert delta == 0.0


class TestRetryAfterParsing:
    def test_integer_seconds(self):
        assert parse_retry_after("45") == 45.0

    def test_http_date(self):
        now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
        date_str = "Tue, 03 Nov 2026 12:02:00 GMT"
        delta = parse_retry_after(date_str, now_utc=now)
        assert delta == pytest.approx(120.0, rel=1e-2)


class TestIETFStructuredFields:
    def test_parse_ietf_headers(self):
        policy = '"rpm";q=500;w=60, "tpm";q=200000;w=60'
        limits = '"rpm";r=42;t=18, "tpm";r=15400;t=18'
        res = parse_ietf_ratelimit_header(limits, policy)

        assert res["rpm_limit"] == 500
        assert res["itpm_limit"] == 200000
        assert res["rpm_remaining"] == 42
        assert res["rpm_reset_s"] == 18.0
        assert res["itpm_remaining"] == 15400
        assert res["itpm_reset_s"] == 18.0

    def test_format_ietf_headers(self):
        headers = format_ietf_ratelimit_headers(
            rpm_limit=500,
            rpm_remaining=120,
            rpm_reset_s=14.2,
            itpm_limit=100000,
            itpm_remaining=45000,
            itpm_reset_s=22.8,
        )
        assert 'RateLimit-Policy' in headers
        assert 'RateLimit' in headers
        assert '"rpm";q=500;w=60' in headers["RateLimit-Policy"]
        assert '"tpm";q=100000;w=60' in headers["RateLimit-Policy"]
        assert 'r=120;t=15' in headers["RateLimit"]
        assert 'r=45000;t=23' in headers["RateLimit"]


class TestGeminiErrorDetails:
    def test_gemini_retry_info_and_daily_quota(self):
        payload = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "42.5s",
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "subject": "project:12345",
                                "description": "Quota exceeded for quota metric 'GenerateContent requests per day'",
                                "quotaId": "GenerateContentRequestsPerDay",
                            }
                        ],
                    },
                ],
            }
        }
        res = parse_gemini_error_details(payload)
        assert res["retry_after_s"] == 42.5
        assert res["is_daily_quota"] is True


class TestProviderHeaderNormalization:
    def test_anthropic_split_tokens_normalization(self):
        now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
        headers = {
            "anthropic-ratelimit-requests-limit": "60",
            "anthropic-ratelimit-requests-remaining": "45",
            "anthropic-ratelimit-requests-reset": "2026-11-03T12:00:15Z",
            "anthropic-ratelimit-input-tokens-limit": "100000",
            "anthropic-ratelimit-input-tokens-remaining": "85000",
            "anthropic-ratelimit-input-tokens-reset": "2026-11-03T12:00:20Z",
            "anthropic-ratelimit-output-tokens-limit": "20000",
            "anthropic-ratelimit-output-tokens-remaining": "18000",
            "anthropic-ratelimit-output-tokens-reset": "2026-11-03T12:00:20Z",
        }
        snapshot = parse_rate_limit_headers(headers, provider="anthropic", now_utc=now)
        assert snapshot.rpm_limit == 60
        assert snapshot.rpm_remaining == 45
        assert snapshot.rpm_reset_s == pytest.approx(15.0, rel=1e-3)
        assert snapshot.itpm_limit == 100000
        assert snapshot.itpm_remaining == 85000
        assert snapshot.itpm_reset_s == pytest.approx(20.0, rel=1e-3)
        assert snapshot.otpm_limit == 20000
        assert snapshot.otpm_remaining == 18000
        assert snapshot.otpm_reset_s == pytest.approx(20.0, rel=1e-3)

    def test_openai_duration_strings_normalization(self):
        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "480",
            "x-ratelimit-reset-requests": "2.4s",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-tokens": "150000",
            "x-ratelimit-reset-tokens": "6m0s",
        }
        snapshot = parse_rate_limit_headers(headers, provider="openai")
        assert snapshot.rpm_limit == 500
        assert snapshot.rpm_remaining == 480
        assert snapshot.rpm_reset_s == 2.4
        assert snapshot.itpm_limit == 200000
        assert snapshot.itpm_remaining == 150000
        assert snapshot.itpm_reset_s == 360.0

    def test_zero_remaining_preserved_and_roundtrips_dict(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "12s",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "0s",
        }
        snapshot = parse_rate_limit_headers(headers, provider="openai")
        assert snapshot.rpm_remaining == 0
        assert snapshot.itpm_remaining == 0
        assert snapshot.itpm_reset_s == 0.0

        from aurarouter.ratelimit.models import RateLimitSnapshot
        rebuilt = RateLimitSnapshot.from_dict(snapshot.to_dict())
        assert rebuilt == snapshot

