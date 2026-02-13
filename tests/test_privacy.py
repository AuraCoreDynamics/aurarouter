"""Tests for the privacy audit engine."""

import pytest

from aurarouter.savings.privacy import (
    PrivacyAuditor,
    PrivacyEvent,
    PrivacyMatch,
    PrivacyPattern,
    PrivacyStore,
)


# ── Auditor tests ────────────────────────────────────────────────────


def test_detect_email():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "Send this to user@example.com please", "gemini-2.0-flash", "google"
    )
    assert event is not None
    assert any(m.pattern_name == "Email Address" for m in event.matches)


def test_detect_api_key():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "Use api_key=sk-abc123def456ghi789 for auth", "gemini-2.0-flash", "google"
    )
    assert event is not None
    assert any(m.pattern_name == "API Key" for m in event.matches)


def test_detect_aws_key():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "My key is AKIAIOSFODNN7EXAMPLE", "gemini-2.0-flash", "google"
    )
    assert event is not None
    assert any(m.pattern_name == "AWS Access Key" for m in event.matches)


def test_detect_ssn():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "SSN: 123-45-6789", "claude-sonnet-4-5-20250929", "claude"
    )
    assert event is not None
    assert any(m.pattern_name == "SSN" for m in event.matches)


def test_detect_confidential():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "This document is CONFIDENTIAL", "gemini-2.0-flash", "google"
    )
    assert event is not None
    assert any(m.pattern_name == "Confidential Marker" for m in event.matches)


def test_detect_private_ip():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "Connect to 192.168.1.100 on port 8080", "gemini-2.0-flash", "google"
    )
    assert event is not None
    assert any(m.pattern_name == "Private IP Address" for m in event.matches)


def test_no_matches_returns_none():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "What is the weather today?", "gemini-2.0-flash", "google"
    )
    assert event is None


def test_local_provider_skipped():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "Send to user@example.com with SSN 123-45-6789",
        "llama3",
        "ollama",
    )
    assert event is None


def test_matched_text_redacted():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "Email me at user@example.com", "gemini-2.0-flash", "google"
    )
    assert event is not None
    email_match = next(m for m in event.matches if m.pattern_name == "Email Address")
    assert email_match.matched_text == "user***"
    assert "@example.com" not in email_match.matched_text


def test_custom_patterns():
    custom = [
        PrivacyPattern(
            name="Project Codename",
            pattern=r"(?i)\bproject\s+starlight\b",
            severity="medium",
            description="Internal project codename detected.",
        )
    ]
    auditor = PrivacyAuditor(custom_patterns=custom)
    event = auditor.audit(
        "Regarding Project Starlight and user@example.com",
        "gemini-2.0-flash",
        "google",
    )
    assert event is not None
    names = {m.pattern_name for m in event.matches}
    assert "Project Codename" in names
    assert "Email Address" in names


def test_multiple_matches():
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "Contact user@example.com with api_key=sk-abc123def456ghi789",
        "gemini-2.0-flash",
        "google",
    )
    assert event is not None
    assert len(event.matches) >= 2
    names = {m.pattern_name for m in event.matches}
    assert "Email Address" in names
    assert "API Key" in names


# ── Store tests ──────────────────────────────────────────────────────


def _make_event(**overrides) -> PrivacyEvent:
    defaults = dict(
        timestamp="2025-06-15T10:00:00Z",
        model_id="gemini-2.0-flash",
        provider="google",
        matches=[
            PrivacyMatch(
                pattern_name="Email Address",
                severity="medium",
                matched_text="user***",
                position=10,
            )
        ],
        prompt_length=50,
        recommendation="Consider routing to a local model",
    )
    defaults.update(overrides)
    return PrivacyEvent(**defaults)


def test_privacy_store_record_and_query(tmp_path):
    store = PrivacyStore(db_path=tmp_path / "usage.db")
    event = _make_event()
    store.record(event)

    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert r["model_id"] == "gemini-2.0-flash"
    assert r["provider"] == "google"
    assert r["match_count"] == 1
    assert r["severities"] == ["medium"]
    assert r["prompt_length"] == 50
    assert r["recommendation"] == "Consider routing to a local model"


def test_privacy_store_summary(tmp_path):
    store = PrivacyStore(db_path=tmp_path / "usage.db")

    # Event 1: email (medium)
    store.record(_make_event(
        timestamp="2025-06-15T10:00:00Z",
        matches=[
            PrivacyMatch("Email Address", "medium", "user***", 10),
        ],
    ))

    # Event 2: API key (high) + SSN (high)
    store.record(_make_event(
        timestamp="2025-06-15T11:00:00Z",
        matches=[
            PrivacyMatch("API Key", "high", "api_***", 5),
            PrivacyMatch("SSN", "high", "123-***", 30),
        ],
    ))

    # Event 3: private IP (low)
    store.record(_make_event(
        timestamp="2025-06-15T12:00:00Z",
        matches=[
            PrivacyMatch("Private IP Address", "low", "192.***", 0),
        ],
    ))

    summary = store.summary()
    assert summary["total_events"] == 3
    assert summary["by_severity"]["medium"] == 1
    assert summary["by_severity"]["high"] == 2
    assert summary["by_severity"]["low"] == 1
    assert summary["by_pattern"]["Email Address"] == 1
    assert summary["by_pattern"]["API Key"] == 1
    assert summary["by_pattern"]["SSN"] == 1
    assert summary["by_pattern"]["Private IP Address"] == 1
