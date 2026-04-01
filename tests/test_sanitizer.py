"""Tests for response sanitizer (TG12)."""

from __future__ import annotations

import pytest

from aurarouter.sanitizer import ResponseSanitizer, SanitizationResult


class TestResponseSanitizer:
    def test_clean_text_unchanged(self):
        sanitizer = ResponseSanitizer()
        result = sanitizer.sanitize("Hello, this is a clean response.")

        assert result.text == "Hello, this is a clean response."
        assert result.was_sanitized is False
        assert result.patterns_matched == []
        assert "X-Sovereignty-Sanitized" not in result.headers

    def test_strips_ssn(self):
        sanitizer = ResponseSanitizer()
        result = sanitizer.sanitize("The SSN is 123-45-6789.")

        assert "123-45-6789" not in result.text
        assert "[REDACTED:ssn]" in result.text
        assert result.was_sanitized is True
        assert "ssn" in result.patterns_matched
        assert result.headers.get("X-Sovereignty-Sanitized") == "true"

    def test_strips_email(self):
        sanitizer = ResponseSanitizer()
        result = sanitizer.sanitize("Contact user@example.com for details.")

        assert "user@example.com" not in result.text
        assert "[REDACTED:email]" in result.text
        assert "email" in result.patterns_matched

    def test_strips_credit_card(self):
        sanitizer = ResponseSanitizer()
        result = sanitizer.sanitize("Card: 4111-1111-1111-1111")

        assert "4111-1111-1111-1111" not in result.text
        assert result.was_sanitized is True

    def test_multiple_patterns_all_redacted(self):
        sanitizer = ResponseSanitizer()
        result = sanitizer.sanitize(
            "SSN: 123-45-6789, email: test@example.com"
        )

        assert "123-45-6789" not in result.text
        assert "test@example.com" not in result.text
        assert len(result.patterns_matched) >= 2

    def test_empty_text_unchanged(self):
        sanitizer = ResponseSanitizer()
        result = sanitizer.sanitize("")

        assert result.text == ""
        assert result.was_sanitized is False

    def test_header_only_when_sanitized(self):
        sanitizer = ResponseSanitizer()

        clean = sanitizer.sanitize("No PII here")
        assert "X-Sovereignty-Sanitized" not in clean.headers

        dirty = sanitizer.sanitize("SSN 123-45-6789")
        assert dirty.headers["X-Sovereignty-Sanitized"] == "true"

    def test_custom_patterns_from_config(self):
        class FakeConfig:
            config = {
                "system": {
                    "sovereignty_patterns": [
                        {"name": "project_code", "pattern": r"PROJECT-\d{4}"},
                    ]
                }
            }

        sanitizer = ResponseSanitizer(config=FakeConfig())
        result = sanitizer.sanitize("Reference: PROJECT-1234")

        assert "PROJECT-1234" not in result.text
        assert "[REDACTED:project_code]" in result.text

    def test_sanitization_result_dataclass(self):
        result = SanitizationResult(text="test", was_sanitized=True)
        assert result.text == "test"
        assert result.was_sanitized is True
        assert result.patterns_matched == []
        assert result.headers == {}
