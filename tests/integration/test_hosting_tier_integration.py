"""Integration tests for the hosting tier system (TG5)."""

from aurarouter.savings.privacy import PrivacyAuditor


def test_privacy_audit_respects_hosting_tier():
    """A Google model with hosting_tier=on-prem should NOT trigger privacy audit."""
    auditor = PrivacyAuditor()
    # Cloud provider but on-prem tier -> no audit
    event = auditor.audit(
        "SSN: 123-45-6789",
        "my-private-gemini",
        "google",
        hosting_tier="on-prem",
    )
    assert event is None


def test_privacy_audit_cloud_tier_triggers():
    """An ollama model with hosting_tier=cloud SHOULD trigger privacy audit."""
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "SSN: 123-45-6789",
        "remote-ollama",
        "ollama",
        hosting_tier="cloud",
    )
    assert event is not None


def test_privacy_audit_backward_compat_no_tier():
    """Without hosting_tier, behavior matches original provider-based logic."""
    auditor = PrivacyAuditor()
    # Google without explicit tier -> cloud -> audit triggers
    event = auditor.audit(
        "SSN: 123-45-6789",
        "gemini-flash",
        "google",
    )
    assert event is not None

    # Ollama without explicit tier -> on-prem -> no audit
    event = auditor.audit(
        "SSN: 123-45-6789",
        "local-llama",
        "ollama",
    )
    assert event is None


def test_privacy_audit_dedicated_tenant_not_audited():
    """dedicated-tenant tier should NOT trigger privacy audit."""
    auditor = PrivacyAuditor()
    event = auditor.audit(
        "SSN: 123-45-6789",
        "private-cloud-model",
        "google",
        hosting_tier="dedicated-tenant",
    )
    assert event is None
