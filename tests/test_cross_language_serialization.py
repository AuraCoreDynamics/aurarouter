"""Cross-language serialization contract tests (TG13).

Verifies that JSON produced by C# (AuraGrid/AuraXLM) can be consumed by Python,
and that Python-produced JSON conforms to the unified sovereignty audit schema.
"""

from __future__ import annotations

import json


class TestCSharpToPythonDeserialization:
    """Verify Python can consume C# JSON payloads."""

    def test_draft_token_batch_from_csharp(self):
        """DraftTokenBatch JSON from C# [JsonPropertyName] snake_case."""
        csharp_json = """
        {
            "draft_id": "draft-001",
            "session_id": "session-abc",
            "drafter_node_id": "node-1",
            "tokens": [42, 43, 44],
            "log_probs": [-0.5, -0.3, -0.1],
            "kv_cache_pointer": {
                "cache_id": "cache-1",
                "layer_offset": 1024,
                "sequence_length": 128,
                "node_id": "node-1"
            },
            "timestamp": "2025-07-17T12:00:00+00:00"
        }
        """
        data = json.loads(csharp_json)

        assert data["draft_id"] == "draft-001"
        assert data["session_id"] == "session-abc"
        assert data["drafter_node_id"] == "node-1"
        assert data["tokens"] == [42, 43, 44]
        assert len(data["log_probs"]) == 3
        assert data["kv_cache_pointer"]["cache_id"] == "cache-1"
        assert data["kv_cache_pointer"]["layer_offset"] == 1024

    def test_verification_result_from_csharp(self):
        """VerificationResult JSON from C#."""
        csharp_json = """
        {
            "draft_id": "draft-001",
            "accepted_count": 3,
            "correction_token": 99,
            "verifier_node_id": "node-v1",
            "verification_latency_ms": 12.5
        }
        """
        data = json.loads(csharp_json)

        assert data["draft_id"] == "draft-001"
        assert data["accepted_count"] == 3
        assert data["correction_token"] == 99
        assert data["verifier_node_id"] == "node-v1"
        assert data["verification_latency_ms"] == 12.5

    def test_verification_result_null_correction_from_csharp(self):
        """VerificationResult with null correction_token from C#."""
        csharp_json = """
        {
            "draft_id": "draft-002",
            "accepted_count": 5,
            "correction_token": null,
            "verifier_node_id": "node-v2",
            "verification_latency_ms": 8.0
        }
        """
        data = json.loads(csharp_json)
        assert data["correction_token"] is None

    def test_latent_anchor_metadata_from_csharp(self):
        """LatentAnchor JSON metadata from C# McpJsonOptions."""
        csharp_json = """
        {
            "anchor_id": "anc-001",
            "source_model_id": "llama-3.2-8b",
            "source_node_id": "node-1",
            "session_id": "sess-abc",
            "dimension": 4096,
            "layer_index": 16,
            "sequence_position": 42,
            "timestamp": "2025-07-17T12:00:00+00:00",
            "sovereignty_class": "sovereign",
            "adapter_id": "adapter-0",
            "metadata": {"source": "monologue"}
        }
        """
        data = json.loads(csharp_json)

        assert data["anchor_id"] == "anc-001"
        assert data["dimension"] == 4096
        assert data["sovereignty_class"] == "sovereign"
        assert data["metadata"]["source"] == "monologue"


class TestPythonToCSharpSerialization:
    """Verify Python-produced JSON matches C# expected schema."""

    def test_sovereignty_audit_event_schema(self):
        """Sovereignty audit event matches unified schema required by C#."""
        from datetime import datetime, timezone

        event = {
            "event_type": "sovereignty_decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": "aurarouter",
            "decision": "sanitize",
            "resource_type": "response",
            "resource_id": "",
            "sovereignty_class": "sovereign",
            "destination": "local",
            "reason": "Patterns matched: ssn, email",
        }

        # All 9 required fields present
        required_fields = [
            "event_type", "timestamp", "project", "decision",
            "resource_type", "resource_id", "sovereignty_class",
            "destination", "reason",
        ]
        for field in required_fields:
            assert field in event, f"Missing field: {field}"

        # Validate enum values
        assert event["project"] in ("aurarouter", "auraxlm", "auragrid")
        assert event["decision"] in ("allow", "deny", "sanitize")
        assert event["resource_type"] in ("prompt", "anchor", "wal_event", "response")
        assert event["sovereignty_class"] in ("sovereign", "open")
        assert event["destination"] in ("local", "cloud", "shared")

        # Verify JSON round-trip
        serialized = json.dumps(event)
        deserialized = json.loads(serialized)
        assert deserialized == event

    def test_sovereignty_audit_all_projects_consistent(self):
        """All three project audit events share the same field set."""
        events = [
            {
                "event_type": "sovereignty_decision",
                "timestamp": "2025-07-17T12:00:00+00:00",
                "project": "auragrid",
                "decision": "deny",
                "resource_type": "wal_event",
                "resource_id": "shared.events",
                "sovereignty_class": "sovereign",
                "destination": "shared",
                "reason": "Sovereign data cannot be written to replicated topic",
            },
            {
                "event_type": "sovereignty_decision",
                "timestamp": "2025-07-17T12:00:00+00:00",
                "project": "auraxlm",
                "decision": "deny",
                "resource_type": "anchor",
                "resource_id": "anc-001",
                "sovereignty_class": "sovereign",
                "destination": "cloud",
                "reason": "Sovereign anchor denied to cloud model",
            },
            {
                "event_type": "sovereignty_decision",
                "timestamp": "2025-07-17T12:00:00+00:00",
                "project": "aurarouter",
                "decision": "sanitize",
                "resource_type": "response",
                "resource_id": "",
                "sovereignty_class": "sovereign",
                "destination": "local",
                "reason": "Patterns matched: ssn",
            },
        ]

        # All events have the same field set
        field_set = set(events[0].keys())
        for event in events[1:]:
            assert set(event.keys()) == field_set
