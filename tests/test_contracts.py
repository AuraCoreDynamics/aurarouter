"""Tests for aurarouter.auragrid.contracts — Pydantic DTOs for inference auction."""

import pytest
from datetime import datetime, timezone


class TestModelState:
    def test_enum_values(self):
        from aurarouter.auragrid.contracts import ModelState
        assert ModelState.WARM == "warm"
        assert ModelState.COLD == "cold"
        assert ModelState.LOADING == "loading"
        assert ModelState.UNKNOWN == "unknown"


class TestModelTelemetry:
    def test_round_trip_json(self):
        from aurarouter.auragrid.contracts import ModelTelemetry, ModelState
        t = ModelTelemetry(
            model_id="llama3",
            provider_name="OllamaProvider",
            state=ModelState.WARM,
            vram_usage_mb=4096.0,
            last_used=datetime(2026, 1, 1, tzinfo=timezone.utc),
            context_slots_free=8,
        )
        json_str = t.model_dump_json(by_alias=True)
        assert "modelId" in json_str
        assert "providerName" in json_str
        assert "vramUsageMb" in json_str
        roundtrip = ModelTelemetry.model_validate_json(json_str)
        assert roundtrip.model_id == "llama3"
        assert roundtrip.state == ModelState.WARM

    def test_defaults(self):
        from aurarouter.auragrid.contracts import ModelTelemetry, ModelState
        t = ModelTelemetry(model_id="m1", provider_name="p1", state=ModelState.UNKNOWN)
        assert t.vram_usage_mb is None
        assert t.last_used is None
        assert t.context_slots_free is None


class TestProviderHealthState:
    def test_round_trip_json(self):
        from aurarouter.auragrid.contracts import ProviderHealthState
        h = ProviderHealthState(
            provider_name="OllamaProvider",
            is_healthy=True,
            consecutive_failures=0,
            circuit_state="closed",
        )
        json_str = h.model_dump_json(by_alias=True)
        assert "providerName" in json_str
        assert "isHealthy" in json_str
        assert "circuitState" in json_str
        roundtrip = ProviderHealthState.model_validate_json(json_str)
        assert roundtrip.provider_name == "OllamaProvider"
        assert roundtrip.is_healthy is True


class TestInferenceRequest:
    def test_round_trip_json(self):
        from aurarouter.auragrid.contracts import InferenceRequest
        r = InferenceRequest(
            request_id="req-1",
            model_id="llama3",
            prompt_token_estimate=100,
            max_tokens=200,
            originator_node_id="node-1",
        )
        json_str = r.model_dump_json(by_alias=True)
        assert "requestId" in json_str
        assert "modelId" in json_str
        assert "originatorNodeId" in json_str
        roundtrip = InferenceRequest.model_validate_json(json_str)
        assert roundtrip.request_id == "req-1"

    def test_defaults(self):
        from aurarouter.auragrid.contracts import InferenceRequest
        r = InferenceRequest(
            request_id="r1", model_id="m1",
            prompt_token_estimate=10, max_tokens=50,
            originator_node_id="n1",
        )
        assert r.is_transient is False
        assert r.timeout_ms == 30000
        assert r.vram_requirement_mb is None


class TestInferenceBid:
    def test_valid_score(self):
        from aurarouter.auragrid.contracts import InferenceBid
        bid = InferenceBid(
            request_id="r1", node_id="n1", score=0.8,
            is_warm=True, estimated_latency_ms=100,
            vram_free_mb=4000.0,
            bid_timestamp=datetime.now(timezone.utc),
        )
        assert bid.score == 0.8

    def test_invalid_score_raises(self):
        from aurarouter.auragrid.contracts import InferenceBid
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            InferenceBid(
                request_id="r1", node_id="n1", score=1.5,
                is_warm=True, estimated_latency_ms=100,
                vram_free_mb=4000.0,
                bid_timestamp=datetime.now(timezone.utc),
            )

    def test_round_trip_json(self):
        from aurarouter.auragrid.contracts import InferenceBid
        bid = InferenceBid(
            request_id="r1", node_id="n1", score=0.7,
            is_warm=False, estimated_latency_ms=5000,
            vram_free_mb=8000.0,
            bid_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        json_str = bid.model_dump_json(by_alias=True)
        assert "requestId" in json_str
        assert "nodeId" in json_str
        assert "isWarm" in json_str
        roundtrip = InferenceBid.model_validate_json(json_str)
        assert roundtrip.score == pytest.approx(0.7)


class TestICapacityAdvisor:
    def test_protocol_is_protocol(self):
        from aurarouter.auragrid.contracts import ICapacityAdvisor
        from typing import Protocol
        assert issubclass(ICapacityAdvisor, Protocol)
