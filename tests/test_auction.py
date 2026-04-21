"""Tests for aurarouter.auragrid.auction — inference auction bid calculation."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timezone


class TestAuctionListenerBidCalculation:
    """Test bid calculation logic (does not require event system)."""

    def _make_listener(self, model_state=None, circuit_available=True,
                       total_vram_mb=None, vram_usage=0.0):
        from aurarouter.auragrid.auction import AuctionListener
        from aurarouter.auragrid.contracts import ModelState, ModelTelemetry

        registry = Mock()
        registry.get_model_state.return_value = model_state or ModelState.UNKNOWN
        if model_state and model_state != ModelState.UNKNOWN:
            telemetry = ModelTelemetry(
                model_id="test-model",
                provider_name="TestProvider",
                state=model_state,
                vram_usage_mb=vram_usage,
            )
            registry.get_all_telemetry.return_value = {"test-model": telemetry}
        else:
            registry.get_all_telemetry.return_value = {}

        breaker_reg = Mock()
        breaker = Mock()
        breaker.is_available.return_value = circuit_available
        breaker_reg.get_or_create.return_value = breaker

        event_bridge = Mock()
        event_bridge.is_active = True

        return AuctionListener(
            event_bridge=event_bridge,
            model_registry=registry,
            circuit_breaker_registry=breaker_reg,
            node_id="test-node",
            total_vram_mb=total_vram_mb,
        )

    def _make_request(self, model_id="test-model", is_transient=False,
                      vram_requirement_mb=None):
        from aurarouter.auragrid.contracts import InferenceRequest
        return InferenceRequest(
            request_id="req-1",
            model_id=model_id,
            prompt_token_estimate=100,
            max_tokens=200,
            originator_node_id="other-node",
            is_transient=is_transient,
            vram_requirement_mb=vram_requirement_mb,
        )

    def test_warm_model_high_score(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(model_state=ModelState.WARM)
        request = self._make_request()
        bid = listener.calculate_bid(request)
        assert bid is not None
        assert bid.score >= 0.8
        assert bid.is_warm is True

    def test_cold_model_medium_score(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(model_state=ModelState.COLD)
        request = self._make_request()
        bid = listener.calculate_bid(request)
        assert bid is not None
        assert 0.3 <= bid.score <= 0.6

    def test_unknown_model_no_bid(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(model_state=ModelState.UNKNOWN)
        request = self._make_request()
        bid = listener.calculate_bid(request)
        assert bid is None

    def test_circuit_breaker_open_no_bid(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(
            model_state=ModelState.WARM, circuit_available=False
        )
        request = self._make_request()
        bid = listener.calculate_bid(request)
        assert bid is None

    def test_transient_cold_penalty(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(model_state=ModelState.COLD)
        normal = self._make_request(is_transient=False)
        transient = self._make_request(is_transient=True)
        normal_bid = listener.calculate_bid(normal)
        transient_bid = listener.calculate_bid(transient)
        assert transient_bid is not None
        assert normal_bid is not None
        assert transient_bid.score < normal_bid.score

    def test_vram_pressure_suppresses_all_bids(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(
            model_state=ModelState.WARM,
            total_vram_mb=10000.0,
            vram_usage=9500.0,  # 95% > default 90% threshold
        )
        request = self._make_request()
        bid = listener.calculate_bid(request)
        assert bid is None

    def test_insufficient_vram_no_bid(self):
        from aurarouter.auragrid.contracts import ModelState
        listener = self._make_listener(
            model_state=ModelState.WARM,
            total_vram_mb=10000.0,
            vram_usage=8000.0,
        )
        request = self._make_request(vram_requirement_mb=5000.0)
        bid = listener.calculate_bid(request)
        assert bid is None

    def test_capacity_advisor_adjusts_score(self):
        from aurarouter.auragrid.auction import AuctionListener
        from aurarouter.auragrid.contracts import ModelState, ModelTelemetry

        registry = Mock()
        registry.get_model_state.return_value = ModelState.WARM
        registry.get_all_telemetry.return_value = {
            "test-model": ModelTelemetry(
                model_id="test-model", provider_name="p", state=ModelState.WARM
            )
        }

        advisor = Mock()
        advisor.adjust_bid_score.return_value = 0.95

        listener = AuctionListener(
            event_bridge=Mock(is_active=True),
            model_registry=registry,
            circuit_breaker_registry=Mock(
                get_or_create=Mock(return_value=Mock(is_available=Mock(return_value=True)))
            ),
            node_id="n1",
            capacity_advisor=advisor,
        )
        request = self._make_request()
        bid = listener.calculate_bid(request)
        assert bid is not None
        assert bid.score == pytest.approx(0.95)
        advisor.adjust_bid_score.assert_called_once()


class TestAuctionListenerLifecycle:
    @pytest.mark.asyncio
    async def test_start_without_event_bridge(self):
        from aurarouter.auragrid.auction import AuctionListener
        listener = AuctionListener(event_bridge=None, node_id="n1")
        await listener.start()
        assert listener._running is False

    @pytest.mark.asyncio
    async def test_start_without_active_bridge(self):
        from aurarouter.auragrid.auction import AuctionListener
        bridge = Mock()
        bridge.is_active = False
        listener = AuctionListener(event_bridge=bridge, node_id="n1")
        await listener.start()
        assert listener._running is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        from aurarouter.auragrid.auction import AuctionListener
        listener = AuctionListener(event_bridge=None, node_id="n1")
        await listener.stop()  # Should not raise
