"""Tests for aurarouter.auragrid.events — EventBridge wrapping SDK AsyncEventClient."""

import json
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from aurarouter.auragrid.events import EventBridge


class TestEventBridgeInit:
    def test_active_with_client(self):
        bridge = EventBridge(event_client=Mock())
        assert bridge.is_active is True

    def test_inactive_without_client(self):
        bridge = EventBridge()
        assert bridge.is_active is False
        assert bridge.event_client is None

    def test_processed_events_deque(self):
        bridge = EventBridge()
        assert len(bridge.processed_events) == 0
        assert bridge.processed_events.maxlen == 10000


class TestEventBridgePublish:
    @pytest.mark.asyncio
    async def test_publish_serializes_json(self):
        client = AsyncMock()
        bridge = EventBridge(event_client=client)

        await bridge.publish("test.topic", {"key": "value"}, event_type="test")

        client.publish.assert_awaited_once()
        call_kwargs = client.publish.call_args
        assert call_kwargs.kwargs["topic_id"] == "test.topic"
        assert call_kwargs.kwargs["event_type"] == "test"
        payload = json.loads(call_kwargs.kwargs["payload"])
        assert payload["key"] == "value"

    @pytest.mark.asyncio
    async def test_publish_inactive_noop(self):
        bridge = EventBridge()
        # Should not raise
        await bridge.publish("test.topic", {"key": "value"})

    @pytest.mark.asyncio
    async def test_publish_error_propagates(self):
        client = AsyncMock()
        client.publish.side_effect = ConnectionError("lost")
        bridge = EventBridge(event_client=client)

        with pytest.raises(ConnectionError):
            await bridge.publish("test.topic", {"k": "v"})


class TestEventBridgePublishRoutingResult:
    @pytest.mark.asyncio
    async def test_result_payload_structure(self):
        client = AsyncMock()
        bridge = EventBridge(event_client=client)

        await bridge.publish_routing_result(
            request_id="r1",
            result={"model": "llama3", "output": "hello"},
            return_topic="aurarouter.routing_results.r1",
        )

        client.publish.assert_awaited_once()
        payload_bytes = client.publish.call_args.kwargs["payload"]
        payload = json.loads(payload_bytes)
        assert payload["request_id"] == "r1"
        assert payload["result"]["model"] == "llama3"
        assert "timestamp" in payload


class TestEventBridgeCreateRequest:
    def test_creates_valid_request(self):
        bridge = EventBridge()
        req = bridge.create_routing_request(
            task="Write a function",
            language="python",
            context={"key": "value"},
        )
        assert "request_id" in req
        assert req["task"] == "Write a function"
        assert req["language"] == "python"
        assert req["context"]["key"] == "value"
        assert "return_topic" in req

    def test_return_topic_contains_request_id(self):
        bridge = EventBridge()
        req = bridge.create_routing_request(task="test")
        assert req["request_id"] in req["return_topic"]


class TestEventBridgeSubscription:
    @pytest.mark.asyncio
    async def test_subscribe_inactive_yields_nothing(self):
        bridge = EventBridge()  # no client
        events = []
        async for e in bridge.subscribe_to_routing_requests():
            events.append(e)
        assert events == []

    @pytest.mark.asyncio
    async def test_subscribe_deduplicates(self):
        # Create a mock event client with subscribe returning events
        event1 = Mock()
        event1.payload = json.dumps({"request_id": "r1", "task": "test"})
        event2 = Mock()
        event2.payload = json.dumps({"request_id": "r1", "task": "test"})  # dup
        event3 = Mock()
        event3.payload = json.dumps({"request_id": "r2", "task": "test2"})

        async def mock_subscribe(topic):
            for e in [event1, event2, event3]:
                yield e

        client = Mock()
        client.subscribe = mock_subscribe
        bridge = EventBridge(event_client=client)

        events = []
        async for e in bridge.subscribe_to_routing_requests():
            events.append(e)

        assert len(events) == 2
        assert events[0]["request_id"] == "r1"
        assert events[1]["request_id"] == "r2"

    @pytest.mark.asyncio
    async def test_subscribe_handles_malformed_json(self):
        event_good = Mock()
        event_good.payload = json.dumps({"request_id": "r1", "task": "t"})
        event_bad = Mock()
        event_bad.payload = "not json {{{}"

        async def mock_subscribe(topic):
            for e in [event_bad, event_good]:
                yield e

        client = Mock()
        client.subscribe = mock_subscribe
        bridge = EventBridge(event_client=client)

        events = []
        async for e in bridge.subscribe_to_routing_requests():
            events.append(e)

        # Should get the good event despite the bad one
        assert len(events) == 1
        assert events[0]["request_id"] == "r1"
