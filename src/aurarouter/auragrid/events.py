"""
Event integration for AuraRouter on AuraGrid.

Wraps the AuraGrid SDK's AsyncEventClient for type-safe event pub/sub.
Falls back to no-op when the SDK is not installed (standalone mode).
"""

import collections
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

# Guard SDK imports for standalone mode
_HAS_SDK = False
try:
    from auragrid import AsyncEventClient, AuraEventDto
    _HAS_SDK = True
except ImportError:
    AsyncEventClient = None  # type: ignore[assignment,misc]
    AuraEventDto = None  # type: ignore[assignment,misc]


class EventBridge:
    """
    Bridge between AuraGrid's event system and AuraRouter services.

    Wraps the SDK's AsyncEventClient for proper base64 framing,
    marker management, and streaming consumption. In standalone mode
    (no SDK), all operations are no-ops.
    """

    # Topic names for aurarouter events
    ROUTING_REQUESTS_TOPIC = "aurarouter.routing_requests"
    ROUTING_RESULTS_TOPIC_PREFIX = "aurarouter.routing_results"

    def __init__(self, event_client: Optional[Any] = None):
        """
        Initialize event bridge.

        Args:
            event_client: An auragrid.AsyncEventClient instance (from GridContext.events).
        """
        self.event_client: Optional[Any] = event_client
        self.processed_events = collections.deque(maxlen=10000)

    @property
    def is_active(self) -> bool:
        """True if an event client is configured and the SDK is available."""
        return self.event_client is not None

    async def subscribe_to_routing_requests(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to incoming routing requests via SDK streaming.

        Yields event payloads as dicts. Handles dedup via processed_events ring.
        """
        if not self.is_active:
            logger.warning("Event client not configured; skipping event subscription")
            return

        logger.info("Subscribing to routing requests on %s", self.ROUTING_REQUESTS_TOPIC)

        try:
            async for event in self.event_client.subscribe(
                self.ROUTING_REQUESTS_TOPIC,
            ):
                try:
                    payload = json.loads(event.payload) if isinstance(event.payload, (str, bytes)) else event.payload
                    request_id = payload.get("request_id")

                    # Skip if already processed
                    if request_id and request_id in self.processed_events:
                        logger.debug("Skipping duplicate request %s", request_id)
                        continue

                    logger.debug("Received routing request: %s", request_id)
                    yield payload

                    if request_id:
                        self.processed_events.append(request_id)

                except (json.JSONDecodeError, TypeError) as e:
                    logger.error("Failed to deserialize event: %s", e)
                    continue
        except Exception as e:
            logger.error("Event subscription failed: %s", e)

    async def publish(
        self, topic_id: str, payload: dict, event_type: Optional[str] = None
    ) -> None:
        """
        Publish an event to a topic via the SDK EventClient.

        Handles JSON serialization and proper base64 framing through the SDK.

        Args:
            topic_id: Target topic identifier
            payload: Dict to serialize as JSON payload
            event_type: Optional event type tag
        """
        if not self.is_active:
            logger.warning("Event client not configured; cannot publish to %s", topic_id)
            return

        try:
            payload_bytes = json.dumps(payload).encode("utf-8")
            await self.event_client.publish(
                topic_id=topic_id,
                payload=payload_bytes,
                event_type=event_type,
            )
            logger.debug("Published event to %s", topic_id)
        except Exception as e:
            logger.error("Failed to publish event to %s: %s", topic_id, e)
            raise

    async def publish_routing_result(
        self, request_id: str, result: Any, return_topic: str
    ) -> None:
        """
        Publish routing result to a return topic.

        Args:
            request_id: Unique request ID for correlation
            result: The routing result
            return_topic: Topic to publish to
        """
        payload = {
            "request_id": request_id,
            "result": result,
            "timestamp": self._get_timestamp(),
        }
        await self.publish(return_topic, payload, event_type="aurarouter.result")
        logger.debug("Published result for request %s to %s", request_id, return_topic)

    def create_routing_request(
        self,
        task: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a routing request for publishing.

        Args:
            task: Task description
            language: Target language
            context: Optional context data

        Returns:
            Request payload ready for publishing
        """
        request_id = str(uuid.uuid4())
        return_topic = f"{self.ROUTING_RESULTS_TOPIC_PREFIX}.{request_id}"

        return {
            "request_id": request_id,
            "task": task,
            "language": language,
            "context": context or {},
            "return_topic": return_topic,
            "timestamp": self._get_timestamp(),
        }

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO8601 format."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
