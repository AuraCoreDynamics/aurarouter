"""
Event integration for AuraRouter on AuraGrid.

Enables async communication with aurarouter via AuraGrid's event substrate.
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EventBridge:
    """
    Bridge between AuraGrid's event system and AuraRouter services.

    Enables both synchronous RPC calls and asynchronous event-driven
    communication patterns.
    """

    # Topic names for aurarouter events
    ROUTING_REQUESTS_TOPIC = "aurarouter.routing_requests"
    ROUTING_RESULTS_TOPIC_PREFIX = "aurarouter.routing_results"

    def __init__(self, event_publisher=None, event_consumer=None):
        """
        Initialize event bridge.

        Args:
            event_publisher: AuraGrid IEventPublisher instance
            event_consumer: AuraGrid IEventConsumer instance
        """
        self.event_publisher = event_publisher
        self.event_consumer = event_consumer
        self.processed_events = set()  # Track processed events for idempotency

    async def subscribe_to_routing_requests(self):
        """
        Subscribe to incoming routing requests.

        Yields events with schema:
        {
            "request_id": "uuid",
            "task": "task description",
            "language": "python",
            "return_topic": "topic to publish results to",
            "timestamp": ISO8601
        }
        """
        if not self.event_consumer:
            logger.warning("Event consumer not configured; skipping event subscription")
            return

        logger.info(f"Subscribing to routing requests on {self.ROUTING_REQUESTS_TOPIC}")

        try:
            async for event in self.event_consumer.consume(
                self.ROUTING_REQUESTS_TOPIC
            ):
                try:
                    payload = json.loads(event.payload)
                    request_id = payload.get("request_id")

                    # Skip if already processed
                    if request_id and request_id in self.processed_events:
                        logger.debug(f"Skipping duplicate request {request_id}")
                        continue

                    logger.debug(f"Received routing request: {request_id}")
                    yield payload

                    if request_id:
                        self.processed_events.add(request_id)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to deserialize event: {e}")
                    continue
        except Exception as e:
            logger.error(f"Event subscription failed: {e}")

    async def publish_routing_result(
        self, request_id: str, result: Any, return_topic: str
    ) -> None:
        """
        Publish routing result to a return topic.

        Args:
            request_id: Unique request ID for correlation
            result: The routing result (code, plan, decision, etc.)
            return_topic: Topic to publish to
        """
        if not self.event_publisher:
            logger.warning(
                "Event publisher not configured; cannot publish result"
            )
            return

        try:
            payload = {
                "request_id": request_id,
                "result": result,
                "timestamp": self._get_timestamp(),
            }

            await self.event_publisher.publish(
                topic_id=return_topic,
                payload=json.dumps(payload).encode(),
                event_type="aurarouter.result",
            )

            logger.debug(f"Published result for request {request_id} to {return_topic}")

        except Exception as e:
            logger.error(f"Failed to publish routing result: {e}")
            raise

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
