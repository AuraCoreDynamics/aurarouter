"""Inference auction listener for AuraRouter grid integration.

Listens for inference requests on the Event Substrate, evaluates
local capacity, and submits volunteer bids.

IoC principle: this node *decides* to bid based on self-assessed
capability.  No external entity commands it.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aurarouter.registry import RuntimeModelRegistry
    from aurarouter.circuit_breaker import CircuitBreakerRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard all grid imports — module must be importable in standalone mode
# ---------------------------------------------------------------------------
try:
    from aurarouter.auragrid.contracts import (
        InferenceBid,
        InferenceRequest,
        ModelState,
        ModelTelemetry,
    )
except ImportError:
    InferenceBid = None  # type: ignore[assignment,misc]
    InferenceRequest = None  # type: ignore[assignment,misc]
    ModelState = None  # type: ignore[assignment,misc]
    ModelTelemetry = None  # type: ignore[assignment,misc]

try:
    from aurarouter.auragrid.contracts import ICapacityAdvisor
except ImportError:
    ICapacityAdvisor = None  # type: ignore[assignment,misc]

try:
    from aurarouter.auragrid.events import EventBridge
except ImportError:
    EventBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Topic constants — follows EventBridge naming convention
# ---------------------------------------------------------------------------
INFERENCE_REQUESTS_TOPIC = "aurarouter.inference_requests"
INFERENCE_BIDS_TOPIC_PREFIX = "aurarouter.inference_bids"


class AuctionListener:
    """Listens for inference auction requests and submits volunteer bids.

    Subscribes to ``aurarouter.inference_requests`` on EventBridge.
    For each request the node evaluates its own capacity and publishes
    a bid to ``aurarouter.inference_bids.{request_id}`` — or stays
    silent when it cannot serve the request.
    """

    def __init__(
        self,
        event_bridge: Any,
        model_registry: Optional["RuntimeModelRegistry"] = None,
        circuit_breaker_registry: Optional["CircuitBreakerRegistry"] = None,
        node_id: str = "unknown",
        vram_pressure_threshold: float = 0.9,
        total_vram_mb: Optional[float] = None,
        capacity_advisor: Any = None,
    ):
        self._event_bridge = event_bridge
        self._model_registry = model_registry
        self._circuit_breaker_registry = circuit_breaker_registry
        self._node_id = node_id
        self._vram_pressure_threshold = vram_pressure_threshold
        self._total_vram_mb = total_vram_mb
        self._capacity_advisor = capacity_advisor
        # Dedup ring — mirrors EventBridge.processed_events pattern
        self._processed_requests: collections.deque[str] = collections.deque(maxlen=10_000)
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start listening for inference requests."""
        if self._running:
            return
        if self._event_bridge is None:
            logger.warning("EventBridge not available — auction listener disabled")
            return
        if not self._event_bridge.is_active:
            logger.warning("Event client not configured — auction listener disabled")
            return
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("Auction listener started on topic %s", INFERENCE_REQUESTS_TOPIC)

    async def stop(self) -> None:
        """Stop listening for inference requests."""
        self._running = False
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        logger.info("Auction listener stopped")

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Consume inference requests via EventBridge and evaluate bids."""
        try:
            async for event in self._event_bridge.event_client.subscribe(
                INFERENCE_REQUESTS_TOPIC
            ):
                if not self._running:
                    break
                try:
                    payload = json.loads(event.payload) if isinstance(event.payload, (str, bytes)) else event.payload
                    request_id = payload.get("requestId") if isinstance(payload, dict) else None

                    # Dedup — follows EventBridge processed_events pattern
                    if request_id and request_id in self._processed_requests:
                        logger.debug("Skipping duplicate inference request %s", request_id)
                        continue

                    if InferenceRequest is None:
                        logger.warning("InferenceRequest contract not available — skipping")
                        continue

                    request = InferenceRequest.model_validate(payload)
                    bid = self.calculate_bid(request)
                    if bid is not None:
                        await self._submit_bid(bid, request.model_id)

                    if request_id:
                        self._processed_requests.append(request_id)

                except json.JSONDecodeError as e:
                    logger.error("Failed to deserialize inference request: %s", e)
                except Exception as e:
                    logger.error("Error processing inference request: %s", e, exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Auction listener loop failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Bid calculation (T9.2)
    # ------------------------------------------------------------------

    def calculate_bid(self, request: "InferenceRequest") -> "InferenceBid | None":
        """Evaluate local capacity and produce a bid, or ``None`` to abstain.

        Scoring:
            WARM    → base 1.0
            COLD    → base 0.5
            LOADING → base 0.3
            UNKNOWN / absent → ``None`` (abstain)

        Modifiers:
            is_transient + COLD  → score \u00d7 0.3
            Circuit breaker open → ``None``
            VRAM free < required → ``None``
            ICapacityAdvisor.adjust_bid_score() if wired
        """
        if InferenceBid is None or ModelState is None:
            logger.debug(
                "Bid suppressed: model=%s, reason=contracts_unavailable",
                request.model_id,
            )
            return None

        # ---- VRAM pressure gate (suppresses ALL bids) ----
        if self._is_vram_pressure_active():
            logger.warning("VRAM pressure gate active — suppressing all bids")
            return None

        # ---- Model state lookup ----
        model_state = ModelState.UNKNOWN
        telemetry = None
        if self._model_registry is not None:
            model_state = (
                self._model_registry.get_model_state(request.model_id)
                or ModelState.UNKNOWN
            )
            all_telemetry = self._model_registry.get_all_telemetry()
            telemetry = all_telemetry.get(request.model_id)

        if model_state == ModelState.UNKNOWN:
            logger.debug(
                "Bid suppressed: model=%s, reason=unknown_model", request.model_id
            )
            return None

        # ---- Circuit breaker check ----
        if self._circuit_breaker_registry is not None and telemetry is not None:
            provider_name = getattr(telemetry, "provider_name", request.model_id)
            breaker = self._circuit_breaker_registry.get_or_create(provider_name)
            if not breaker.is_available():
                logger.debug(
                    "Bid suppressed: model=%s, reason=circuit_breaker_open",
                    request.model_id,
                )
                return None

        # ---- VRAM requirement check ----
        vram_free = self._get_vram_free_mb()
        if request.vram_requirement_mb is not None and vram_free is not None:
            if vram_free < request.vram_requirement_mb:
                logger.debug(
                    "Bid suppressed: model=%s, reason=insufficient_vram "
                    "(free=%.0f, required=%.0f)",
                    request.model_id,
                    vram_free,
                    request.vram_requirement_mb,
                )
                return None

        # ---- Base scoring ----
        is_warm = model_state == ModelState.WARM
        if model_state == ModelState.WARM:
            score = 1.0
        elif model_state == ModelState.COLD:
            score = 0.5
        elif model_state == ModelState.LOADING:
            score = 0.3
        else:
            logger.debug(
                "Bid suppressed: model=%s, reason=unhandled_state_%s",
                request.model_id,
                model_state,
            )
            return None

        # Transient + COLD penalty
        if request.is_transient and model_state == ModelState.COLD:
            score *= 0.3

        # Latency estimate by state
        estimated_latency_ms = {
            ModelState.WARM: 100,
            ModelState.LOADING: 5000,
            ModelState.COLD: 10000,
        }.get(model_state, 10000)

        bid = InferenceBid(
            request_id=request.request_id,
            node_id=self._node_id,
            score=score,
            is_warm=is_warm,
            estimated_latency_ms=estimated_latency_ms,
            vram_free_mb=vram_free,
            bid_timestamp=datetime.now(timezone.utc),
        )

        # ---- Optional capacity advisor adjustment ----
        if self._capacity_advisor is not None:
            try:
                adjusted = self._capacity_advisor.adjust_bid_score(bid, request)
                if 0.0 <= adjusted <= 1.0:
                    bid = bid.model_copy(update={"score": adjusted})
                else:
                    logger.warning(
                        "CapacityAdvisor returned out-of-range score %.2f; keeping %.2f",
                        adjusted,
                        score,
                    )
            except Exception as e:
                logger.warning("CapacityAdvisor.adjust_bid_score failed: %s", e)

        return bid

    # ------------------------------------------------------------------
    # Bid submission (T9.3)
    # ------------------------------------------------------------------

    async def _submit_bid(self, bid: "InferenceBid", model_id: str) -> None:
        """Publish bid to ``aurarouter.inference_bids.{request_id}``."""
        topic = f"{INFERENCE_BIDS_TOPIC_PREFIX}.{bid.request_id}"
        try:
            payload = bid.model_dump(by_alias=True, mode="json")
            await self._event_bridge.publish(
                topic, payload, event_type="aurarouter.inference_bid"
            )
            logger.info(
                "Bid submitted: model=%s, score=%.2f, warm=%s",
                model_id,
                bid.score,
                bid.is_warm,
            )
        except Exception as e:
            logger.error(
                "Failed to submit bid for request %s: %s", bid.request_id, e
            )

    # ------------------------------------------------------------------
    # VRAM helpers (T9.4)
    # ------------------------------------------------------------------

    def _is_vram_pressure_active(self) -> bool:
        """True when aggregate VRAM utilisation exceeds the pressure threshold."""
        if self._total_vram_mb is None or self._total_vram_mb <= 0:
            return False
        if self._model_registry is None:
            return False
        total_usage = sum(
            t.vram_usage_mb
            for t in self._model_registry.get_all_telemetry().values()
            if getattr(t, "vram_usage_mb", None) is not None
        )
        return (total_usage / self._total_vram_mb) > self._vram_pressure_threshold

    def _get_vram_free_mb(self) -> float | None:
        """Calculate free VRAM from registry telemetry."""
        if self._total_vram_mb is None or self._model_registry is None:
            return None
        total_usage = sum(
            t.vram_usage_mb
            for t in self._model_registry.get_all_telemetry().values()
            if getattr(t, "vram_usage_mb", None) is not None
        )
        return max(0.0, self._total_vram_mb - total_usage)
