"""Federated Broker for AuraRouter.

Broadcasts routing requests to all registered analyzers, collects structured
bids, and merges non-conflicting bids into a scatter-gather plan.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader

logger = logging.getLogger(__name__)

BROADCAST_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Bid and result schemas
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerBid:
    """Structured bid from an analyzer claiming work on a task."""

    analyzer_id: str
    confidence: float  # 0.0–1.0
    claimed_files: list[str] = field(default_factory=list)
    proposed_tasks: list[dict] = field(default_factory=list)
    role: str = "coding"

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def overlaps_with(self, other: AnalyzerBid) -> bool:
        """True if both bids claim any of the same files."""
        if not self.claimed_files or not other.claimed_files:
            return False
        return bool(set(self.claimed_files) & set(other.claimed_files))


@dataclass
class BrokerResult:
    """Aggregated result from the federated broker."""

    bids: list[AnalyzerBid] = field(default_factory=list)
    collisions: list[tuple[AnalyzerBid, AnalyzerBid]] = field(default_factory=list)
    merged_plan: list[dict] | None = None
    mismatch: bool = False
    execution_trace: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Broadcast-and-collect loop
# ---------------------------------------------------------------------------

def _parse_bid(analyzer_id: str, response: dict) -> AnalyzerBid | None:
    """Try to parse a response dict into an AnalyzerBid."""
    try:
        confidence = float(response.get("confidence", 0.0))
        return AnalyzerBid(
            analyzer_id=analyzer_id,
            confidence=confidence,
            claimed_files=response.get("claimed_files", []),
            proposed_tasks=response.get("proposed_tasks", []),
            role=response.get("role", "coding"),
        )
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("Failed to parse bid from %s: %s", analyzer_id, exc)
        return None


async def _call_single_analyzer(
    endpoint: str,
    tool_name: str,
    prompt: str,
    options: dict | None,
    timeout: float,
) -> dict | None:
    """Call a single remote analyzer via MCP JSON-RPC."""
    import httpx

    payload: dict = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": {"prompt": prompt},
        },
        "id": 1,
    }
    if options:
        payload["params"]["arguments"]["options"] = options

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=payload)
        if resp.status_code == 200:
            import json
            data = resp.json()
            result = data.get("result", {})
            if isinstance(result, str):
                result = json.loads(result)
            return result
    return None


async def broadcast_to_analyzers(
    config: ConfigLoader,
    prompt: str,
    options: dict | None = None,
    timeout: float = BROADCAST_TIMEOUT_S,
) -> list[AnalyzerBid]:
    """Broadcast prompt to all registered analyzers and collect bids.

    Discovers analyzers via ``config.catalog_query(kind="analyzer")``,
    skips ``aurarouter-default`` and analyzers without ``mcp_endpoint``,
    and collects structured bids concurrently.
    """
    analyzers = config.catalog_query(kind="analyzer")
    trace: list[str] = []

    # Filter to remote analyzers (exclude broker itself)
    targets: list[dict] = []
    for a in analyzers:
        aid = a.get("artifact_id", "")
        if aid == "aurarouter-default":
            continue
        if not a.get("mcp_endpoint"):
            continue
        targets.append(a)

    trace.append(f"Broker: broadcast to {len(targets)} analyzers")
    logger.info("Broker: broadcasting to %d analyzers", len(targets))

    if not targets:
        return []

    async def _collect(analyzer: dict) -> AnalyzerBid | None:
        aid = analyzer["artifact_id"]
        endpoint = analyzer["mcp_endpoint"]
        tool_name = analyzer.get("mcp_tool_name", "")
        try:
            result = await asyncio.wait_for(
                _call_single_analyzer(endpoint, tool_name, prompt, options, timeout),
                timeout=timeout,
            )
            if result is None:
                trace.append(f"Broker: {aid} returned empty response")
                return None
            bid = _parse_bid(aid, result)
            if bid is not None:
                trace.append(
                    f"Broker: {aid} responded (confidence={bid.confidence})"
                )
            else:
                trace.append(f"Broker: {aid} returned unparseable data")
            return bid
        except asyncio.TimeoutError:
            trace.append(f"Broker: {aid} timed out")
            logger.warning("Broker: analyzer %s timed out", aid)
            return None
        except Exception as exc:
            trace.append(f"Broker: {aid} failed ({exc})")
            logger.warning("Broker: analyzer %s failed: %s", aid, exc)
            return None

    results = await asyncio.gather(
        *[_collect(a) for a in targets],
        return_exceptions=True,
    )

    bids: list[AnalyzerBid] = []
    for r in results:
        if isinstance(r, AnalyzerBid):
            bids.append(r)
        elif isinstance(r, Exception):
            trace.append(f"Broker: gather exception: {r}")

    # Attach trace to the bids list as an attribute for the caller to read
    # (the caller should use merge_bids which builds its own trace)
    # Store on module-level for broadcast_to_analyzers callers
    broadcast_to_analyzers._last_trace = trace  # type: ignore[attr-defined]

    return bids


# ---------------------------------------------------------------------------
# Bid merging with hint validation
# ---------------------------------------------------------------------------

def merge_bids(
    bids: list[AnalyzerBid],
    routing_hints: list[str] | None = None,
) -> BrokerResult:
    """Merge non-conflicting bids into a scatter-gather plan.

    HINT-TO-BID VALIDATION: Before merging, verify that at least one bid
    corresponds to the routing hints. If no bid matches, flag ROUTING_MISMATCH.

    If any two bids claim the same files with confidence > 0.5, flag as collisions.
    Otherwise, merge into sequential plan ordered by confidence (descending).
    """
    trace: list[str] = []

    # Pull in broadcast trace if available
    broadcast_trace = getattr(broadcast_to_analyzers, "_last_trace", [])
    trace.extend(broadcast_trace)

    trace.append(f"Broker: {len(bids)} bids received")

    if not bids:
        return BrokerResult(bids=[], execution_trace=trace)

    # Hint validation
    mismatch = False
    if routing_hints is not None and len(routing_hints) > 0:
        hint_set = {h.lower() for h in routing_hints}
        # A bid matches hints if its role matches any hint, or if any of its
        # claimed_files extensions match, or if the analyzer_id contains a hint
        matched = False
        for bid in bids:
            bid_tokens = {bid.role.lower(), bid.analyzer_id.lower()}
            # Also check claimed file extensions
            for f in bid.claimed_files:
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                if ext:
                    bid_tokens.add(ext)
            if hint_set & bid_tokens:
                matched = True
                break
        if not matched:
            mismatch = True
            trace.append("Broker: hint validation failed")
        else:
            trace.append("Broker: hint validation passed")
    else:
        trace.append("Broker: hint validation skipped (no hints)")

    # Detect collisions: pairs of bids claiming same files both with confidence > 0.5
    collisions: list[tuple[AnalyzerBid, AnalyzerBid]] = []
    for i in range(len(bids)):
        for j in range(i + 1, len(bids)):
            if (
                bids[i].confidence > 0.5
                and bids[j].confidence > 0.5
                and bids[i].overlaps_with(bids[j])
            ):
                collisions.append((bids[i], bids[j]))
                trace.append(
                    f"Broker: collision detected between "
                    f"{bids[i].analyzer_id} and {bids[j].analyzer_id}"
                )

    # Merge plan if no collisions
    merged_plan: list[dict] | None = None
    if not collisions:
        sorted_bids = sorted(bids, key=lambda b: b.confidence, reverse=True)
        merged_plan = []
        for bid in sorted_bids:
            merged_plan.append({
                "analyzer_id": bid.analyzer_id,
                "role": bid.role,
                "confidence": bid.confidence,
                "claimed_files": bid.claimed_files,
                "proposed_tasks": bid.proposed_tasks,
            })
        trace.append("Broker: fast-path merge")

    return BrokerResult(
        bids=bids,
        collisions=collisions,
        merged_plan=merged_plan,
        mismatch=mismatch,
        execution_trace=trace,
    )
