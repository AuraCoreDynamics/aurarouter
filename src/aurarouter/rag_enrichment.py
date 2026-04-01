"""RAG context enrichment pipeline for the IPE loop.

Calls AuraXLM's ``auraxlm.search`` MCP tool to retrieve relevant context
snippets and injects them into the prompt before execution.  Degrades
gracefully on timeout or MCP failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from aurarouter.config import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichedContext:
    """Result of RAG context enrichment."""

    original_task: str
    rag_snippets: list[dict] = field(default_factory=list)
    total_tokens_used: int = 0
    retrieval_latency_ms: float = 0.0
    source: str = "none"  # "auraxlm" or "none"


class RagEnrichmentPipeline:
    """Retrieve relevant context from AuraXLM via MCP and inject into prompts."""

    def __init__(
        self,
        mcp_registry,  # McpClientRegistry
        config: ConfigLoader,
    ) -> None:
        self._registry = mcp_registry
        self._config = config

    def is_enabled(self) -> bool:
        """Check if RAG enrichment is enabled in config."""
        system = self._config.config.get("system", {})
        return system.get("rag_enrichment", False)

    async def enrich(
        self,
        task: str,
        context: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 5.0,
    ) -> EnrichedContext:
        """Call auraxlm.search to retrieve relevant context snippets.

        Falls back gracefully (returns original context) on timeout or
        MCP failure.
        """
        if not self.is_enabled():
            return EnrichedContext(original_task=task)

        endpoint = self._config.get_xlm_endpoint()
        if not endpoint:
            logger.debug("RAG enrichment enabled but no XLM endpoint configured.")
            return EnrichedContext(original_task=task)

        start = time.monotonic()
        try:
            client = self._get_xlm_client(endpoint)
            if client is None:
                return EnrichedContext(original_task=task)

            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.call_tool(
                        "auraxlm.search",
                        query=task,
                        maxResults=5,
                    ),
                ),
                timeout=timeout,
            )

            latency = (time.monotonic() - start) * 1000
            snippets = self._extract_snippets(result)
            tokens = self._estimate_tokens(snippets, max_tokens)

            return EnrichedContext(
                original_task=task,
                rag_snippets=tokens,
                total_tokens_used=sum(
                    len(s.get("content", "")) // 4 for s in tokens
                ),
                retrieval_latency_ms=latency,
                source="auraxlm",
            )

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            logger.warning(
                "RAG enrichment timed out after %.0fms (limit %.0fms).",
                latency,
                timeout * 1000,
            )
        except Exception as exc:
            logger.warning("RAG enrichment failed: %s", exc)
            logger.debug("RAG enrichment error details", exc_info=True)

        return EnrichedContext(original_task=task)

    def build_enriched_prompt(
        self, task: str, enriched: EnrichedContext
    ) -> str:
        """Inject RAG snippets into a prompt if available."""
        if not enriched.rag_snippets:
            return task

        snippet_text = "\n".join(
            f"- {s.get('content', '')}" for s in enriched.rag_snippets
        )
        return f"{task}\n\n--- Relevant Context ---\n{snippet_text}"

    def _get_xlm_client(self, endpoint: str):
        """Get or create an XLM MCP client."""
        # Prefer a registered client with auraxlm.search capability.
        clients = self._registry.get_clients_with_capability("search")
        if clients:
            return clients[0]

        # Fall back to direct connection.
        from aurarouter.mcp_client.client import GridMcpClient

        client = GridMcpClient(
            base_url=endpoint, name="rag-enrichment", timeout=5.0
        )
        if client.connect():
            return client
        return None

    def _extract_snippets(self, result) -> list[dict]:
        """Extract search result snippets from MCP response."""
        if isinstance(result, dict):
            results = result.get("results", [])
            if isinstance(results, list):
                return results
        if isinstance(result, list):
            return result
        return []

    def _estimate_tokens(
        self, snippets: list[dict], max_tokens: int
    ) -> list[dict]:
        """Trim snippets to fit within the token budget."""
        budget = max_tokens
        trimmed: list[dict] = []
        for s in snippets:
            content = s.get("content", "")
            est = len(content) // 4  # rough char-to-token ratio
            if est > budget:
                break
            trimmed.append(s)
            budget -= est
        return trimmed
