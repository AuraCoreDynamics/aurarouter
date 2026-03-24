"""McpProvider -- BaseProvider adapter for remote MCP provider servers.

Wraps a remote MCP server that implements the MCP Provider Protocol
(see :mod:`aurarouter.providers.protocol`) behind the standard
:class:`BaseProvider` interface, so the :class:`ComputeFabric` can
route to it transparently.

Config keys::

    my_remote_model:
      provider: mcp
      mcp_endpoint: http://localhost:9001   # or 'endpoint'
      model_name: gemini-2.0-flash
      timeout: 120.0                        # optional, seconds
"""

from __future__ import annotations

from aurarouter._logging import get_logger
from aurarouter.mcp_client.client import GridMcpClient
from aurarouter.providers.base import BaseProvider
from aurarouter.providers.protocol import (
    REQUIRED_TOOLS,
    TOOL_CAPABILITIES,
    TOOL_GENERATE,
    TOOL_GENERATE_WITH_HISTORY,
    TOOL_LIST_MODELS,
    validate_provider_tools,
)
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.McpProvider")


class McpProvider(BaseProvider):
    """Provider adapter that delegates to a remote MCP server.

    The remote server must implement at least ``provider.generate`` and
    ``provider.list_models``.  Optional tools like
    ``provider.generate_with_history`` are used when available.
    """

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)
        endpoint = model_config.get("mcp_endpoint") or model_config.get(
            "endpoint", ""
        )
        if not endpoint:
            raise ValueError(
                "McpProvider requires 'mcp_endpoint' or 'endpoint' in config"
            )
        self._endpoint = endpoint
        self._model_name = model_config.get("model_name", "")
        self._timeout = float(model_config.get("timeout", 120.0))
        self._client = GridMcpClient(
            base_url=self._endpoint,
            name=f"mcp-provider:{self._model_name or endpoint}",
            timeout=self._timeout,
        )
        self._remote_capabilities: set[str] = set()
        self._validated = False

    # ------------------------------------------------------------------
    # Connection & validation
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Connect to the remote MCP server and validate protocol compliance.

        Raises:
            ConnectionError: If the server is unreachable.
            RuntimeError: If the server does not satisfy the MCP Provider Protocol.
        """
        if self._validated and self._client.connected:
            return

        if not self._client.connect():
            raise ConnectionError(
                f"McpProvider could not connect to {self._endpoint}"
            )

        tools = self._client.get_tools()
        valid, errors = validate_provider_tools(tools)
        if not valid:
            raise RuntimeError(
                f"MCP server at {self._endpoint} does not satisfy the "
                f"provider protocol: {'; '.join(errors)}"
            )

        self._remote_capabilities = self._client.get_capabilities()
        self._validated = True

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Single-shot generation via ``provider.generate``."""
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
        """Generate and return a :class:`GenerateResult` with usage metadata."""
        self._ensure_connected()

        result = self._client.call_tool(
            TOOL_GENERATE,
            prompt=prompt,
            model=self._model_name,
            json_mode=json_mode,
        )

        if isinstance(result, str):
            return GenerateResult(
                text=result,
                model_id=self._model_name,
                provider="mcp",
            )

        if isinstance(result, dict):
            return GenerateResult(
                text=result.get("text", ""),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                model_id=result.get("model_id", self._model_name),
                provider="mcp",
                context_limit=result.get("context_limit", 0),
            )

        return GenerateResult(
            text=str(result),
            model_id=self._model_name,
            provider="mcp",
        )

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation, delegating to remote if supported."""
        self._ensure_connected()

        if TOOL_GENERATE_WITH_HISTORY not in self._remote_capabilities:
            # Fall back to BaseProvider's concatenation strategy
            return super().generate_with_history(
                messages, system_prompt=system_prompt, json_mode=json_mode
            )

        result = self._client.call_tool(
            TOOL_GENERATE_WITH_HISTORY,
            messages=messages,
            system_prompt=system_prompt,
            model=self._model_name,
            json_mode=json_mode,
        )

        if isinstance(result, dict):
            return GenerateResult(
                text=result.get("text", ""),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                model_id=result.get("model_id", self._model_name),
                provider="mcp",
                context_limit=result.get("context_limit", 0),
                gist=result.get("gist"),
            )

        return GenerateResult(
            text=str(result),
            model_id=self._model_name,
            provider="mcp",
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Call ``provider.list_models`` on the remote server."""
        self._ensure_connected()
        result = self._client.call_tool(TOOL_LIST_MODELS)
        return result if isinstance(result, list) else []

    def get_capabilities(self) -> set[str]:
        """Return the remote server's advertised tool capabilities."""
        self._ensure_connected()
        return set(self._remote_capabilities)
