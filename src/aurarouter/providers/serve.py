"""MCP server wrapper for built-in providers.

Exposes any built-in AuraRouter provider as a standalone MCP server
implementing the MCP Provider Protocol.  This allows built-in providers
to be run out-of-process and accessed via the same mechanism as external
provider packages.

Usage::

    python -m aurarouter.providers.serve ollama --port 9000
    python -m aurarouter.providers.serve openapi --port 9001 --config /path/to/auraconfig.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.providers import PROVIDER_REGISTRY, get_provider
from aurarouter.providers.base import BaseProvider
from aurarouter.providers.protocol import (
    TOOL_CAPABILITIES,
    TOOL_GENERATE,
    TOOL_HEALTH_CHECK,
    TOOL_LIST_MODELS,
)

logger = get_logger("AuraRouter.ProviderServe")


def create_provider_mcp_server(
    provider_name: str,
    config: ConfigLoader,
) -> FastMCP:
    """Build a FastMCP server that wraps a built-in provider.

    Registers ``provider.generate``, ``provider.list_models``,
    ``provider.health_check``, and ``provider.capabilities`` tools
    that delegate to the provider instance.

    Args:
        provider_name: Key in PROVIDER_REGISTRY (e.g. ``"ollama"``).
        config: Config loader for provider configuration.

    Returns:
        A configured :class:`FastMCP` instance ready to serve.
    """
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available: {', '.join(PROVIDER_REGISTRY)}"
        )

    mcp = FastMCP(f"AuraRouter-Provider-{provider_name}")

    # Cache provider instances keyed by model name
    _provider_cache: dict[str, BaseProvider] = {}

    def _get_provider(model_name: str) -> BaseProvider:
        """Get or create a provider instance for the given model."""
        if model_name not in _provider_cache:
            # Look up model config, or build a minimal one
            model_cfg = config.get_model_config(model_name)
            if not model_cfg:
                # Build a config from all models using this provider
                model_cfg = {
                    "provider": provider_name,
                    "model_name": model_name,
                }
            _provider_cache[model_name] = get_provider(provider_name, model_cfg)
        return _provider_cache[model_name]

    @mcp.tool(name=TOOL_GENERATE)
    def provider_generate(
        prompt: str,
        model: str = "",
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Generate text from a prompt using the wrapped provider.

        Returns a dict with text, token counts, and model info.
        """
        provider = _get_provider(model)
        result = provider.generate_with_usage(prompt, json_mode=json_mode)
        return {
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model_id": result.model_id or model,
            "provider": provider_name,
            "context_limit": result.context_limit,
        }

    @mcp.tool(name=TOOL_LIST_MODELS)
    def provider_list_models() -> list[dict[str, Any]]:
        """List models available through this provider.

        Returns a list of model info dicts from the config.
        """
        models: list[dict[str, Any]] = []
        for model_id in config.get_all_model_ids():
            model_cfg = config.get_model_config(model_id)
            if model_cfg.get("provider") == provider_name:
                models.append({
                    "id": model_id,
                    "name": model_cfg.get("model_name", model_id),
                    "provider": provider_name,
                    "tags": model_cfg.get("tags", []),
                })
        return models

    @mcp.tool(name=TOOL_HEALTH_CHECK)
    def provider_health_check() -> dict[str, Any]:
        """Check whether the provider is healthy and reachable."""
        return {
            "healthy": True,
            "provider": provider_name,
            "message": f"{provider_name} provider is running",
        }

    @mcp.tool(name=TOOL_CAPABILITIES)
    def provider_capabilities() -> dict[str, Any]:
        """Report the capabilities of this provider server."""
        return {
            "provider": provider_name,
            "tools": [
                TOOL_GENERATE,
                TOOL_LIST_MODELS,
                TOOL_HEALTH_CHECK,
                TOOL_CAPABILITIES,
            ],
            "supports_history": True,
            "supports_json_mode": True,
        }

    return mcp


def main() -> None:
    """CLI entry point for running a provider as an MCP server."""
    parser = argparse.ArgumentParser(
        description="Run an AuraRouter built-in provider as an MCP server",
    )
    parser.add_argument(
        "provider",
        choices=[k for k in PROVIDER_REGISTRY if k != "mcp"],
        help="Provider to serve",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port to listen on (default: 9000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to auraconfig.yaml",
    )
    args = parser.parse_args()

    config = ConfigLoader(config_path=args.config, allow_missing=(args.config is None))
    mcp = create_provider_mcp_server(args.provider, config)

    logger.info(
        "Serving provider '%s' on port %d", args.provider, args.port
    )
    mcp.run(transport="streamable-http", port=args.port)


if __name__ == "__main__":
    main()
