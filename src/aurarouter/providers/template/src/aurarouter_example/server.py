"""Example MCP provider server implementing the AuraRouter Provider Protocol.

Replace the placeholder logic with your actual provider integration.

Usage::

    python -m aurarouter_example.server --port 9000
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AuraRouter-Example-Provider")


@mcp.tool(name="provider.generate")
def provider_generate(
    prompt: str,
    model: str = "",
    json_mode: bool = False,
) -> dict[str, Any]:
    """Generate text from a prompt.

    Replace this with your actual generation logic.
    """
    # --- Replace with real provider call ---
    text = f"[example] Echo: {prompt}"
    return {
        "text": text,
        "input_tokens": len(prompt.split()),
        "output_tokens": len(text.split()),
        "model_id": model or "example-model",
        "provider": "example",
        "context_limit": 4096,
    }


@mcp.tool(name="provider.list_models")
def provider_list_models() -> list[dict[str, Any]]:
    """List available models.

    Replace with logic to enumerate your provider's models.
    """
    return [
        {
            "id": "example-model",
            "name": "Example Model",
            "provider": "example",
            "tags": ["example"],
        },
    ]


@mcp.tool(name="provider.health_check")
def provider_health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "healthy": True,
        "provider": "example",
        "message": "Example provider is running",
    }


@mcp.tool(name="provider.capabilities")
def provider_capabilities() -> dict[str, Any]:
    """Advertise provider capabilities."""
    return {
        "provider": "example",
        "tools": [
            "provider.generate",
            "provider.list_models",
            "provider.health_check",
            "provider.capabilities",
        ],
        "supports_history": False,
        "supports_json_mode": False,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the example AuraRouter provider MCP server",
    )
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    mcp.run(transport="streamable-http", port=args.port)


if __name__ == "__main__":
    main()
