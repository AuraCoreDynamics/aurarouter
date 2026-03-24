# AuraRouter External Provider Template

This is a minimal skeleton for creating an external AuraRouter provider
package that runs as an MCP server.

## MCP Provider Protocol

External providers communicate with AuraRouter via the MCP Provider
Protocol over JSON-RPC 2.0. Your package must expose an MCP server with
the following tools:

### Required Tools

| Tool Name               | Description                      |
|-------------------------|----------------------------------|
| `provider.generate`     | Single-shot text generation      |
| `provider.list_models`  | Enumerate available models       |

### Optional Tools

| Tool Name                          | Description                    |
|------------------------------------|--------------------------------|
| `provider.generate_with_history`   | Multi-turn generation          |
| `provider.health_check`            | Liveness/readiness probe       |
| `provider.capabilities`            | Advertise provider features    |

## Entry Point Format

Register your provider via a Python entry point in `pyproject.toml`:

```toml
[project.entry-points."aurarouter.providers"]
my_provider = "my_package:get_provider_metadata"
```

The entry point callable must return a `ProviderMetadata` instance:

```python
from aurarouter.providers.protocol import ProviderMetadata

def get_provider_metadata() -> ProviderMetadata:
    return ProviderMetadata(
        name="my-provider",
        provider_type="mcp",
        version="0.1.0",
        description="My custom LLM provider",
        command=["python", "-m", "my_package.server"],
        requires_config=["api_key"],
        homepage="https://github.com/example/my-provider",
    )
```

## Testing

Run protocol compliance tests:

```bash
pip install -e ".[dev]"
pytest tests/test_protocol.py -v
```

## Quick Start

1. Copy this template directory
2. Rename `aurarouter_example` to your package name
3. Update `pyproject.toml.template` (rename to `pyproject.toml`)
4. Implement your provider logic in `server.py`
5. Update the entry point in `__init__.py`
6. Install and test: `pip install -e . && pytest`
