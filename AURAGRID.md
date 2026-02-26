# AuraRouter + AuraGrid Integration Guide

AuraRouter can be deployed as a **Managed Application Service (MAS)** on AuraGrid, enabling other grid applications to access routing, reasoning, and task execution services. AuraRouter is content-agnostic -- it routes any prompt-based task (code generation, summarization, analysis, Q&A, etc.) across local and cloud models. This guide covers deployment, configuration, and usage patterns.

## Overview

- **Standalone**: `pip install aurarouter` works independently
- **On AuraGrid**: `pip install aurarouter[auragrid]` enables grid integration
- **Services**: Four discoverable services (RouterService, ReasoningService, CodingService, UnifiedRouterService)
- **Communication**: Both synchronous (gRPC) and asynchronous (events) modes supported

## Quick Start

### 1. Install with AuraGrid Support

```bash
# On the AuraGrid node where aurarouter will run
pip install aurarouter[auragrid]
```

### 2. Configure AuraRouter

Place `auraconfig.yaml` in one of the standard locations:
- Current directory
- `~/.auracore/aurarouter/auraconfig.yaml`
- Path specified via `AURACORE_ROUTER_CONFIG` environment variable

See [Configuration](#configuration) section below.

### 3. Deploy on AuraGrid

Copy the manifest to your AuraGrid manifests directory:

```bash
# From aurarouter repo
cp manifests/auragrid_manifest.json /path/to/auragrid/manifests/
```

AuraGrid will:
1. Discover the manifest
2. Install aurarouter if not present
3. Start aurarouter as a Distributed MAS (runs on every node)
4. Register all four services in the grid's service registry

## Configuration

### Configuration Precedence

Configuration is loaded with this priority (highest to lowest):

1. **Environment variables** (e.g., `AURAROUTER_MODELS__CLAUDE__API_KEY`)
2. **AuraGrid manifest metadata**
3. **`auraconfig.yaml` file**
4. **Built-in defaults**

### File Format

See `manifests/sample_config.yaml` for a fully commented template. Key sections:

```yaml
system:
  log_level: INFO
  default_timeout: 120.0

models:
  # Define all available models with provider type and config
  my_model:
    provider: ollama          # or: google, claude, llamacpp, llamacpp-server, openapi
    endpoint: http://...      # If applicable
    model_name: ...
    api_key: ...              # Set via env var for security
    tags: [private, fast]     # Optional capability tags (used for privacy routing)
    parameters:
      temperature: 0.1
      num_ctx: 4096

roles:
  router:                      # Intent classification
    models:
      - model_a
      - model_b               # Fallback
  reasoning:                   # Planning
    models:
      - model_c
  coding:                      # Code generation
    models:
      - model_a
```

### Environment Variable Overrides

Use `AURAROUTER_` prefix for any config value:

```bash
# Set Gemini API key
export AURAROUTER_MODELS__CLOUD_GEMINI__API_KEY=sk-...

# Set Ollama endpoint
export AURAROUTER_MODELS__LOCAL_QWEN__ENDPOINT=http://192.168.1.50:11434/api/generate

# Override log level
export AURAROUTER_SYSTEM__LOG_LEVEL=DEBUG
```

Use `__` to denote nesting levels.

## Service API Reference

### RouterService

Classifies task intent (Simple vs Complex).

```yaml
Method: classify_intent
Input:
  task_description: str       # The task to classify
  context: Optional[Dict]     # Additional context
Output:
  {
    "classification": str,    # Classification result
    "task": str,
    "success": bool
  }
```

**Example (gRPC)**:
```python
from auragrid.services import RouterService_client

result = await client.classify_intent(
    task_description="Write a Python function to compute factorials"
)
print(result["classification"])
```

### ReasoningService

Generates execution plans for complex tasks.

```yaml
Method: generate_plan
Input:
  intent: str                 # The task/intent
  context: Optional[Dict]     # Additional context
Output:
  {
    "steps": List[Any],       # Plan steps
    "intent": str,
    "step_count": int,
    "success": bool
  }
```

**Example**:
```python
result = await client.generate_plan(
    intent="Write a function to compute factorials"
)
for step in result["steps"]:
    print(step)
```

### CodingService

Generates code for a given plan step.

```yaml
Method: generate_code
Input:
  plan_step: str              # Description of step to code
  language: str = "python"    # Target language
Output:
  {
    "code": str,              # Generated code
    "language": str,
    "plan_step": str,
    "success": bool
  }
```

**Example**:
```python
result = await client.generate_code(
    plan_step="Implement factorial calculation with recursion",
    language="python"
)
print(result["code"])
```

### UnifiedRouterService

Unified endpoint that orchestrates routing, planning, and code generation.

```yaml
Method: intelligent_code_gen
Input:
  task: str                        # Task description
  language: str = "python"         # Target language
  file_context: Optional[str]      # Existing code context
Output:
  {
    "result": str,                 # Generated code
    "task": str,
    "language": str,
    "context_provided": bool,
    "success": bool
  }
```

**Example**:
```python
result = await client.intelligent_code_gen(
    task="Create a REST API endpoint",
    language="python",
    file_context="# Existing Flask app code..."
)
```

## Communication Patterns

### Synchronous (gRPC Proxy)

Call aurarouter services synchronously through AuraGrid's gRPC proxy:

```python
from auragrid import create_service_client

# Grid framework discovers service endpoint automatically
client = create_service_client("UnifiedRouterService")

result = await client.intelligent_code_gen(
    task="Write a validator function",
    language="typescript"
)
```

**Pros**: Simple, immediate results, type-safe (with proto definitions)  
**Cons**: Blocking, resource-intensive for large tasks

### Asynchronous (Events)

Publish tasks to aurarouter topics; subscribe to results asynchronously:

```python
from auragrid import event_publisher, event_subscriber

# Publish task
request = {
    "request_id": "uuid",
    "task": "Large code generation",
    "language": "python",
    "return_topic": "my_app.results"
}

await event_publisher.publish(
    topic="aurarouter.routing_requests",
    payload=json.dumps(request)
)

# Subscribe to results
async for event in event_subscriber.consume("my_app.results"):
    result = json.loads(event.payload)
    print(result["result"])
```

**Pros**: Non-blocking, efficient for bulk tasks  
**Cons**: Requires event coordination logic

## GUI -- Grid Administration

The AuraRouter desktop GUI supports AuraGrid environments with dedicated panels and controls.

### Switching to AuraGrid

1. Launch the GUI: `aurarouter gui` (or `aurarouter gui --environment auragrid`)
2. Select **AuraGrid** from the environment dropdown in the toolbar.
3. The GUI will rebuild panels for the grid context.

### Cell-Wide Configuration Warning

When saving configuration changes in AuraGrid mode, the GUI displays a prominent warning:

> "This configuration change will propagate to all nodes on your AuraGrid cell. Proceed?"

A yellow banner also appears at the top of the Configuration tab as a reminder.

### Deployment Strategy Panel

The **Deployment** tab (AuraGrid only) provides:
- Model replica count management (current vs. desired replicas per model).
- Cell resource overview (discovered Ollama endpoints and availability).
- "Apply Strategy" button to push deployment changes to the grid orchestration API.

### Cell Status Panel

The **Cell Status** tab (AuraGrid only) provides:
- Node list with ID, address, health status, loaded models, and last-seen timestamp.
- Event log showing recent routing requests and results from the EventBridge.
- Auto-refresh every 30 seconds (or manual "Refresh Now").

### Grid Model Management

The **Models** tab shows a second section for grid models when AuraGrid is active, listing model IDs distributed across the grid via `GridModelStorage`.

See [GUI_GUIDE.md](GUI_GUIDE.md) for the complete GUI reference.

---

## Troubleshooting

### Service Not Discoverable

Check that aurarouter's MAS host started successfully:

```bash
# On AuraGrid node, check logs
journalctl -u auragrid-proxy-worker | grep -i aurarouter

# Verify service registration
auragrid service list | grep -i "Router"
```

### Model Provider Connectivity

Test provider endpoints manually:

```bash
# Test Ollama
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5-coder:7b","prompt":"test"}'

# Test Gemini (requires API key)
python -c "from aurarouter.providers import get_provider; p = get_provider('google', {...}); print(p.generate('test'))"
```

### Configuration Not Applied

Verify precedence:

```bash
# Check env vars
env | grep AURAROUTER_

# Check file location
ls -la ~/.auracore/aurarouter/auraconfig.yaml

# Set debug logging
export AURAROUTER_SYSTEM__LOG_LEVEL=DEBUG
```

### API Key Issues

```bash
# For Gemini
export AURAROUTER_MODELS__CLOUD_GEMINI__API_KEY=$GOOGLE_API_KEY

# For Claude  
export AURAROUTER_MODELS__CLOUD_CLAUDE__API_KEY=$ANTHROPIC_API_KEY

# Verify key is set (without printing it)
echo ${AURAROUTER_MODELS__CLOUD_GEMINI__API_KEY:+SET}
```

## Example: Grid Consumer App

See `examples/grid_consumer_app.py` for a complete example that:
- Creates a service client to UnifiedRouterService
- Calls it synchronously or publishes events asynchronously
- Handles results and errors

## Backwards Compatibility

AuraRouter remains fully functional when deployed standalone:

```bash
# Works without AuraGrid
pip install aurarouter
aurarouter gui        # GUI still works
aurarouter start      # MCP server works
```

AuraGrid integration is purely optional. Remove the `[auragrid]` extra to revert to standalone-only deployment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                      AuraGrid Cluster                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Node A (with AuraRouter MAS)                     │   │
│  │ ┌──────────────────────────────────────────────┐ │   │
│  │ │ AuraRouterMasHost                            │ │   │
│  │ │ ┌──────────────────────────────────────────┐ │ │   │
│  │ │ │ ComputeFabric                            │ │ │   │
│  │ │ │ Models: local_qwen, cloud_gemini, ...    │ │ │   │
│  │ │ └──────────────────────────────────────────┘ │ │   │
│  │ │                                              │ │   │
│  │ │ Services:                                    │ │   │
│  │ │ • RouterService                             │ │   │
│  │ │ • ReasoningService                          │ │   │
│  │ │ • CodingService                             │ │   │
│  │ │ • UnifiedRouterService                      │ │   │
│  │ └──────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Grid Services (auto-discovered):                      │
│  • ServiceRegistry (gossip-elected)                    │
│  • EventPublisher/EventConsumer (WAL-based)            │
│  • ServiceProxy (gRPC + OpenAPI)                       │
└─────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Other Grid Apps  │
│  (Node B, C...)  │
│                  │
│ Create client → │
│ Call service ───┼──> AuraRouter Services (gRPC)
│         or ─────┼──> Publish events (Event topic)
└──────────────────┘
```

## Dynamic Model Registration from Grid Nodes

AuraGrid MAS nodes running fine-tuning or quantization jobs can register newly created GGUF models with AuraRouter via MCP tools, making them immediately available for routing on the next restart.

```python
# On the AuraGrid node after fine-tuning completes:
result = mcp_client.call_tool("register_asset", {
    "model_id": "finetuned-qwen-coding-v2",
    "file_path": "/data/models/finetuned-qwen-v2.gguf",
    "repo": "local",
    "tags": "coding,local,fine-tuned"
})

# Discover all available assets across the node
assets = mcp_client.call_tool("list_assets", {})
```

Registration adds the model to both the physical asset registry and `auraconfig.yaml` with the `llamacpp` provider. The model is **not** auto-added to role chains — orchestration logic (e.g., Black) decides when and where to activate registered models.

---

## Performance Considerations

- **Distributed mode**: Every node runs aurarouter → low latency, high resource usage
- **Model fallback chains**: If primary model fails, grid automatically tries next in list
- **Event-based calls**: Best for bulk/batch operations; minimal grid overhead
- **RPC calls**: Use for interactive/immediate feedback; more grid traffic

## Security Notes

- **API Keys**: Always use environment variables; never commit keys to config files
- **Service Auth**: AuraGrid proxy applies grid-wide auth policies
- **Event Topics**: Published events are visible to all grid nodes; avoid sensitive data in events
- **Model URLs**: Validate Ollama endpoints are on trusted networks

## Future Enhancements

Potential future improvements (documented for reference):

- [ ] Convenience client classes (e.g., `AuraRouterClient` wrapper)
- [ ] Streaming responses for large task output
- [ ] Model provider auto-discovery via service registry
- [x] Metrics/telemetry integration (DAG execution trace with per-node timing and fallback attempts)
- [ ] Request tracing across grid
- [ ] Rate limiting per grid app
- [ ] Result caching for identical requests
- [x] GUI grid administration panels (deployment strategy, cell status)
- [x] Health dashboard with per-model diagnostics (state-aware)
- [x] Singleton instance enforcement (PID + IPC)
- [x] OpenAPI-compatible provider for vLLM, LocalAI, LM Studio, etc.
- [x] Semantic verb synonym resolution for intent classification
- [x] Model capability tags with privacy-aware auto re-routing
- [x] Local GGUF file import (in addition to HuggingFace download)
- [x] Model loading progress state for local GPU models
- [x] MCP asset management tools (`list_assets`, `register_asset`) for programmatic model discovery and registration
