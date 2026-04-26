# AuraRouter + AuraGrid Integration Guide

AuraRouter can be deployed as a **Managed Application Service (MAS)** on AuraGrid, enabling other grid applications to access routing, reasoning, and task execution services. AuraRouter is content-agnostic -- it routes any prompt-based task (code generation, summarization, analysis, Q&A, etc.) across local and cloud models. This guide covers deployment, configuration, and usage patterns.

## Overview

AuraRouter operates as a "Grid-native but Standalone-capable" compute fabric. See [EXECUTION_MODES.md](docs/EXECUTION_MODES.md) for a detailed architectural comparison of IPE vs. Monologue and Standalone vs. Grid operation.

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
    provider: ollama          # or: llamacpp-server, llamacpp, openapi
    endpoint: http://...      # If applicable
    model_name: ...
    api_key: ...              # Set via env var for security
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

# Fault tolerance — circuit breaker per provider
resilience:
  failure_threshold: 5        # consecutive failures before circuit opens (default: 5)
  reset_timeout: 60.0         # seconds before open circuit attempts half-open probe (default: 60.0)

# Model telemetry polling (RuntimeModelRegistry)
telemetry:
  poll_interval: 15.0         # background poll interval in seconds (default: 15.0)
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

## Fault Tolerance

AuraRouter applies per-provider circuit breakers in the routing loop for all operating modes.

### Circuit Breaker States

```
closed (normal) ──[N consecutive failures]──> open (tripped)
  ^                                              |
  |                                     [reset_timeout elapsed]
  |                                              |
  └──[probe success]── half_open (probing) <─────┘
                              |
                      [probe failure]
                              |
                           (open)
```

- **closed**: All requests pass through normally.
- **open**: Requests to this provider are skipped immediately. After `reset_timeout` seconds the circuit moves to `half_open`.
- **half_open**: A single probe request is allowed. Success closes the circuit; failure re-opens it.

### Last-Resort Probe

When **every** provider in a role's fallback chain has an open circuit breaker, AuraRouter does not immediately return `None`. Instead, it selects the provider whose circuit has been open the longest (closest to `reset_timeout`) and attempts one probe. If the probe succeeds the circuit closes and the response is returned; if it fails, `None` is returned as normal.

This behaviour activates only when all skips in the chain were circuit-breaker skips. If at least one provider was actually attempted (and failed with a real error), the last-resort probe does not activate.

### Configuration

```yaml
resilience:
  failure_threshold: 5    # default: 5 — consecutive failures to trip the breaker
  reset_timeout: 60.0     # default: 60.0 s — how long to wait before half-open probe
```

## Model Registry

`RuntimeModelRegistry` polls all configured providers in the background and maintains a live view of model availability (state, VRAM, load). It is started automatically by `LifecycleCallbacks.startup()` in **every** operating mode — standalone, GUI, and AuraGrid MAS.

The registry is the data source for the `AuctionListener`'s bid evaluation when running on AuraGrid.

```yaml
telemetry:
  poll_interval: 15.0     # default: 15.0 s — how often providers are polled
```

The registry never blocks startup — if it fails to initialise, AuraRouter logs a warning and continues routing normally without live telemetry.

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

# Test OpenAPI provider
python -c "from aurarouter.providers import get_provider; p = get_provider('openapi', {...}); print(p.generate('test'))"
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
# For OpenAPI-compatible providers
export AURAROUTER_MODELS__MY_MODEL__API_KEY=$MY_API_KEY

# Verify key is set (without printing it)
echo ${AURAROUTER_MODELS__MY_MODEL__API_KEY:+SET}
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
│  │ │ │ LifecycleCallbacks                       │ │ │   │
│  │ │ │  ├─ ComputeFabric                        │ │ │   │
│  │ │ │  │   Models: local_qwen, cloud_gemini    │ │ │   │
│  │ │ │  │   CircuitBreakerRegistry (per model)  │ │ │   │
│  │ │ │  ├─ RuntimeModelRegistry (background)    │ │ │   │
│  │ │ └──────────────────────────────────────────┘ │ │   │
│  │ │                                              │ │   │
│  │ │ AuctionListener (AuraGrid only)              │ │   │
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

## Performance Considerations

- **Distributed mode**: Every node runs aurarouter → low latency, high resource usage
- **Model fallback chains**: Circuit breakers gate each provider. Open circuits are skipped; if all are open, the least-recently-failed provider is probed once before returning `None`.
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
- [x] Metrics/telemetry integration (routing visualizer with per-model timing)
- [x] Circuit breaker per-provider fault isolation with last-resort probe
- [x] RuntimeModelRegistry — live model state aggregation across providers
- [ ] Request tracing across grid
- [ ] Rate limiting per grid app
- [ ] Result caching for identical requests
- [x] GUI grid administration panels (deployment strategy, cell status)
- [x] Health dashboard with per-model diagnostics
