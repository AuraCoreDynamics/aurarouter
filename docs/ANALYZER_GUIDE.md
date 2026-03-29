# Analyzer Developer Guide

This guide covers AuraRouter's route analyzer subsystem: how analyzers work, how to build one, and how custom domain-specific intents flow through the routing pipeline.

## 1. Overview

A **route analyzer** is the component that classifies incoming tasks and decides which role (and therefore which model chain) handles execution. It sits at the top of the routing pipeline, above the ComputeFabric and provider layers.

```
Task Prompt
    |
    v
[Active Analyzer]  -- classifies intent, assigns role
    |
    v
[Intent Registry]  -- resolves intent -> target role
    |
    v
[ComputeFabric]    -- executes through role's model chain
    |
    v
[Provider Layer]   -- Ollama, llama.cpp, OpenAPI, MCP
```

Analyzers are registered as `kind: analyzer` artifacts in the unified catalog (`catalog` section of `auraconfig.yaml`). The active analyzer is set via `system.active_analyzer`.

## 2. Analyzer Types

### `intent_triage` (Built-in)

The built-in `aurarouter-default` analyzer uses intent classification with complexity-based triage routing. It asks a fast router model to classify the task (e.g., `SIMPLE_CODE`, `COMPLEX_REASONING`, `DIRECT`) and assign a complexity score (1-10). Based on these, it routes to the appropriate role chain.

- Simple tasks (complexity 1-3) go straight to execution.
- Moderate tasks (4-7) go through Plan then Execute.
- Complex tasks (8-10) go through Plan, Execute, Review, and optionally Correct.

This analyzer is auto-registered on server startup if not already present.

### `moe_ranking` (AuraXLM)

A remote Mixture-of-Experts analyzer provided by AuraXLM. When active, AuraRouter delegates the routing decision to AuraXLM via MCP JSON-RPC. AuraXLM returns a ranked list of models and a recommended role.

The MCP tool interface is defined in `contracts/auraxlm.py`:

```python
ANALYZE_ROUTE_PARAMS = {
    "prompt": {"type": "string", "required": True},
    "intent": {"type": "string", "required": False},
    "candidates": {"type": "array", "items": "ModelMetadata", "required": False},
    "cost_ceiling": {"type": "number", "required": False},
    "latency_ceiling_ms": {"type": "number", "required": False},
    "top_n": {"type": "integer", "required": False, "default": 3},
}
```

### Custom Domain Analyzers

Any external system can act as an analyzer by implementing the MCP endpoint contract (see Section 6) and declaring custom intents via `role_bindings`. For example, a SAR processing system might declare intents like `sar_coherent_change`, `sar_detection`, and `sar_geolocation` that map to specialized roles.

## 3. Registration

Analyzers are registered in the unified catalog via `catalog_set()` or the `aurarouter catalog register` CLI command.

### Via `auraconfig.yaml`

```yaml
catalog:
  my-sar-analyzer:
    kind: analyzer
    display_name: SAR Processing Analyzer
    description: Domain-specific analyzer for SAR imagery workflows
    analyzer_kind: intent_triage
    capabilities: [sar, geolocation, detection]
    role_bindings:
      sar_coherent_change: reasoning
      sar_detection: coding
      sar_geolocation: reasoning
      explain_result: reasoning
      chat: coding
```

### Via CLI

```bash
aurarouter catalog register my-sar-analyzer \
  --kind analyzer \
  --display-name "SAR Processing Analyzer"
```

### Via MCP Tool

```json
{
  "tool": "aurarouter.catalog.register",
  "arguments": {
    "artifact_id": "my-sar-analyzer",
    "kind": "analyzer",
    "display_name": "SAR Processing Analyzer",
    "analyzer_kind": "intent_triage",
    "role_bindings": {
      "sar_coherent_change": "reasoning",
      "sar_detection": "coding"
    }
  }
}
```

### Activating an Analyzer

```bash
aurarouter analyzer set my-sar-analyzer
```

Or via MCP: `aurarouter.analyzer.set_active(analyzer_id="my-sar-analyzer")`

## 4. Spec Schema Reference

Analyzer artifacts use the following spec fields (merged at the top level in YAML alongside the common `kind`, `display_name`, etc. fields):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `analyzer_kind` | `string` | **Yes** | -- | Analyzer type identifier. Known values: `"intent_triage"`, `"moe_ranking"`. Custom values are allowed. |
| `role_bindings` | `dict[str, str]` | No | `{}` | Maps intent names to target roles. Each key becomes a custom intent; each value must be a configured role name. |
| `mcp_endpoint` | `string` (URL) | No | `null` | MCP JSON-RPC endpoint for remote analyzers. When set, the analyzer is considered remote. |
| `mcp_tool_name` | `string` | No | `null` | The MCP tool name to call on the remote endpoint (e.g., `"auraxlm.analyze_route"`). |
| `capabilities` | `list[str]` | No | `[]` | Declared capabilities for catalog query matching (e.g., `["code", "reasoning", "sar"]`). |
| `description` | `string` | No | `""` | Human-readable description shown in the GUI and CLI. |

### Validation

The `validate_analyzer_spec()` function in `analyzer_schema.py` checks:

- Required fields are present (`analyzer_kind`).
- `role_bindings` keys are valid Python identifiers.
- `role_bindings` values reference configured roles (when `available_roles` is provided).
- `mcp_endpoint` is a well-formed URL if present.
- `capabilities` is a list of strings if present.

Validation is warn-only for backwards compatibility. The result is an `AnalyzerSpecValidation` dataclass:

```python
@dataclass
class AnalyzerSpecValidation:
    valid: bool                        # True if no errors
    warnings: list[str]                # Non-fatal issues
    errors: list[str]                  # Fatal issues (e.g., missing analyzer_kind)
    declared_intents: list[str]        # Intent names extracted from role_bindings
```

## 5. Role Bindings

The `role_bindings` dict is the mechanism by which analyzers declare custom intents. Each key is treated as an intent name, and its value is the target role that handles tasks classified under that intent.

### How It Works

1. When an analyzer is set as active, `build_intent_registry()` reads its `role_bindings` from the catalog.
2. Each key-value pair is converted into an `IntentDefinition` with `priority=10` (higher than built-in priority of 0).
3. These definitions are registered in the `IntentRegistry`, making them available for classification.
4. If a custom intent has the same name as a built-in intent, the higher-priority custom intent wins.

### Example

```yaml
role_bindings:
  sar_coherent_change: reasoning   # New custom intent -> reasoning role
  sar_detection: coding            # New custom intent -> coding role
  SIMPLE_CODE: coding              # Overrides the built-in SIMPLE_CODE
```

### Constraints

- Keys must be valid Python identifiers (letters, digits, underscores; cannot start with a digit).
- Values must reference roles that exist in the `roles` section of `auraconfig.yaml`.
- If a value references a non-existent role, validation produces a warning (not an error).

## 6. MCP Endpoint Contract

Remote analyzers communicate via MCP JSON-RPC. When a remote analyzer is active, `route_task` sends a `tools/call` request to the analyzer's `mcp_endpoint`.

### Request

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "<mcp_tool_name>",
    "arguments": {
      "prompt": "The user's task description",
      "intent": "optional_intent_hint",
      "candidates": [
        {"model_id": "...", "provider": "...", "capabilities": [...]}
      ]
    }
  },
  "id": 1
}
```

### Response

The analyzer should return a JSON-RPC result with routing decisions:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"role\": \"coding\", \"ranked_models\": [\"model-a\", \"model-b\"], \"reasoning\": \"...\"}"
      }
    ]
  },
  "id": 1
}
```

The `text` field contains a JSON string with:

| Field | Type | Description |
|-------|------|-------------|
| `role` | `string` | The role to route to (e.g., `"coding"`, `"reasoning"`) |
| `ranked_models` | `list[str]` | Ordered list of model IDs to try |
| `reasoning` | `string` | Human-readable explanation of the routing decision |

### Failure Handling

If the remote analyzer is unreachable, returns an error, or times out, AuraRouter automatically falls back to the built-in `aurarouter-default` analyzer. This ensures tasks always get processed.

## 7. Intent Lifecycle

Custom intents flow through five stages from declaration to execution:

### Stage 1: Declaration

An analyzer declares intents via `role_bindings` in its catalog spec:

```yaml
role_bindings:
  sar_detection: coding
```

### Stage 2: Registration

When the analyzer is set as active, `build_intent_registry()` calls `register_from_role_bindings()`, which creates an `IntentDefinition` for each entry:

```python
IntentDefinition(
    name="sar_detection",
    description="Intent 'sar_detection' declared by analyzer 'my-sar-analyzer'",
    target_role="coding",
    source="my-sar-analyzer",
    priority=10,
)
```

### Stage 3: Classification

During `route_task`, the router model classifies the user's prompt. The `IntentRegistry.build_classifier_choices()` method generates the prompt listing all available intents (built-in + custom) with their descriptions. The router model returns a JSON object like:

```json
{"intent": "sar_detection", "complexity": 5}
```

Alternatively, the user can force a specific intent via the CLI `--intent` flag or the GUI intent combobox, bypassing classification entirely.

### Stage 4: Role Resolution

The `IntentRegistry.resolve_role()` method maps the classified intent to its target role:

```python
role = registry.resolve_role("sar_detection")  # Returns "coding"
```

### Stage 5: Execution

The `ComputeFabric` executes the task through the resolved role's model chain. If models declare `supported_intents`, the chain is filtered via `filter_chain_by_intent()` to prefer models that explicitly support the classified intent:

```python
filtered = fabric.filter_chain_by_intent(chain, "sar_detection")
```

If no models in the chain declare `supported_intents`, the full chain is used (backwards compatible).

## 8. Routing Advisors

Routing advisors are MCP services that can reorder a role's model chain before execution. They sit between intent classification and model execution.

### Registration

Register a routing advisor programmatically:

```python
fabric.register_routing_advisor(client)
```

Or declare a service in the catalog with the `routing_advisor` capability for auto-registration:

```yaml
catalog:
  my-advisor-service:
    kind: service
    display_name: My Routing Advisor
    capabilities: [routing_advisor]
    endpoint: http://advisor-host:9090
```

### How Advisors Work

During execution, `ComputeFabric.consult_routing_advisors()` queries each registered advisor:

```python
chain = fabric.consult_routing_advisors(role, chain, intent="sar_detection")
```

Advisors with the `chain_reorder` capability receive the role, current chain, and classified intent. They return a reordered chain. If no advisor responds (or all fail), the original chain is used.

### API

| Method | Description |
|--------|-------------|
| `register_routing_advisor(client)` | Register an MCP client as a routing advisor. Idempotent. |
| `unregister_routing_advisor(client_id)` | Remove an advisor by its identifier. |
| `list_routing_advisors()` | Return identifiers of all registered advisors. |
| `consult_routing_advisors(role, chain, intent=None)` | Query advisors for chain reordering. |

## 9. Worked Example: SAR Processing Analyzer

This walkthrough builds a complete analyzer for SAR (Synthetic Aperture Radar) image processing.

### Step 1: Define Your Intents

Decide what domain-specific intents your system needs:

| Intent | Target Role | Description |
|--------|-------------|-------------|
| `sar_coherent_change` | `reasoning` | Multi-step coherent change detection |
| `sar_detection` | `coding` | Target detection in SAR imagery |
| `sar_geolocation` | `reasoning` | Geolocation and coordinate extraction |
| `explain_result` | `reasoning` | Explain processing results |
| `chat` | `coding` | General questions about SAR |

### Step 2: Register the Analyzer

Add to `auraconfig.yaml`:

```yaml
catalog:
  sar-processor:
    kind: analyzer
    display_name: SAR Processing Analyzer
    description: Domain-specific intents for SAR imagery workflows
    analyzer_kind: intent_triage
    capabilities: [sar, geolocation, detection, coherent-change]
    role_bindings:
      sar_coherent_change: reasoning
      sar_detection: coding
      sar_geolocation: reasoning
      explain_result: reasoning
      chat: coding
```

### Step 3: (Optional) Tag Models with Supported Intents

If you want certain models to handle specific intents, declare `supported_intents` on the model artifact:

```yaml
catalog:
  sar-specialist-model:
    kind: model
    display_name: SAR Specialist
    provider: ollama
    supported_intents: [sar_coherent_change, sar_detection, sar_geolocation]
```

When this model is in a role's chain and the classified intent matches, `filter_chain_by_intent()` will prefer it over models that do not declare support.

### Step 4: Activate

```bash
aurarouter analyzer set sar-processor
```

### Step 5: Verify

```bash
# List all intents including the new SAR ones
aurarouter intent list

# Describe a specific intent
aurarouter intent describe sar_detection

# Route a task with explicit intent
aurarouter run "Detect targets in SAR scene" --intent sar_detection
```

### Step 6: (Optional) Use the GUI

In the GUI workspace panel, the intent combobox will show your SAR intents under an "Analyzer: SAR Processing Analyzer" group. Select one to bypass auto-classification.

## 10. Reference Contracts

AuraRouter includes two reference analyzer contracts that serve as templates:

### `contracts/auracode.py`

The AuraCode contract defines intents for code-focused workflows:

```python
AURACODE_INTENTS = {
    "generate_code": "coding",
    "edit_code": "coding",
    "complete_code": "coding",
    "explain_code": "reasoning",
    "review": "reasoning",
    "chat": "reasoning",
    "plan": "reasoning",
}

def create_auracode_analyzer_spec() -> dict:
    """Return the canonical analyzer spec for an AuraCode-compatible analyzer."""
    return {
        "analyzer_kind": "intent_triage",
        "role_bindings": AURACODE_INTENTS,
        "capabilities": ["code", "reasoning", "review", "planning"],
    }
```

### `contracts/auraxlm.py`

The AuraXLM contract defines the MoE ranking analyzer interface:

```python
AURAXLM_ANALYZER_SPEC = {
    "analyzer_kind": "moe_ranking",
    "capabilities": ["code", "reasoning", "review", "planning", "domain-expert"],
    "mcp_tool_name": "auraxlm.analyze_route",
}
```

It also defines `ANALYZE_ROUTE_PARAMS` (the expected parameters for the `auraxlm.analyze_route` tool) and `ANALYZE_ROUTE_RESPONSE` (the expected response schema).

---

See also:
- [README.md](../README.md) for the intent classification overview
- [DEPLOYMENT.md](../DEPLOYMENT.md) for configuration reference
- [BACKEND_PLUGINS.md](../BACKEND_PLUGINS.md) for backend vs. analyzer plugin comparison
