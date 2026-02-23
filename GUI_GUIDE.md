# AuraRouter GUI Guide

Comprehensive guide to the AuraRouter desktop GUI application. The GUI provides full administration of AuraRouter: environment management, service lifecycle controls, task execution with routing visualization, model management, and configuration editing.

## Launch

```bash
# Via main CLI
aurarouter gui

# With explicit config
aurarouter gui --config /path/to/auraconfig.yaml

# With explicit environment
aurarouter gui --environment auragrid

# Standalone entry point
aurarouter-gui
```

### Singleton Enforcement

Only one AuraRouter instance can run at a time. On launch, the GUI checks for an existing instance via a PID file (`~/.auracore/aurarouter/aurarouter.pid`) and a platform-specific lock (Windows mutex / Unix socket). If another instance is detected:

1. A dialog appears: "AuraRouter is already running (PID X). Connect to existing instance?"
2. If **yes**: the GUI connects to the running instance via IPC and proxies operations through it.
3. If **no**: the GUI exits.

This prevents port conflicts and resource contention when AuraRouter is embedded in another application (e.g., AuraGrid, sprint-snitch).

---

## Environment Selection

AuraRouter supports two deployment environments. The GUI adapts its panels, controls, and available features based on the active environment.

### Local (Default)

- Configuration is read from and saved to the local `auraconfig.yaml` file.
- Models are stored locally in `~/.auracore/models/`.
- The MCP server runs as a local subprocess.
- Config changes affect only this machine.

### AuraGrid

Available when `pip install aurarouter[auragrid]` is installed.

- Configuration merges environment variables, AuraGrid manifest metadata, and the local YAML file.
- Models include both local files and grid-distributed models.
- Service lifecycle is managed through AuraGrid's MAS (Managed Application Service).
- **Config changes propagate to all nodes on the cell** -- the GUI warns before saving.
- Additional tabs appear: **Deployment** and **Cell Status**.

### Switching at Runtime

The environment selector dropdown is in the toolbar at the top of the window. Switching environments:

1. Prompts for confirmation if a service is running (switching stops the running service).
2. Disposes the old environment context (terminates subprocesses, cancels background tasks).
3. Creates the new environment context.
4. Rebuilds all environment-aware panels (Models, Configuration, extra tabs).

---

## Service Controls

The service toolbar provides lifecycle management for the AuraRouter service.

### States

| State | Indicator | Description |
|-------|-----------|-------------|
| Stopped | Red | Service is not running |
| Starting | Gray | Service is launching |
| Loading Model | Blue | A local GPU model is loading into memory (indeterminate progress bar shown) |
| Running | Green | Service is active and accepting requests |
| Pausing | Gray | Service is transitioning to paused |
| Paused | Yellow | Service is suspended (will resume on request) |
| Stopping | Gray | Service is shutting down |
| Error | Red | Service encountered an error |

### Controls

| Button | Available When | Action |
|--------|---------------|--------|
| Start | Stopped, Error | Launches the MCP server subprocess (Local) or MAS startup (AuraGrid) |
| Pause | Running | Stops accepting new requests; finishes in-flight work |
| Resume | Paused | Restarts the service from paused state |
| Stop | Running, Paused | Gracefully shuts down the service |
| Check | Always | Runs a health check against all configured providers |

### Local Service Behavior

- **Start** spawns a `python -m aurarouter` subprocess with the current config path. If the configuration includes local GPU models (e.g., `llamacpp`, `llamacpp-server`), the service transitions through a **Loading Model** state while the model loads into memory, showing an indeterminate progress bar in the toolbar. Once the model is ready, the state transitions to **Running**.
- **Stop** sends a termination signal (SIGTERM on Unix, terminate on Windows) and waits up to 5 seconds.
- **Pause** terminates the subprocess and marks the state as paused; **Resume** restarts it.
- The GUI monitors the subprocess PID and transitions to Error state if it exits unexpectedly.

### AuraGrid Service Behavior

- **Start** calls `LifecycleCallbacks.startup()` which initializes the compute fabric and registered services.
- **Stop** calls `LifecycleCallbacks.shutdown()`.
- **Pause** sets the health flag to false, causing the grid load balancer to stop routing requests to this node.

---

## Execute Tab

The main task execution interface.

### Task Input

- **Recent Tasks** dropdown: Select from the last 20 executed tasks to restore the prompt and view previous results.
- **Task Description**: Free-text area describing what you need. AuraRouter is content-agnostic -- tasks can be code generation, summarization, analysis, Q&A, or any other prompt-based work.
- **Context (optional)**: Paste context text directly or attach files.
- **Output Format**: Select the desired output format (text, markdown, python, etc.).

### Document Upload

The context input supports file attachments:

- Click **Attach Files...** to browse for supported file types.
- Attached files appear as removable chips showing the filename.
- Supported types: `.txt`, `.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`, `.csv`, `.xml`, `.html`, `.css`, `.go`, `.rs`, `.java`, `.cpp`, `.c`, `.h`, `.sh`, `.sql`, and more.
- A size estimate (in approximate tokens) is displayed when files are attached.
- File contents are concatenated into the context sent to the model.

### DAG Execution Visualization

Below the task input, a dynamic DAG (directed acyclic graph) visualizer shows the execution trace in real-time:

- **Collapsed by default** — a single-line summary (e.g., `Classify > Execute (1.3s)` or `Classify > Plan (3 steps) > Steps 1-3 (4.2s)`)
- **Click to expand** — reveals the full DAG drawn as a left-to-right graph with colored nodes and directed edges

Each node in the DAG represents an execution stage:
- **Label**: Stage name (e.g., "Classify Intent", "Step 3: Generate API")
- **Role**: Which role handled it (`router`, `reasoning`, `coding`, etc.)
- **Model**: Which model succeeded
- **Status color**: Gray (pending), Blue (running), Green (success), Red (failed)

**Click a node** to open a detail dialog showing:
- Role and model used
- All fallback attempts (with success/fail status and elapsed time per attempt)
- Token counts (input/output)
- Result preview (first ~200 characters)
- Error messages (if failed)

For **Direct** (simple) tasks, the DAG shows: `Classify -> Execute`.
For **Multi-Step** tasks: `Classify -> Plan -> Step 1, Step 2, ...` with edges from the plan node to each step.

### Output

Generated results appear in the output panel with a monospaced font. For multi-step tasks, each step's output is appended as it completes.

---

## Models Tab

Manages locally downloaded GGUF model files.

### Local Models

- **Table**: Shows filename, size (MB), source repository, and download date.
- **Refresh**: Rescans the model storage directory.
- **Download from HuggingFace...**: Opens a dialog to search and download GGUF models.
- **Import Local File...**: Browse for a `.gguf` file already on disk. The file stays in-place (not copied) and is registered in the model storage with `repo="local-import"`.
- **Remove Selected**: Deletes the selected model file from disk.
- **Storage Info**: Displays the storage directory path, model count, and total disk usage.

### Grid Models (AuraGrid Only)

When the AuraGrid environment is active, a second section shows models distributed across the grid:

- **Table**: Lists model IDs available on the grid.
- **Refresh Grid**: Fetches the current model list from `GridModelStorage`.

---

## Configuration Tab

Full CRUD editor for `auraconfig.yaml`.

### Models Section

- **Table**: All configured models with their ID, provider type, endpoint/model path, and **tags**.
- **Add**: Opens the model dialog to configure a new model (provider, endpoint, API key, parameters, tags, connection test).
- **Edit**: Modify an existing model's configuration.
- **Remove**: Delete a model from config (with confirmation).

**Model Tags**: Each model can have comma-separated capability tags (e.g., `private, fast, coding`). Tags are used for privacy-aware routing -- models tagged `private` are preferred when prompts contain PII. See [Privacy-Aware Routing](#privacy-aware-routing) below.

### Routing (Fallback Chains) Section

Each role has a priority-ordered fallback chain. The router tries the first model in the chain; if it fails, it falls back to the next. Only one model handles each request.

- **Table**: Shows each role and its fallback order (e.g., `router: local_qwen > cloud_gemini`).
- **Role selector**: Editable dropdown pre-populated with known roles (`router`, `reasoning`, `coding`, `summarization`, `analysis`) plus any custom roles from the config.
- **Append**: Add a model to a role's chain.
- **Up/Down**: Reorder models in the chain (affects fallback priority).
- **Remove from Chain**: Remove the last model from a chain.
- **Delete Role**: Remove an entire role.
- **Missing roles warning**: A red label appears if any required role (`router`, `reasoning`, `coding`) is not configured. These roles are required for the routing engine to function.

### Semantic Verbs Section

Below the routing table, the **Semantic Verbs** section maps synonyms to canonical role names. This allows the intent classifier to return synonyms (e.g., "programming") which are automatically normalized to the correct role (e.g., "coding") before routing.

Built-in verbs:

| Role | Required | Synonyms |
|------|----------|----------|
| `router` | Yes | classifier, triage, intent |
| `reasoning` | Yes | planner, architect, planning |
| `coding` | Yes | code generation, programming, developer |
| `summarization` | No | summarize, tldr, digest |
| `analysis` | No | analyze, evaluate, assess |

Custom synonyms can be added via the config file under `semantic_verbs` or through the GUI.

### YAML Preview

- Live preview of the current configuration as YAML.
- **Copy to Clipboard** button for easy sharing.

### Save / Revert

- **Save**: Persists all changes to `auraconfig.yaml`. In AuraGrid mode, shows a warning that changes propagate to all nodes on the cell.
- **Revert**: Discards unsaved changes and reloads from disk.
- An "Unsaved changes" indicator appears when edits haven't been saved.

### AuraGrid Warnings

When the AuraGrid environment is active, a yellow banner appears at the top of the Configuration tab:

> "Changes will propagate to all nodes on this cell."

A confirmation dialog also appears before saving.

---

## AuraGrid Panels (AuraGrid Only)

These tabs appear only when the AuraGrid environment is active.

### Deployment Tab

- **Model Deployment Strategy**: Table of grid models with current/desired replica counts and a spinner to adjust desired replicas.
- **Cell Resources**: Shows discovered Ollama endpoints and their availability.
- **Apply Strategy**: Sends deployment changes to the AuraGrid orchestration API.
- **Refresh**: Re-fetches deployment and resource information.

### Cell Status Tab

- **Cell Nodes**: Table of nodes in the cell with ID, address, health status, loaded models, and last-seen timestamp.
- **Event Log**: Recent events from the AuraGrid EventBridge (routing requests and results).
- **Auto-refresh**: Node status refreshes automatically every 30 seconds.
- **Refresh Now**: Manual refresh button.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Execute the current task |
| `Ctrl+N` | New prompt (clear all inputs and outputs) |
| `Escape` | Cancel a running execution |

---

## Health Dashboard

Click the **Health** label in the toolbar (underlined, clickable) to open the health dashboard popup:

- **Overall status**: OK or error message.
- **Per-model status**: Green checkmark or red X for each configured model.
- **Check All**: Triggers a full health check and closes the popup.

**State-aware health checks**: Health checks only run provider-level diagnostics when the service is in the **Running** state. When the service is stopped, starting, paused, or in an error state, the health check returns the current service state (e.g., "Service is stopped") without probing individual providers. This prevents misleading "OK" results when the service isn't actually processing requests.

When the service is **Running**, the health check tests:
- **Ollama**: HTTP request to `/api/tags` endpoint.
- **llama.cpp Server**: HTTP request to `/health` endpoint.
- **llama.cpp Embedded**: Verifies the model file exists on disk.
- **OpenAPI**: HTTP request to `{endpoint}/models` endpoint.
- **Cloud providers (Google, Claude)**: Verifies an API key is configured.
- **Local service**: Verifies the MCP server subprocess is alive.

## Privacy-Aware Routing

When the savings/privacy auditor detects PII (personally identifiable information) in a prompt, AuraRouter automatically skips cloud-bound models that are not tagged `private` and continues down the fallback chain to find a suitable model. If a local or `private`-tagged model is available in the chain, the request is routed there instead.

This extends the existing fallback chain -- no separate routing logic is needed. The skip condition is applied alongside the existing budget-based skipping.

To enable this behavior:
1. Tag your local/trusted models with `private` in their config (via the model dialog or YAML).
2. Ensure the role's fallback chain includes both cloud and local models.

Example: If `coding: [cloud_gemini, local_qwen]` and `local_qwen` has `tags: [private]`, a prompt with PII will skip `cloud_gemini` and route to `local_qwen`.

---

## Prompt History

The last 20 executed tasks and their results are persisted to:

```
~/.auracore/aurarouter/history.json
```

- Select a previous task from the **Recent Tasks** dropdown to restore the prompt and view its result.
- History is updated automatically after each successful execution.
- The most recent task appears first in the dropdown.

---

## Troubleshooting

### GUI Won't Launch

- Ensure PySide6 is installed: `pip install PySide6`
- On headless servers, PySide6 requires a display server. Use `aurarouter` (MCP server mode) instead.

### Service Won't Start

- Check that `auraconfig.yaml` exists and is valid YAML.
- Verify that the `aurarouter` module is importable: `python -m aurarouter --help`
- Check the status bar for error messages.

### Health Check Fails

- **Ollama**: Ensure Ollama is running (`ollama serve`) and the endpoint is correct.
- **Cloud providers**: Verify API keys are set (via config or environment variables).
- **llama.cpp**: Ensure the model file path is valid and the file exists.

### Environment Switch Fails

- Switching to AuraGrid requires `pip install aurarouter[auragrid]`.
- If the switch fails silently, check that the AuraGrid SDK is importable: `python -c "import aurarouter.auragrid"`

### Config Changes Not Applying

- Click **Save** in the Configuration tab (unsaved changes are not applied).
- After saving, the ComputeFabric is automatically refreshed.
- In AuraGrid mode, config changes may take a few seconds to propagate to other nodes.
