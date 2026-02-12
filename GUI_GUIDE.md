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

- **Start** spawns a `python -m aurarouter` subprocess with the current config path.
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

### Routing Pipeline Visualization

Below the plan steps display, the routing visualizer shows the pipeline in real-time:

```
[ Classifier ] --> [ Planner ] --> [ Worker ]
```

Each stage box shows:
- Which model was tried
- Whether it succeeded or failed
- Elapsed time
- If a model failed and a fallback succeeded, the failed attempt is shown with strikethrough

For **Direct** (simple) tasks, the Planner stage is marked as "skipped".

### Output

Generated results appear in the output panel with a monospaced font. For multi-step tasks, each step's output is appended as it completes.

---

## Models Tab

Manages locally downloaded GGUF model files.

### Local Models

- **Table**: Shows filename, size (MB), source repository, and download date.
- **Refresh**: Rescans the model storage directory.
- **Download from HuggingFace...**: Opens a dialog to search and download GGUF models.
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

- **Table**: All configured models with their ID, provider type, and endpoint/model path.
- **Add**: Opens the model dialog to configure a new model (provider, endpoint, API key, parameters, connection test).
- **Edit**: Modify an existing model's configuration.
- **Remove**: Delete a model from config (with confirmation).

### Routing (Role Chains) Section

- **Table**: Shows each role and its model chain (e.g., `router: local_qwen -> cloud_gemini`).
- **Append**: Add a model to a role's chain.
- **Up/Down**: Reorder models in the chain (affects fallback priority).
- **Remove from Chain**: Remove the last model from a chain.
- **Delete Role**: Remove an entire role.

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

The health check tests:
- **Ollama**: HTTP request to `/api/tags` endpoint.
- **llama.cpp Server**: HTTP request to `/health` endpoint.
- **llama.cpp Embedded**: Verifies the model file exists on disk.
- **Cloud providers (Google, Claude)**: Verifies an API key is configured.
- **Local service**: Verifies the MCP server subprocess is alive (when running).

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
