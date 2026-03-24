# AuraRouter GUI Guide (v0.3.1)

Comprehensive guide to the AuraRouter desktop GUI application. The v0.3.1 redesign replaces the legacy tab-based interface with a sidebar-driven layout consisting of six main sections, a top bar with service controls, and a status bar.

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

---

## Application Shell

The main window is structured as:

- **Top Bar (48px)**: Title, loading progress bar, environment selector dropdown, status badge, play/pause/stop buttons.
- **Sidebar Navigation**: Collapsible sidebar on the left with icon+label entries for each section. Click or use Ctrl+1 through Ctrl+6 to switch.
- **Content Area**: A stacked widget that lazily instantiates panels on first navigation.
- **Status Bar (24px)**: Service status message, model count, and "F1 for help" hint.

### Sidebar Sections

| # | Icon | Section | Shortcut | Description |
|---|------|---------|----------|-------------|
| 1 | Play | Workspace | Ctrl+1 | Task execution (hero screen) |
| 2 | Diamond | Routing | Ctrl+2 | Visual routing editor |
| 3 | Square | Models | Ctrl+3 | Model management |
| 4 | Circle | Monitor | Ctrl+4 | Observability dashboard |
| 5 | Gear | Settings | Ctrl+5 or Ctrl+, | System configuration |
| 6 | ? | Help | Ctrl+6 or F1 | Contextual help browser |
| 7 | Grid | Grid | -- | AuraGrid panels (only when available) |

---

## Environment Selection

AuraRouter supports two deployment environments. The GUI adapts its panels, controls, and available features based on the active environment.

### Local (Default)

- Configuration is read from and saved to the local `auraconfig.yaml` file.
- Models are stored locally in `~/.auracore/models/`.
- The MCP server runs as a local subprocess.

### AuraGrid

Available when `pip install aurarouter[auragrid]` is installed.

- Configuration merges environment variables, AuraGrid manifest metadata, and the local YAML file.
- Models include both local files and grid-distributed models.
- Service lifecycle is managed through AuraGrid's MAS (Managed Application Service).
- Config changes propagate to all nodes on the cell -- the GUI warns before saving.
- Additional Grid section appears in the sidebar.

### Switching at Runtime

The environment selector dropdown is in the top bar. Switching environments:

1. Prompts for confirmation if a service is running (switching stops the running service).
2. Disposes the old environment context.
3. Creates the new environment context.
4. Rewires the service controller.

---

## Service Controls

The top bar provides lifecycle management for the AuraRouter service.

### States

| State | Badge | Description |
|-------|-------|-------------|
| Stopped | stopped | Service is not running |
| Starting | loading | Service is launching |
| Loading Model | loading | A local GPU model is loading into memory (progress bar shown) |
| Running | running | Service is active and accepting requests |
| Pausing | loading | Service is transitioning to paused |
| Paused | paused | Service is suspended |
| Stopping | loading | Service is shutting down |
| Error | error | Service encountered an error |

### Controls

| Button | Available When | Action |
|--------|---------------|--------|
| Play | Stopped, Error | Launches the MCP server subprocess (Local) or MAS startup (AuraGrid) |
| Pause | Running | Stops accepting new requests; finishes in-flight work |
| Resume | Paused | Restarts the service from paused state |
| Stop | Running, Paused | Gracefully shuts down the service |

---

## 1. Workspace Panel

The hero screen for task execution. Three-column layout:

### Left Column: Task History

- Searchable list of the last 20 executed tasks.
- Click an entry to restore the prompt and view previous results.
- Clear button to reset history.
- History persisted to `~/.auracore/aurarouter/history.json`.

### Center Column: Task Input and Output

- **Task Description**: Free-text area for the prompt.
- **DAG Visualizer**: Always-visible directed acyclic graph showing the execution trace in real time. Nodes colored by status (gray=pending, blue=running, green=success, red=failed). Click a node for detail dialog (role, model, fallback attempts, tokens, result preview).
  - Direct tasks: `Classify -> Execute`
  - Multi-step tasks: `Classify -> Plan -> Step 1, Step 2, ...`
- **Output**: Syntax-highlighted output with monospaced font.

### Right Column: Context and Settings

- File attachment area with chip display and token estimate.
- Output format selector (text, markdown, python, JSON, etc.).
- Execution settings.
- Contextual help tooltips.

### Signals

The workspace panel exposes `execute_requested`, `new_requested`, and `cancel_requested` signals, wired to the shell's keyboard shortcuts (Ctrl+Return, Ctrl+N, Escape).

---

## 2. Routing Panel

Visual flowchart editor for role-to-model fallback chains.

- **Canvas**: Flowchart-style visualization of roles (left) connected to models (right) with directed edges showing fallback order.
- **Right-click context menu**: Add/remove models from chains, reorder priority.
- **Properties sidebar (250px)**: Shows details for the selected role or model node.
- **Triage preview**: Collapsible section showing how a sample task would be routed through the current configuration.
- **Save/Revert**: All changes require explicit Save to persist to `auraconfig.yaml`.

---

## 3. Models Panel

Unified model management replacing the old split between file storage and config editing.

- **Category sidebar (left)**: Filter by provider type, hosting tier, or tag.
- **Model cards (right)**: Scrollable card-based layout showing each model with provider badge, health status, role assignments, and action buttons.
- **Toolbar**: Add model, import local GGUF, download from HuggingFace, refresh, search.
- **Provider catalog integration**: Browse and install external MCP providers.
- **Grid models (AuraGrid)**: Additional section showing grid-distributed models.

---

## 4. Monitor Panel

Unified observability dashboard with shared time-range controls, summary cards, and four sub-panels.

### Sub-tabs (via left mini-nav)

- **Overview**: Summary stat cards (total tokens, spend, privacy events, model health) with sparkline trends.
- **Traffic**: Token traffic per model, provider breakdown, cost projection, request-rate table.
- **Privacy**: Privacy event log with severity filtering, pattern breakdown, timeline.
- **Health**: Per-model health cards with latency, last-check timestamp, and manual re-check buttons.

### Shared Controls

- Time-range selector (1h, 24h, 7d, 30d, custom).
- Auto-refresh toggle with configurable interval.

---

## 5. Settings Panel

System configuration organized into collapsible sections.

- **MCP Tools**: Enable/disable individual MCP tools exposed by the server.
- **Budget**: Configure daily and monthly spend limits, view current usage.
- **Privacy**: PII detection patterns, privacy routing rules, audit settings.
- **YAML Editor**: Raw YAML view with syntax highlighting, copy to clipboard, save/revert.
- **System**: Log level, default timeout, storage paths, theme preferences.

All changes require explicit Save. Unsaved changes indicator shown in the section header.

---

## 6. Help Panel

Searchable, filterable help browser.

- **Left pane (40%)**: Topic list with search bar and category filter dropdown.
- **Right pane (60%)**: Topic detail view with related-topic links (clickable).
- **Onboarding wizard**: Shown on first launch; can be re-triggered via Help section or `restart_onboarding()`.
- **Contextual help**: Help tooltips throughout the GUI link to relevant topics.

---

## Grid Panels (AuraGrid Only)

Appear as a combined panel in the Grid sidebar section when AuraGrid is available.

### Deployment Sub-panel

- Model deployment strategy table with current/desired replica counts.
- Cell resource overview (Ollama endpoints, availability).
- Apply Strategy button to push changes to the grid.

### Status Sub-panel

- Cell node table (ID, address, health, loaded models, last seen).
- Event log from AuraGrid EventBridge.
- Auto-refresh every 30 seconds with manual refresh button.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Return` | Execute the current task |
| `Ctrl+N` | New prompt (clear inputs and outputs) |
| `Escape` | Cancel a running execution |
| `Ctrl+1` | Go to Workspace |
| `Ctrl+2` | Go to Routing |
| `Ctrl+3` | Go to Models |
| `Ctrl+4` | Go to Monitor |
| `Ctrl+5` | Go to Settings |
| `Ctrl+6` | Go to Help |
| `Ctrl+,` | Go to Settings (alternative) |
| `F1` | Go to Help (alternative) |

---

## Provider Catalog Usage

The provider catalog (accessible via Models panel and `aurarouter catalog` CLI) manages both built-in and external providers:

1. **List**: View all known providers with install status and health.
2. **Add**: Register a custom MCP provider by endpoint URL.
3. **Start/Stop**: Launch or terminate external MCP provider servers.
4. **Health**: Check connectivity and response for any provider.
5. **Discover**: Probe a running provider to discover available models and optionally auto-register them in `auraconfig.yaml`.

External providers (e.g., aurarouter-claude, aurarouter-gemini) are connected through `MCPProvider`, which wraps any MCP-compatible server as a standard AuraRouter provider.

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
- **OpenAPI providers**: Verify the endpoint is reachable and API keys are set if needed.
- **llama.cpp**: Ensure the model file path is valid and the file exists.

### Environment Switch Fails

- Switching to AuraGrid requires `pip install aurarouter[auragrid]`.
- If the switch fails silently, check that the AuraGrid SDK is importable: `python -c "import aurarouter.auragrid"`
