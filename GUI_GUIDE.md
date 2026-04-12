# AuraRouter GUI Guide (v0.5.5)

This guide provides an overview of the AuraRouter desktop application. The GUI is designed for transparency, giving you full visibility into how your AI tasks are classified, routed, and executed across local and cloud resources.

## Launch

```bash
# Main command
aurarouter gui

# With a specific configuration file
aurarouter gui --config /path/to/auraconfig.yaml
```

---

## 1. Workspace Panel (Ctrl+1)

The Workspace is where you interact with models. It supports two modes of operation:

### Single-Shot Mode (Default)
Type a task, click **Execute**, and see the result. This is ideal for quick questions or code generation tasks.

### Multi-Turn Conversation (Sessions)
Toggle **Session** mode to enable a chat-like interface. 
- **Persistence:** AuraRouter manages conversation history, automatically summarizing ("gisting") older messages to keep the context window efficient.
- **Token Pressure Gauge:** A real-time meter shows how much of the model's memory is currently in use. When it turns red, the system will soon condense the history to save space.

### Intelligent Transparency
- **Routing Insight Pill:** Every response from the assistant includes a small "pill" showing exactly why a model was chosen (e.g., `[Local · SIMPLE_CODE · $0.02 saved]`). Click the pill for a detailed breakdown of the routing decision.
- **Progressive Output:** Watch responses stream in real-time. If a "Review Loop" is triggered, you'll see the system critique its own work and apply corrections before showing you the final answer.
- **Verified Responses:** Look for the green shield icon next to responses. This indicates the answer was "verified" by a more powerful model during a speculative decoding run.

---

## 2. Routing Panel (Ctrl+2)

The Routing panel is the "brain" of the application. It shows the logic used to dispatch your prompts.

### Routing Pipeline
A visual flow diagram at the top shows the four stages every prompt goes through:
1. **Pre-filter:** Fast check for task complexity.
2. **Intent Classifier:** Determines if the task is coding, reasoning, or a direct question.
3. **Sovereignty Gate:** Checks for sensitive data (PII) to enforce local-only routing.
4. **Triage Router:** Maps the complexity score to the best available role (e.g., "coding" or "reasoning").

### Route Simulator
Paste a prompt into the simulator to watch it move through the pipeline step-by-step. This is a "dry-run" that helps you debug your configuration. If you like the result, you can click **Promote to Rule** to save that routing logic permanently.

### Intent Registry
Manage how different types of tasks are mapped to model roles. You can add custom intents for specific domains like "legal_analysis" or "image_processing."

---

## 3. Models Panel (Ctrl+3)

Manage your library of local and cloud models.
- **Card View:** Each model shows its provider (Ollama, Anthropic, etc.), health status, and active roles.
- **HuggingFace Integration:** Download and install GGUF models directly from the UI.
- **Provider Catalog:** Discover and manage external MCP providers for cloud access.

---

## 4. Monitor Panel (Ctrl+4)

The Monitor panel provides deep observability into your AI usage.

- **ROI & Telemetry:** View an interactive sparkline showing your cumulative savings. Hover over the graph to see daily cost avoidance. You can **Export a Savings Report** (CSV/PDF) to share these stats with your team.
- **Speculative Decoding:** Track the "acceptance rate" of your fast drafter models. See how often the verifier had to step in to correct a draft.
- **Monologue Reasoning:** View full "reasoning traces" for complex tasks. See the step-by-step conversation between the Generator, Critic, and Refiner experts.
- **Model Performance:** A heatmap showing success rates for each model across different complexity levels (1-10).

---

## 5. Settings Panel (Ctrl+5)

Configure advanced subsystems through collapsible sections:
- **Speculative & Monologue:** Fine-tune the thresholds for when these advanced reasoning modes trigger.
- **Sovereignty:** Manage custom PII patterns and enforcement rules.
- **RAG Enrichment:** Configure how AuraRouter retrieves external knowledge (via AuraXLM) before executing a task.
- **Persona Alignment:** If you used the Onboarding Wizard, your settings will show icons (⚡/🔒/🔬) indicating which values were preset by your chosen persona.

---

## Onboarding Wizard

On your first launch, the wizard helps you pick a **Persona** to auto-configure the system:
- **Performance First (⚡):** Optimizes for speed using speculative decoding and RAG.
- **Privacy First (🔒):** Strict local-only routing and sovereignty enforcement.
- **Researcher (🔬):** Enables all reasoning traces and deep monitor tabs for power users.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Return` | Execute task / Send message |
| `Ctrl+N` | New prompt / New session |
| `Escape` | Cancel running execution |
| `Ctrl+1-6` | Switch panels |
| `Ctrl+,` | Open Settings |
| `F1` | Open Help |

## Navigation Tips
The GUI features **Cross-Panel Navigation**. You can click on a model name in the Monitor or Routing panels to jump directly to its configuration in the Models panel.
