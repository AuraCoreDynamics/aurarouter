# AuraRouter Execution Modes & Architectures

This guide provides a distilled, practical-focused explanation of how AuraRouter operates in different modes (IPE vs. Monologue) and environments (Standalone vs. Grid).

## 1. Core Concepts: IPE vs. AuraMonologue

AuraRouter implements an **Intent -> Plan -> Execute (IPE)** loop as its foundational orchestration pattern. This is extended by **AuraMonologue** for high-complexity reasoning tasks.

### Intent -> Plan -> Execute (IPE)
The standard loop for most tasks:
1.  **Classifier:** A fast local model (the `router` role) determines the task intent (e.g., `DIRECT`, `SIMPLE_CODE`, `COMPLEX_REASONING`) and assigns a complexity score (1-10).
2.  **Planner:** For complex tasks, a reasoning model (the `reasoning` role) generates a sequential list of steps (a DAG).
3.  **Worker:** An execution model (the `coding` role) carries out the plan step-by-step.
4.  **Reviewer (Optional):** A final stage where the output is passed to a `reviewer` role for quality assessment and correction.

### AuraMonologue
A recursive multi-expert reasoning mode triggered for tasks with `COMPLEX_REASONING` intent and complexity >= 8.
*   **Recursive Reasoning:** Uses a Generator -> Critic -> Refiner loop.
*   **Blackboard Pattern:** Experts read and write to a shared reasoning "blackboard" (a Write-Ahead Log or WAL).
*   **Convergence:** The loop continues until the Critic approves the result, similarity converges, or the maximum number of iterations is reached.

---

## 2. Environments: Standalone vs. AuraGrid

AuraRouter is designed to be "Grid-native but Standalone-capable." Standalone mode is the architectural foundation upon which Grid features are layered.

### Standalone Mode
*   **Operation:** Operates independently using the local `auraconfig.yaml`.
*   **Routing:** Driven by manual/static fallback chains defined in the `roles` and `models` sections of the config.
*   **Execution:** `ComputeFabric` iterates through model chains locally or via direct cloud API calls.
*   **Monologue in Standalone:** Performs the Generator/Critic/Refiner loop locally without external anchor retrieval or scoring.

### AuraGrid Mode
*   **Operation:** Deployed as a **Managed Application Service (MAS)** on AuraGrid.
*   **Shared Reasoning WAL:** Monologue uses a Sharded Reasoning WAL via AuraXLM as a shared blackboard across the grid.
*   **MAS-Score-Gated Node Idling:** Uses MAS scoring (via AuraXLM) to optimize compute by only activating relevant expert nodes on the grid.
*   **Sovereignty:** Enforces local-only execution or blocks unsafe requests based on grid-wide policies.
*   **Discovery:** Models and services are auto-discovered and synced across the grid cell.

---

## 3. Execution Priority & Fallback

Execution priority is strictly enforced:
**`Monologue > Speculative > Standard`**

1.  **Monologue Path:** Attempted first if enabled and triggers are met (Complexity >= 8).
2.  **Speculative Path:** Attempted if Monologue is disabled/skipped but `speculative_decoding` is enabled and complexity >= 7. Uses a drafter/verifier pattern for faster, verified output.
3.  **Standard Path:** The fallback for all tasks. Uses the standard IPE loop or direct single-model execution.

---

## 4. Implementation Reference

*   **`src/aurarouter/routing.py`**: The main orchestration layer. Contains the IPE logic and dispatchers for Monologue and Speculative modes.
*   **`src/aurarouter/fabric.py`**: The `ComputeFabric` class handles low-level execution across model chains, sovereignty gating, and RAG enrichment.
*   **`src/aurarouter/monologue.py`**: Implements the `MonologueOrchestrator`. It dynamically detects if it should use Grid-enabled features (like anchors) or fall back to local-only reasoning.
*   **`src/aurarouter/mcp_tools.py`**: Exposes `route_task` (standard), `monologue_execute` (explicit), and `speculative_execute` (explicit) as MCP tools.
