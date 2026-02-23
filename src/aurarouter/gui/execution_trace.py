"""Data model for DAG-based execution tracing.

Tracks every node in the intent -> plan -> execute pipeline as a directed
acyclic graph, including fallback attempts per node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ModelAttempt:
    """A single model attempt within a trace node."""

    model_id: str
    success: bool
    elapsed_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


@dataclass
class TraceNode:
    """A single node in the execution DAG."""

    id: str                    # "classify-0", "plan-0", "step-2"
    label: str                 # "Classify Intent", "Step 3: Generate API"
    role: str                  # "router", "reasoning", "coding"
    status: NodeStatus = NodeStatus.PENDING
    model_id: str | None = None
    elapsed_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    result_preview: str = ""
    parent_ids: list[str] = field(default_factory=list)
    attempts: list[ModelAttempt] = field(default_factory=list)


@dataclass
class ExecutionTrace:
    """Container for all trace nodes forming the execution DAG."""

    nodes: dict[str, TraceNode] = field(default_factory=dict)

    def add_node(self, node: TraceNode) -> None:
        self.nodes[node.id] = node

    def set_status(self, node_id: str, status: NodeStatus) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].status = status

    def get_roots(self) -> list[TraceNode]:
        return [n for n in self.nodes.values() if not n.parent_ids]

    def get_children(self, node_id: str) -> list[TraceNode]:
        return [n for n in self.nodes.values() if node_id in n.parent_ids]

    def total_elapsed(self) -> float:
        return sum(n.elapsed_s for n in self.nodes.values())

    def summary(self) -> str:
        """One-line summary for the collapsed DAG header."""
        if not self.nodes:
            return ""

        parts: list[str] = []
        visited: set[str] = set()
        step_ids_seen = False

        # Walk breadth-first from roots.
        queue = [n.id for n in self.get_roots()]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)

            node = self.nodes.get(nid)
            if node is None:
                continue

            if node.role == "router":
                parts.append("Classify")
            elif node.role == "reasoning":
                parts.append("Plan")
            elif node.id.startswith("step-") and not step_ids_seen:
                step_ids_seen = True
                count = sum(
                    1 for n in self.nodes.values() if n.id.startswith("step-")
                )
                if count > 1:
                    parts.append(f"Steps 1\u2013{count}")
                else:
                    parts.append("Step 1")
            elif node.id.startswith("step-"):
                pass  # already summarized
            elif node.id.startswith("execute-"):
                parts.append("Execute")
            else:
                parts.append(node.label)

            for child in self.get_children(nid):
                queue.append(child.id)

        total = self.total_elapsed()
        return " \u2192 ".join(parts) + (f" ({total:.1f}s)" if total > 0 else "")
