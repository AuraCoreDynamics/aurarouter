"""Tests for the DAGVisualizer widget."""

import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from aurarouter.gui.dag_visualizer import DAGVisualizer
from aurarouter.gui.execution_trace import NodeStatus

# Ensure a QApplication exists for widget tests.
_app = QApplication.instance() or QApplication([])


class TestDAGVisualizer:
    def _make(self) -> DAGVisualizer:
        return DAGVisualizer()

    def test_initial_state(self):
        v = self._make()
        assert v._expanded is False
        assert v._summary_label.text() == "Execution trace"
        assert len(v._trace.nodes) == 0

    def test_reset(self):
        v = self._make()
        v.add_node({
            "id": "a",
            "label": "A",
            "role": "router",
            "status": "running",
        })
        assert len(v._trace.nodes) == 1
        v.reset()
        assert len(v._trace.nodes) == 0
        assert v._summary_label.text() == "Execution trace"

    def test_add_node(self):
        v = self._make()
        v.add_node({
            "id": "classify-0",
            "label": "Classify",
            "role": "router",
            "status": "running",
            "parent_ids": [],
        })
        assert "classify-0" in v._trace.nodes
        node = v._trace.nodes["classify-0"]
        assert node.status == NodeStatus.RUNNING
        assert node.role == "router"

    def test_update_node(self):
        v = self._make()
        v.add_node({
            "id": "classify-0",
            "label": "Classify",
            "role": "router",
            "status": "running",
        })
        v.update_node("classify-0", {"status": "success", "model_id": "gpt-4"})
        node = v._trace.nodes["classify-0"]
        assert node.status == NodeStatus.SUCCESS
        assert node.model_id == "gpt-4"

    def test_update_nonexistent_noop(self):
        v = self._make()
        v.update_node("missing", {"status": "failed"})  # no error

    def test_on_model_tried_records_attempt(self):
        v = self._make()
        v.add_node({
            "id": "exec-0",
            "label": "Execute",
            "role": "coding",
            "status": "running",
        })
        v.on_model_tried("coding", "model-a", False, 0.5)
        v.on_model_tried("coding", "model-b", True, 1.2)
        node = v._trace.nodes["exec-0"]
        assert len(node.attempts) == 2
        assert node.model_id == "model-b"
        assert node.status == NodeStatus.SUCCESS

    def test_on_intent_simple_skips_reasoning(self):
        v = self._make()
        v.add_node({
            "id": "classify-0",
            "label": "Classify",
            "role": "router",
            "status": "success",
        })
        v.add_node({
            "id": "plan-0",
            "label": "Plan",
            "role": "reasoning",
            "status": "pending",
            "parent_ids": ["classify-0"],
        })
        v.on_intent_detected("SIMPLE_CODE")
        assert v._trace.nodes["plan-0"].status == NodeStatus.SKIPPED

    def test_toggle_expand_collapse(self):
        v = self._make()
        assert v._expanded is False
        v._toggle_expanded()
        assert v._expanded is True
        assert not v._canvas.isHidden()
        v._toggle_expanded()
        assert v._expanded is False
        assert v._canvas.isHidden()

    def test_summary_updates(self):
        v = self._make()
        v.add_node({
            "id": "classify-0",
            "label": "Classify",
            "role": "router",
            "status": "success",
            "elapsed_s": 0.3,
        })
        v.add_node({
            "id": "execute-0",
            "label": "Execute",
            "role": "coding",
            "status": "success",
            "parent_ids": ["classify-0"],
            "elapsed_s": 1.0,
        })
        text = v._summary_label.text()
        assert "Classify" in text
        assert "Execute" in text
