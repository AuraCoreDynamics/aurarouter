"""Tests for GUI review loop and trace integration (TG-B3).

Tests the data model and signal attributes without requiring a running
Qt event loop (avoids pytest-qt dependency).
"""

from aurarouter.gui.execution_trace import (
    ExecutionTrace,
    NodeStatus,
    TraceNode,
)


class TestInferenceWorkerSignals:
    """Verify the InferenceWorker class has the required review signals."""

    def test_review_signal_attributes(self):
        """InferenceWorker has review_started, review_completed, correction_started."""
        from aurarouter.gui.main_window import InferenceWorker

        assert hasattr(InferenceWorker, "review_started")
        assert hasattr(InferenceWorker, "review_completed")
        assert hasattr(InferenceWorker, "correction_started")

    def test_existing_signals_preserved(self):
        """InferenceWorker retains all original signals."""
        from aurarouter.gui.main_window import InferenceWorker

        assert hasattr(InferenceWorker, "intent_detected")
        assert hasattr(InferenceWorker, "plan_generated")
        assert hasattr(InferenceWorker, "step_started")
        assert hasattr(InferenceWorker, "step_completed")
        assert hasattr(InferenceWorker, "model_tried")
        assert hasattr(InferenceWorker, "finished")
        assert hasattr(InferenceWorker, "error")
        assert hasattr(InferenceWorker, "trace_node_added")
        assert hasattr(InferenceWorker, "trace_node_updated")


class TestTraceSummaryWithReview:
    """Tests for ExecutionTrace.summary() with review and correction nodes."""

    def test_summary_simple_with_review_pass(self):
        """Summary includes review verdict for simple execution."""
        trace = ExecutionTrace()
        trace.add_node(TraceNode(
            id="classify-0", label="Classify", role="router",
            status=NodeStatus.SUCCESS,
        ))
        trace.add_node(TraceNode(
            id="execute-0", label="Execute", role="coding",
            status=NodeStatus.SUCCESS, parent_ids=["classify-0"],
        ))
        trace.add_node(TraceNode(
            id="review-1", label="Review #1", role="reviewer",
            status=NodeStatus.SUCCESS, result_preview="PASS",
        ))

        s = trace.summary()
        assert "Classify" in s
        assert "Execute" in s
        assert "Review PASS" in s

    def test_summary_with_correction(self):
        """Summary includes correction info after failed review."""
        trace = ExecutionTrace()
        trace.add_node(TraceNode(
            id="classify-0", label="Classify", role="router",
            status=NodeStatus.SUCCESS,
        ))
        trace.add_node(TraceNode(
            id="execute-0", label="Execute", role="coding",
            status=NodeStatus.SUCCESS, parent_ids=["classify-0"],
        ))
        trace.add_node(TraceNode(
            id="review-1", label="Review #1", role="reviewer",
            status=NodeStatus.FAILED, result_preview="FAIL",
        ))
        trace.add_node(TraceNode(
            id="correction-1-step-0", label="Correct 1.1", role="coding",
            status=NodeStatus.SUCCESS, parent_ids=["review-1"],
        ))
        trace.add_node(TraceNode(
            id="correction-1-step-1", label="Correct 1.2", role="coding",
            status=NodeStatus.SUCCESS, parent_ids=["review-1"],
        ))
        trace.add_node(TraceNode(
            id="review-2", label="Review #2", role="reviewer",
            status=NodeStatus.SUCCESS, result_preview="PASS",
        ))

        s = trace.summary()
        assert "Classify" in s
        assert "Execute" in s
        assert "Correct" in s
        assert "Review PASS" in s

    def test_summary_no_review_nodes(self):
        """Summary without review nodes works as before."""
        trace = ExecutionTrace()
        trace.add_node(TraceNode(
            id="classify-0", label="Classify", role="router",
            status=NodeStatus.SUCCESS,
        ))
        trace.add_node(TraceNode(
            id="execute-0", label="Execute", role="coding",
            status=NodeStatus.SUCCESS, parent_ids=["classify-0"],
        ))

        s = trace.summary()
        assert "Classify" in s
        assert "Execute" in s
        assert "Review" not in s
