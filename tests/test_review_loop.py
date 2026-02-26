"""Tests for review loop business logic (TG-B2)."""

import json
from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import ReviewResult, review_output, generate_correction_plan
from aurarouter.mcp_tools import route_task, generate_code


def _make_review_fabric(
    max_iterations: int = 3,
    reviewer_chain: list[str] | None = None,
) -> ComputeFabric:
    """Build a ComputeFabric with reviewer role configured."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
            "reviewer": reviewer_chain if reviewer_chain is not None else ["m1"],
        },
        "execution": {"max_review_iterations": max_iterations},
    }
    return ComputeFabric(cfg)


# ------------------------------------------------------------------
# review_output
# ------------------------------------------------------------------

class TestReviewOutput:
    def test_pass_verdict(self):
        """review_output returns PASS when reviewer approves."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", return_value=json.dumps({
            "verdict": "PASS",
            "feedback": "Good",
            "correction_hints": [],
        })):
            result = review_output(fabric, "build a widget", "widget code here")
            assert result.verdict == "PASS"
            fabric.execute.assert_called_once()
            assert fabric.execute.call_args[0][0] == "reviewer"

    def test_fail_verdict(self):
        """review_output returns FAIL with feedback."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", return_value=json.dumps({
            "verdict": "FAIL",
            "feedback": "Missing error handling",
            "correction_hints": ["Add try/except"],
        })):
            result = review_output(fabric, "build a widget", "widget code here")
            assert result.verdict == "FAIL"
            assert "error handling" in result.feedback
            assert len(result.correction_hints) == 1

    def test_fail_open_on_error(self):
        """review_output returns PASS if reviewer throws an exception."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", side_effect=Exception("Model unavailable")):
            result = review_output(fabric, "task", "output")
            assert result.verdict == "PASS"

    def test_fail_open_on_empty(self):
        """review_output returns PASS if reviewer returns None."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", return_value=None):
            result = review_output(fabric, "task", "output")
            assert result.verdict == "PASS"

    def test_fail_open_on_invalid_json(self):
        """review_output returns PASS if reviewer returns non-JSON."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", return_value="not json at all"):
            result = review_output(fabric, "task", "output")
            assert result.verdict == "PASS"


# ------------------------------------------------------------------
# generate_correction_plan
# ------------------------------------------------------------------

class TestGenerateCorrectionPlan:
    def test_returns_steps(self):
        """generate_correction_plan returns parsed correction steps."""
        fabric = _make_review_fabric()
        with patch.object(
            fabric, "execute",
            return_value='["Fix error handling", "Add tests"]',
        ):
            review = ReviewResult("FAIL", "Missing error handling", ["Add try/except"])
            steps = generate_correction_plan(fabric, "build widget", "bad output", review)
            assert len(steps) == 2
            assert steps[0] == "Fix error handling"

    def test_fallback_on_failure(self):
        """generate_correction_plan falls back to single step on failure."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", return_value=None):
            review = ReviewResult("FAIL", "Bad output", [])
            steps = generate_correction_plan(fabric, "build widget", "bad output", review)
            assert len(steps) == 1
            assert "Redo" in steps[0]

    def test_fallback_on_exception(self):
        """generate_correction_plan falls back on exception."""
        fabric = _make_review_fabric()
        with patch.object(fabric, "execute", side_effect=Exception("boom")):
            review = ReviewResult("FAIL", "Bad", [])
            steps = generate_correction_plan(fabric, "task", "output", review)
            assert len(steps) == 1
            assert "Redo" in steps[0]


# ------------------------------------------------------------------
# Review loop in route_task
# ------------------------------------------------------------------

class TestRouteTaskReviewLoop:
    def test_skips_review_when_no_reviewer(self):
        """route_task skips review when reviewer chain is empty."""
        fabric = _make_review_fabric(reviewer_chain=[])
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE", "complexity": 3}),
            "result text",
        ]):
            result = route_task(fabric, None, task="simple task")
            assert result == "result text"
            # Only 2 calls: analyze_intent + execute. No reviewer call.
            assert fabric.execute.call_count == 2
            reviewer_calls = [
                c for c in fabric.execute.call_args_list
                if c[0][0] == "reviewer"
            ]
            assert len(reviewer_calls) == 0

    def test_skips_review_when_iterations_zero(self):
        """route_task skips review when max_review_iterations is 0."""
        fabric = _make_review_fabric(max_iterations=0)
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),
            "output",
        ]):
            result = route_task(fabric, None, task="simple task")
            assert result == "output"
            reviewer_calls = [
                c for c in fabric.execute.call_args_list
                if c[0][0] == "reviewer"
            ]
            assert len(reviewer_calls) == 0

    def test_review_pass_first_time(self):
        """route_task with review PASS on first iteration returns output."""
        fabric = _make_review_fabric(max_iterations=3)
        with patch.object(fabric, "execute", side_effect=[
            # analyze_intent
            json.dumps({"intent": "SIMPLE_CODE"}),
            # execute task
            "good output",
            # review_output (reviewer role) -> PASS
            json.dumps({"verdict": "PASS", "feedback": "Looks good", "correction_hints": []}),
        ]):
            result = route_task(fabric, None, task="simple task")
            assert result == "good output"

    def test_review_fail_correct_pass(self):
        """route_task: FAIL on first review, correction, PASS on second."""
        fabric = _make_review_fabric(max_iterations=3)
        with patch.object(fabric, "execute", side_effect=[
            # analyze_intent
            json.dumps({"intent": "SIMPLE_CODE"}),
            # execute task
            "bad output",
            # review_output #1 -> FAIL
            json.dumps({
                "verdict": "FAIL",
                "feedback": "Missing error handling",
                "correction_hints": ["Add try/except"],
            }),
            # generate_correction_plan (reasoning role)
            json.dumps(["Fix error handling"]),
            # execute correction step (coding role)
            "corrected output",
            # review_output #2 -> PASS
            json.dumps({"verdict": "PASS", "feedback": "Fixed", "correction_hints": []}),
        ]):
            result = route_task(fabric, None, task="build widget")
            assert result == "corrected output"

    def test_review_bounded_by_max_iterations(self):
        """route_task stops after max_iterations even if still FAIL."""
        fabric = _make_review_fabric(max_iterations=2)
        with patch.object(fabric, "execute", side_effect=[
            # analyze_intent
            json.dumps({"intent": "SIMPLE_CODE"}),
            # execute task
            "bad output",
            # review #1 -> FAIL
            json.dumps({"verdict": "FAIL", "feedback": "Bad", "correction_hints": []}),
            # generate_correction_plan
            json.dumps(["Fix it"]),
            # correction step
            "still bad",
            # review #2 -> FAIL (max reached, stops here)
            json.dumps({"verdict": "FAIL", "feedback": "Still bad", "correction_hints": []}),
        ]):
            result = route_task(fabric, None, task="build widget")
            # Returns the last corrected output
            assert result == "still bad"


# ------------------------------------------------------------------
# Review loop in generate_code
# ------------------------------------------------------------------

class TestGenerateCodeReviewLoop:
    def test_review_pass_first_time(self):
        """generate_code with review PASS on first iteration returns output."""
        fabric = _make_review_fabric(max_iterations=3)
        with patch.object(fabric, "execute", side_effect=[
            # analyze_intent
            json.dumps({"intent": "SIMPLE_CODE"}),
            # execute task
            "def add(a, b): return a + b",
            # review -> PASS
            json.dumps({"verdict": "PASS", "feedback": "Good", "correction_hints": []}),
        ]):
            result = generate_code(
                fabric, None,
                task_description="add function",
                language="python",
            )
            assert "def add" in result

    def test_skips_review_when_no_reviewer(self):
        """generate_code skips review when reviewer chain is empty."""
        fabric = _make_review_fabric(reviewer_chain=[])
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),
            "def foo(): pass",
        ]):
            result = generate_code(
                fabric, None,
                task_description="write foo",
            )
            assert "def foo" in result
            reviewer_calls = [
                c for c in fabric.execute.call_args_list
                if c[0][0] == "reviewer"
            ]
            assert len(reviewer_calls) == 0
