"""Tests for the ExecutionTrace data model."""

from aurarouter.gui.execution_trace import (
    ExecutionTrace,
    ModelAttempt,
    NodeStatus,
    TraceNode,
)


def _make_node(id_: str, role: str = "coding", parent_ids: list | None = None, **kw):
    return TraceNode(
        id=id_,
        label=id_.replace("-", " ").title(),
        role=role,
        parent_ids=parent_ids or [],
        **kw,
    )


class TestTraceNode:
    def test_defaults(self):
        n = TraceNode(id="x", label="X", role="coding")
        assert n.status == NodeStatus.PENDING
        assert n.model_id is None
        assert n.elapsed_s == 0.0
        assert n.attempts == []
        assert n.parent_ids == []


class TestExecutionTrace:
    def test_add_and_retrieve(self):
        t = ExecutionTrace()
        n = _make_node("a")
        t.add_node(n)
        assert "a" in t.nodes
        assert t.nodes["a"] is n

    def test_set_status(self):
        t = ExecutionTrace()
        t.add_node(_make_node("a"))
        t.set_status("a", NodeStatus.RUNNING)
        assert t.nodes["a"].status == NodeStatus.RUNNING

    def test_set_status_missing_noop(self):
        t = ExecutionTrace()
        t.set_status("missing", NodeStatus.FAILED)  # no error

    def test_get_roots(self):
        t = ExecutionTrace()
        t.add_node(_make_node("root"))
        t.add_node(_make_node("child", parent_ids=["root"]))
        roots = t.get_roots()
        assert len(roots) == 1
        assert roots[0].id == "root"

    def test_get_children(self):
        t = ExecutionTrace()
        t.add_node(_make_node("root"))
        t.add_node(_make_node("c1", parent_ids=["root"]))
        t.add_node(_make_node("c2", parent_ids=["root"]))
        t.add_node(_make_node("other"))
        children = t.get_children("root")
        assert {c.id for c in children} == {"c1", "c2"}

    def test_total_elapsed(self):
        t = ExecutionTrace()
        t.add_node(_make_node("a", elapsed_s=1.5))
        t.add_node(_make_node("b", elapsed_s=2.5))
        assert t.total_elapsed() == 4.0

    def test_summary_empty(self):
        assert ExecutionTrace().summary() == ""

    def test_summary_simple_code(self):
        t = ExecutionTrace()
        t.add_node(_make_node("classify-0", role="router", elapsed_s=0.3))
        t.add_node(
            _make_node("execute-0", role="coding", parent_ids=["classify-0"], elapsed_s=1.0)
        )
        s = t.summary()
        assert "Classify" in s
        assert "Execute" in s
        assert "1.3s" in s

    def test_summary_multi_step(self):
        t = ExecutionTrace()
        t.add_node(_make_node("classify-0", role="router", elapsed_s=0.2))
        t.add_node(
            _make_node("plan-0", role="reasoning", parent_ids=["classify-0"], elapsed_s=0.5)
        )
        t.add_node(
            _make_node("step-0", role="coding", parent_ids=["plan-0"], elapsed_s=1.0)
        )
        t.add_node(
            _make_node("step-1", role="coding", parent_ids=["plan-0"], elapsed_s=1.5)
        )
        s = t.summary()
        assert "Classify" in s
        assert "Plan" in s
        assert "Steps 1" in s


class TestModelAttempt:
    def test_defaults(self):
        a = ModelAttempt(model_id="m", success=True, elapsed_s=0.5)
        assert a.input_tokens == 0
        assert a.error == ""
