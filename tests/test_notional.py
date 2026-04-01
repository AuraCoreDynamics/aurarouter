"""Tests for notional response streaming and correction protocol (TG7)."""

from __future__ import annotations

from aurarouter.notional import CorrectionEvent, NotionalEmitter, NotionalResponse


class TestNotionalResponse:
    def test_to_dict(self):
        resp = NotionalResponse(
            session_id="s1",
            content="draft output",
            drafter_model="drafter-3b",
        )
        d = resp.to_dict()
        assert d["session_id"] == "s1"
        assert d["status"] == "notional"
        assert d["content"] == "draft output"

    def test_frozen(self):
        resp = NotionalResponse(session_id="s1", content="x", drafter_model="d")
        try:
            resp.content = "y"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestCorrectionEvent:
    def test_to_dict(self):
        event = CorrectionEvent(
            session_id="s1",
            correction_position=5,
            correction_tokens=[42, 43],
        )
        d = event.to_dict()
        assert d["correction_position"] == 5
        assert d["correction_tokens"] == [42, 43]
        assert d["reason"] == "verifier_rejection"

    def test_default_reason(self):
        event = CorrectionEvent(session_id="s1", correction_position=0)
        assert event.reason == "verifier_rejection"


class TestNotionalEmitter:
    def test_should_emit_above_threshold(self):
        emitter = NotionalEmitter(confidence_threshold=0.85)
        assert emitter.should_emit(0.9)

    def test_should_not_emit_below_threshold(self):
        emitter = NotionalEmitter(confidence_threshold=0.85)
        assert not emitter.should_emit(0.5)

    def test_should_emit_at_threshold(self):
        emitter = NotionalEmitter(confidence_threshold=0.85)
        assert emitter.should_emit(0.85)

    def test_emit_notional_gated(self):
        emitter = NotionalEmitter(confidence_threshold=0.85)
        result = emitter.emit_notional("s1", "content", "d", confidence=0.5)
        assert result is None
        assert emitter.emitted_count == 0

    def test_emit_notional_passes(self):
        emitter = NotionalEmitter(confidence_threshold=0.85)
        result = emitter.emit_notional("s1", "content", "d", confidence=0.9)
        assert result is not None
        assert result.status == "notional"
        assert emitter.emitted_count == 1

    def test_emit_notional_callback_invoked(self):
        received = []
        emitter = NotionalEmitter(
            confidence_threshold=0.5,
            on_notional=lambda r: received.append(r),
        )
        emitter.emit_notional("s1", "content", "d", confidence=0.8)
        assert len(received) == 1
        assert isinstance(received[0], NotionalResponse)

    def test_emit_correction(self):
        emitter = NotionalEmitter()
        event = emitter.emit_correction("s1", correction_position=3, correction_tokens=[42])
        assert event.correction_position == 3
        assert emitter.correction_count == 1

    def test_emit_correction_callback_invoked(self):
        received = []
        emitter = NotionalEmitter(on_correction=lambda e: received.append(e))
        emitter.emit_correction("s1", correction_position=0)
        assert len(received) == 1
        assert isinstance(received[0], CorrectionEvent)

    def test_correction_ordering(self):
        """Corrections should be recorded in order."""
        emitter = NotionalEmitter()
        emitter.emit_correction("s1", correction_position=5)
        emitter.emit_correction("s1", correction_position=10)
        emitter.emit_correction("s1", correction_position=15)
        assert emitter.correction_count == 3
        assert emitter._corrections[0].correction_position == 5
        assert emitter._corrections[2].correction_position == 15
