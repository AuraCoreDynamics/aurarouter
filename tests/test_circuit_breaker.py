"""Tests for aurarouter.circuit_breaker — per-provider circuit breaker."""

import threading
import time
import pytest
from aurarouter.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test-provider")
        assert cb.is_available() is True
        state = cb.get_health_state()
        assert state.circuit_state == "closed"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_available() is False
        state = cb.get_health_state()
        assert state.circuit_state == "open"
        assert state.consecutive_failures == 3

    def test_success_resets_failures(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.is_available() is True
        state = cb.get_health_state()
        assert state.consecutive_failures == 0

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_available() is False
        time.sleep(0.15)
        # After timeout, should transition to half_open and allow one probe
        assert cb.is_available() is True

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.is_available()  # triggers half_open
        cb.record_success()
        state = cb.get_health_state()
        assert state.circuit_state == "closed"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.is_available()  # triggers half_open
        cb.record_failure()
        state = cb.get_health_state()
        assert state.circuit_state == "open"

    def test_health_state_timestamps(self):
        from datetime import datetime, timezone
        cb = CircuitBreaker("test")
        cb.record_success()
        state = cb.get_health_state()
        assert state.last_success is not None
        assert isinstance(state.last_success, datetime)
        assert state.last_success.tzinfo is not None

        cb.record_failure()
        state = cb.get_health_state()
        assert state.last_failure is not None
        assert isinstance(state.last_failure, datetime)

    def test_thread_safety(self):
        cb = CircuitBreaker("test", failure_threshold=100)
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    cb.record_failure()
                    cb.record_success()
                    cb.is_available()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestCircuitBreakerRegistry:
    def test_get_or_create(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("provider-a")
        cb2 = reg.get_or_create("provider-a")
        assert cb1 is cb2

    def test_different_providers(self):
        reg = CircuitBreakerRegistry()
        cb_a = reg.get_or_create("a")
        cb_b = reg.get_or_create("b")
        assert cb_a is not cb_b

    def test_health_summary(self):
        reg = CircuitBreakerRegistry()
        reg.get_or_create("a").record_success()
        reg.get_or_create("b").record_failure()
        summary = reg.get_health_summary()
        assert "a" in summary
        assert "b" in summary
