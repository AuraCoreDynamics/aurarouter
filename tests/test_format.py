"""Tests for GUI formatting utilities."""

from aurarouter.gui._format import format_cost, format_duration, format_tokens


def test_format_tokens_small():
    assert format_tokens(500) == "500"


def test_format_tokens_thousands():
    assert format_tokens(1500) == "1,500"


def test_format_tokens_millions():
    assert format_tokens(1500000) == "1.50M"


def test_format_cost():
    assert format_cost(0.035) == "$0.04"


def test_format_cost_dollars():
    assert format_cost(12.5) == "$12.50"


def test_format_duration_seconds():
    assert format_duration(0.5) == "0.5s"


def test_format_duration_minutes():
    assert format_duration(65) == "1m 5s"
