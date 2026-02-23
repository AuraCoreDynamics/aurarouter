"""Tests for GenerateResult extensions (context_limit, gist, usage)."""

from aurarouter.savings.models import GenerateResult


def test_generate_result_new_fields_defaults():
    r = GenerateResult(text="hi")
    assert r.context_limit == 0
    assert r.gist is None


def test_generate_result_backwards_compat():
    r = GenerateResult(text="hi", input_tokens=10, output_tokens=5)
    assert r.text == "hi"
    assert r.input_tokens == 10
    assert r.output_tokens == 5
    assert r.context_limit == 0
    assert r.gist is None


def test_generate_result_usage_property():
    r = GenerateResult(text="hi", input_tokens=100, output_tokens=50, context_limit=8192)
    usage = r.usage
    assert set(usage.keys()) == {"input", "output", "remaining", "limit"}
    assert usage["input"] == 100
    assert usage["output"] == 50
    assert usage["remaining"] == 8042
    assert usage["limit"] == 8192


def test_generate_result_usage_no_limit():
    r = GenerateResult(text="hi", input_tokens=100, output_tokens=50)
    assert r.usage["remaining"] == 0
    assert r.usage["limit"] == 0


def test_generate_result_usage_with_limit():
    r = GenerateResult(
        text="hi", input_tokens=1000, output_tokens=500, context_limit=8192
    )
    assert r.usage["remaining"] == 6692


def test_generate_result_str_unchanged():
    r = GenerateResult(text="hello world", context_limit=1000, gist="summary")
    assert str(r) == "hello world"
