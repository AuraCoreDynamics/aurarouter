"""Tests for savings data models."""

import dataclasses

from aurarouter.savings.models import GenerateResult, UsageRecord


def test_generate_result_str():
    result = GenerateResult(text="hello")
    assert str(result) == "hello"


def test_generate_result_defaults():
    result = GenerateResult(text="test")
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.model_id == ""
    assert result.provider == ""


def test_generate_result_with_tokens():
    result = GenerateResult(
        text="output", input_tokens=100, output_tokens=50, model_id="m1", provider="ollama"
    )
    assert result.text == "output"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.model_id == "m1"
    assert result.provider == "ollama"
    assert str(result) == "output"


def test_usage_record_creation():
    rec = UsageRecord(
        timestamp="2025-01-15T10:30:00Z",
        model_id="gemini-pro",
        provider="google",
        role="router",
        intent="SIMPLE_CODE",
        input_tokens=200,
        output_tokens=400,
        elapsed_s=1.5,
        success=True,
        is_cloud=True,
    )
    assert rec.timestamp == "2025-01-15T10:30:00Z"
    assert rec.model_id == "gemini-pro"
    assert rec.provider == "google"
    assert rec.role == "router"
    assert rec.intent == "SIMPLE_CODE"
    assert rec.input_tokens == 200
    assert rec.output_tokens == 400
    assert rec.elapsed_s == 1.5
    assert rec.success is True
    assert rec.is_cloud is True


def test_usage_record_is_dataclass():
    assert dataclasses.is_dataclass(UsageRecord)
    assert dataclasses.is_dataclass(GenerateResult)
