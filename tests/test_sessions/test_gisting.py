"""Tests for gist extraction, injection, and prompt building."""

from aurarouter.sessions.gisting import (
    GIST_MARKER,
    GIST_INSTRUCTION,
    inject_gist_instruction,
    extract_gist,
    build_condensation_prompt,
    build_fallback_gist_prompt,
)


def test_inject_gist_instruction():
    content = "Write a fibonacci function"
    result = inject_gist_instruction(content)
    assert result.startswith(content)
    assert GIST_INSTRUCTION in result
    assert "---GIST---" in result


def test_extract_gist_with_marker():
    response = "Here is the code.\n---GIST---\nThe response provides a fibonacci function."
    content, gist = extract_gist(response)
    assert content == "Here is the code."
    assert gist == "The response provides a fibonacci function."


def test_extract_gist_no_marker():
    response = "Here is the code with no gist marker."
    content, gist = extract_gist(response)
    assert content == response
    assert gist is None


def test_extract_gist_multiple_markers():
    response = (
        "First part\n---GIST---\nFirst gist\n"
        "More content\n---GIST---\nSecond gist (should use this one)"
    )
    content, gist = extract_gist(response)
    assert gist == "Second gist (should use this one)"
    assert "First part" in content
    assert "First gist" in content


def test_extract_gist_empty_gist():
    response = "Some content\n---GIST---\n"
    content, gist = extract_gist(response)
    assert content == "Some content"
    assert gist is None


def test_extract_gist_whitespace_only_gist():
    response = "Some content\n---GIST---\n   \n  "
    content, gist = extract_gist(response)
    assert content == "Some content"
    assert gist is None


def test_build_condensation_prompt():
    messages = [
        {"role": "user", "content": "Write a function"},
        {"role": "assistant", "content": "Here is the function"},
    ]
    prompt = build_condensation_prompt(messages)
    assert "USER: Write a function" in prompt
    assert "ASSISTANT: Here is the function" in prompt
    assert "summarizer" in prompt.lower()
    assert "SUMMARY:" in prompt


def test_build_fallback_gist_prompt():
    response = "Here is a detailed implementation of the algorithm."
    prompt = build_fallback_gist_prompt(response)
    assert response in prompt
    assert "2-sentence" in prompt
    assert "SUMMARY:" in prompt
