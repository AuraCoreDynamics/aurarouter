"""Gist extraction, injection, and summarizer fallback."""

from __future__ import annotations

GIST_MARKER = "---GIST---"

GIST_INSTRUCTION = (
    "\n\n---\n"
    "After your response, on a new line add the exact marker '---GIST---' "
    "followed by a 2-sentence factual summary of your response. "
    "This summary will be used as context for future turns."
)


def inject_gist_instruction(content: str) -> str:
    """Append the hidden gisting instruction to a message's content.

    This is appended to the last user message before sending to the model.
    """
    return content + GIST_INSTRUCTION


def extract_gist(response_text: str) -> tuple[str, str | None]:
    """Split a model response into clean content and an optional gist.

    Returns:
        (content, gist) — gist is None if no marker found.
        If marker appears multiple times, uses the last occurrence.
    """
    if GIST_MARKER not in response_text:
        return response_text, None

    # Use the last occurrence of the marker
    idx = response_text.rfind(GIST_MARKER)
    content = response_text[:idx].rstrip()
    gist = response_text[idx + len(GIST_MARKER):].strip()

    if not gist:
        return content, None

    return content, gist


def build_condensation_prompt(messages: list[dict]) -> str:
    """Build a prompt asking a summarizer to condense message history.

    Args:
        messages: List of {"role": ..., "content": ...} dicts to summarize.

    Returns:
        A prompt string for the summarizer role.
    """
    history_text = "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
        for m in messages
    )
    return (
        "You are a precise technical summarizer. "
        "Condense the following conversation into a brief factual summary "
        "that preserves all key decisions, constraints, and technical details. "
        "The summary must be complete enough to serve as context for "
        "continuing the conversation without the original messages.\n\n"
        f"CONVERSATION:\n{history_text}\n\n"
        "SUMMARY:"
    )


def build_fallback_gist_prompt(response_text: str) -> str:
    """Build a prompt to generate a gist when the model didn't provide one.

    Args:
        response_text: The model's response that lacks a gist.

    Returns:
        A prompt string for the summarizer role.
    """
    return (
        "Provide a 2-sentence factual summary of the following text. "
        "Include only key facts and decisions. No commentary.\n\n"
        f"TEXT:\n{response_text}\n\n"
        "SUMMARY:"
    )
