"""Tokenizer utility for accurate context budgeting."""

from __future__ import annotations

import logging
from typing import Optional, Callable, Dict

logger = logging.getLogger("AuraRouter.Tokens")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logger.debug("tiktoken not found, falling back to heuristic token counting")

# Registry for custom tokenizers (Task 4.3)
# Key: model_id pattern or provider name
# Value: Callable[[text], int]
_TOKENIZER_REGISTRY: Dict[str, Callable[[str], int]] = {}


def register_tokenizer(name: str, counter: Callable[[str], int]) -> None:
    """Register a custom tokenizer function.
    
    Args:
        name: Name or pattern to match model_id or provider.
        counter: Function that takes text and returns token count.
    """
    _TOKENIZER_REGISTRY[name] = counter
    logger.info(f"Registered custom tokenizer: {name}")


def count_tokens(text: str, model_id: str = "") -> int:
    """Count tokens in a string using registered tokenizers, tiktoken, or fallback.
    
    Args:
        text: The string to tokenize.
        model_id: Optional model name or provider to select the tokenizer.
        
    Returns:
        The estimated token count.
    """
    if not text:
        return 0

    # 1. Check custom registry (Task 4.3)
    if model_id:
        # Exact match
        if model_id in _TOKENIZER_REGISTRY:
            return _TOKENIZER_REGISTRY[model_id](text)
        
        # Pattern match (simple prefix)
        for pattern, counter in _TOKENIZER_REGISTRY.items():
            if model_id.startswith(pattern):
                return counter(text)

    # 2. Check tiktoken
    if HAS_TIKTOKEN:
        try:
            # Map common model prefixes to tiktoken encodings
            if any(m in model_id.lower() for m in ["gpt-4", "gpt-3.5"]):
                encoding = tiktoken.encoding_for_model(model_id)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug(f"Tiktoken encoding failed for model {model_id}: {e}")

    # 3. Fallback heuristic
    words = len(text.split())
    chars = len(text)
    return max(1, int(max(words * 1.3, chars / 4)))
