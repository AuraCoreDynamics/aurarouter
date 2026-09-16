"""Anonymization framework for the AuraRouter pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aurarouter.sovereignty.privacy import PrivacyMatch


class BaseAnonymizer:
    """Base interface for anonymization plugins."""
    
    supports_streaming: bool = False
    
    def anonymize(self, prompt: str, segments: list["PrivacyMatch"]) -> tuple[str, dict[str, str]]:
        """Scrub the prompt using the detected segments.
        
        Returns:
            A tuple of (anonymized_prompt, mapping)
        """
        raise NotImplementedError
        
    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        """Restore the original text using the mapping.
        
        Returns:
            The deanonymized text.
        """
        raise NotImplementedError


class RegexAnonymizer(BaseAnonymizer):
    """Deterministically scrubs prompt using Regex-based PrivacyMatches.
    
    Replaces sensitive data with sequential tags like <PII_EMAIL_1>.
    """
    
    supports_streaming: bool = False
    
    def anonymize(self, prompt: str, segments: list["PrivacyMatch"]) -> tuple[str, dict[str, str]]:
        mapping = {}
        anonymized_prompt = prompt
        
        # We need to sort segments by length descending or position to avoid partial overlaps,
        # but for simplicity we'll just replace them one by one.
        # It's safer to sort by length descending to replace longest strings first.
        segments_sorted = sorted(segments, key=lambda x: len(x.matched_text), reverse=True)
        
        category_counters: dict[str, int] = {}
        
        for match in segments_sorted:
            if not match.matched_text:
                continue
                
            cat = match.category.upper()
            if cat not in category_counters:
                category_counters[cat] = 1
            else:
                category_counters[cat] += 1
                
            placeholder = f"<{cat}_{category_counters[cat]}>"
            
            # Map the placeholder back to the original text
            mapping[placeholder] = match.matched_text
            
            # Replace in prompt
            anonymized_prompt = anonymized_prompt.replace(match.matched_text, placeholder)
            
        return anonymized_prompt, mapping

    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        deanonymized_text = text
        for placeholder, original in mapping.items():
            deanonymized_text = deanonymized_text.replace(placeholder, original)
        return deanonymized_text


class AnonymizationPipeline:
    """Orchestrates multiple BaseAnonymizer plugins."""
    
    def __init__(self, plugins: list[BaseAnonymizer] = None) -> None:
        self.plugins = plugins or []
        
    @property
    def supports_streaming(self) -> bool:
        """Pipeline supports streaming only if all its plugins do."""
        if not self.plugins:
            return True
        return all(p.supports_streaming for p in self.plugins)

    def anonymize(self, prompt: str, segments: list["PrivacyMatch"]) -> tuple[str, dict[str, str]]:
        """Run all anonymizers sequentially on the prompt."""
        current_prompt = prompt
        combined_mapping = {}
        
        for plugin in self.plugins:
            current_prompt, plugin_map = plugin.anonymize(current_prompt, segments)
            combined_mapping.update(plugin_map)
            
        return current_prompt, combined_mapping
        
    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        """Run deanonymization in reverse order."""
        current_text = text
        for plugin in reversed(self.plugins):
            current_text = plugin.deanonymize(current_text, mapping)
        return current_text


class SemanticSLMAnonymizer(BaseAnonymizer):
    """Probabilistically scrubs prompt using a local SLM via ComputeFabric."""
    
    supports_streaming: bool = False
    
    def __init__(self):
        self._fabric = None
        
    def set_fabric(self, fabric):
        """Inject the fabric instance to avoid circular initialization dependencies."""
        self._fabric = fabric

    def anonymize(self, prompt: str, segments: list["PrivacyMatch"]) -> tuple[str, dict[str, str]]:
        """Ask the SLM to find and replace sensitive entities."""
        if not self._fabric:
            import logging
            logging.getLogger(__name__).warning("SemanticSLMAnonymizer called without fabric attached. Skipping.")
            return prompt, {}
            
        slm_prompt = (
            "Analyze the following text and identify ANY highly sensitive information (PII, names, financials, secrets). "
            "Output ONLY a valid JSON object mapping a placeholder (e.g. '<SLM_REDACTED_1>') to the exact original sensitive string you found. "
            "Do not include any explanation or markdown blocks.\n\n"
            f"Text: {prompt}"
        )
        
        try:
            # We must use bypass_anonymization=True or similar in options to prevent infinite recursion
            # if the 'anonymizer' role evaluates to a chain that triggers sovereignty checks.
            # We'll just pass a flag in options.
            result = self._fabric.execute(
                role="anonymizer",
                prompt=slm_prompt,
                json_mode=True,
                options={"bypass_anonymization": True, "intent": "anonymize"},
            )
            
            if not result or not result.success:
                return prompt, {}
                
            import json
            text = result.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
                
            mapping = json.loads(text.strip())
            if not isinstance(mapping, dict):
                return prompt, {}
                
            anonymized_prompt = prompt
            valid_mapping = {}
            for placeholder, original in mapping.items():
                if isinstance(original, str) and original in prompt:
                    anonymized_prompt = anonymized_prompt.replace(original, placeholder)
                    valid_mapping[placeholder] = original
                    
            return anonymized_prompt, valid_mapping
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"SemanticSLMAnonymizer execution failed: {e}")
            return prompt, {}

    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        deanonymized = text
        for placeholder, original in mapping.items():
            if isinstance(placeholder, str) and isinstance(original, str):
                deanonymized = deanonymized.replace(placeholder, original)
        return deanonymized
