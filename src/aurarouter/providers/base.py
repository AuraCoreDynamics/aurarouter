import os
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Optional

from aurarouter.savings.models import GenerateResult


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model_config: dict):
        self.config = model_config

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False,
                 response_schema: dict | None = None) -> str:
        """Send a prompt and return the text response."""
        ...

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> GenerateResult:
        """Generate a response with token-usage metadata."""
        try:
            text = self.generate(prompt, json_mode=json_mode, response_schema=response_schema)
        except TypeError:
            text = self.generate(prompt, json_mode=json_mode)
        return GenerateResult(text=text)

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Session-aware generation with message history."""
        parts = []
        if system_prompt:
            parts.append(f"[System]\n{system_prompt}\n")
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}\n")
        combined_prompt = "\n".join(parts)
        return self.generate_with_usage(combined_prompt, json_mode=json_mode)

    async def generate_stream(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> AsyncIterator[str]:
        try:
            result = self.generate(prompt, json_mode=json_mode, response_schema=response_schema)
        except TypeError:
            result = self.generate(prompt, json_mode=json_mode)
        yield result

    def generate_stream_sync(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
    ):
        try:
            result = self.generate(prompt, json_mode=json_mode, response_schema=response_schema)
        except TypeError:
            result = self.generate(prompt, json_mode=json_mode)
        yield result

    async def generate_stream_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        result = self.generate_with_history(
            messages, system_prompt, json_mode=json_mode
        )
        yield result.text

    def get_context_limit(self) -> int:
        return self.config.get("context_limit", 0)

    def get_telemetry(self):
        try:
            from aurarouter.auragrid.contracts import ModelState, ModelTelemetry
        except ImportError:
            return None
        return ModelTelemetry(
            model_id=self.config.get("model_name", "unknown"),
            provider_name=self.__class__.__name__,
            state=ModelState.UNKNOWN,
        )

    def resolve_api_key(self) -> Optional[str]:
        key = self.config.get("api_key")
        if key and "YOUR_PASTED_KEY" not in str(key) and "YOUR_API_KEY" not in str(key):
            return key
        env_key = self.config.get("env_key")
        if env_key:
            return os.environ.get(env_key)
        return None


class MockProvider(BaseProvider):
    """Fakes LLM responses for development and demo purposes."""

    def generate(self, prompt: str, json_mode: bool = False, response_schema: dict | None = None) -> str:
        p = prompt.lower()
        
        # Priority 1: Handle Intent Classification (Internal AuraRouter protocol)
        # If AuraCode's [ROUTE_OPTIONS] is present BUT we are in a 'router' context
        # (detected by 'classify' or 'options:'), return the JSON.
        if "classify" in p or "options:" in p:
            return '{"intent": "chat", "complexity": 1, "confidence": 0.99}'
            
        # Priority 2: Specific ZReach demo content
        # Strip [ROUTE_OPTIONS] and [System] markers for cleaner matching
        clean_p = p.replace("[route_options]", "").replace("[/route_options]", "").replace("[system]", "")
        
        if "who can" in clean_p or "experts" in clean_p or "container" in clean_p or "experience" in clean_p:
            return "Based on the knowledge graph, several people have expertise in that area. For example, John Doe has deep experience with containerization and orchestration."
        elif "analyze" in clean_p or "extract" in clean_p:
            # Fake JSON for intent extraction (ZReach local RAG)
            return '{"target_entities": ["Person"], "skills": ["containerization"], "projects": [], "search_query": "containers docker kubernetes"}'
        elif "summarize" in clean_p:
            return "These results show a strong alignment with your request for competency data."
        
        # Priority 3: General JSON mode support
        if json_mode:
            return '{"answer": "This is a mock JSON response from the MockProvider."}'
            
        # Catch-all
        return f"This is a mock response from the AuraRouter MockProvider for query: {clean_p[:100].strip()}..."

    def generate_with_history(self, messages: list[dict], system_prompt: str = "", json_mode: bool = False) -> GenerateResult:
        # Check all messages for classification hints
        is_classification = False
        for msg in messages:
            content = msg.get("content", "").lower()
            if "classify" in content or "options:" in content:
                is_classification = True
                break
        
        if is_classification:
            return GenerateResult(
                text='{"intent": "chat", "complexity": 1, "confidence": 0.99}',
                model_id=self.config.get("model_name", "mock-llm"),
                provider=self.__class__.__name__
            )

        prompt = messages[-1].get("content", "")
        text = self.generate(prompt, json_mode=json_mode)
        return GenerateResult(
            text=text,
            model_id=self.config.get("model_name", "mock-llm"),
            provider=self.__class__.__name__
        )

    def get_telemetry(self):
        try:
            from aurarouter.auragrid.contracts import ModelState, ModelTelemetry
        except ImportError:
            return None
        return ModelTelemetry(
            model_id=self.config.get("model_name", "mock-llm"),
            provider_name=self.__class__.__name__,
            state=ModelState.ONLINE,
            latency=0.01
        )
