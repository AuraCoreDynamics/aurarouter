import anthropic

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider

logger = get_logger("AuraRouter.Claude")


class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        api_key = self.resolve_api_key()
        if not api_key:
            raise ValueError(
                f"No API key found for Claude model '{self.config.get('model_name')}'. "
                "Set 'api_key' in config or use 'env_key: ANTHROPIC_API_KEY'."
            )

        params = self.config.get("parameters", {})

        messages = [{"role": "user", "content": prompt}]

        # For JSON mode, add a system instruction requesting structured output
        system = None
        if json_mode:
            system = (
                "You must respond with valid JSON only. "
                "No markdown fences, no commentary — raw JSON."
            )

        client = anthropic.Anthropic(api_key=api_key)
        kwargs: dict = {
            "model": self.config["model_name"],
            "max_tokens": params.get("max_tokens", 4096),
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]

        message = client.messages.create(**kwargs)
        return message.content[0].text
