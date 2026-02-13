import anthropic

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.Claude")


class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
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

        input_tokens = 0
        output_tokens = 0
        try:
            usage = message.usage
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
        except Exception:
            pass

        return GenerateResult(
            text=message.content[0].text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
