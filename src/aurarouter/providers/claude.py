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

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via Anthropic messages API with full history."""
        api_key = self.resolve_api_key()
        if not api_key:
            raise RuntimeError("Anthropic API key not configured")

        client = anthropic.Anthropic(api_key=api_key)
        params = self.config.get("parameters", {})

        # Filter out system messages — Anthropic uses a separate system parameter
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]

        # Collect any system messages from the history and prepend to system_prompt
        history_system = "\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        full_system = "\n".join(filter(None, [system_prompt, history_system]))

        kwargs = {
            "model": self.config["model_name"],
            "max_tokens": params.get("max_tokens", 8192),
            "messages": api_messages,
        }
        if full_system:
            kwargs["system"] = full_system
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]

        message = client.messages.create(**kwargs)

        text = message.content[0].text
        return GenerateResult(
            text=text,
            input_tokens=getattr(message.usage, "input_tokens", 0),
            output_tokens=getattr(message.usage, "output_tokens", 0),
            model_id=self.config.get("model_name", ""),
            provider="claude",
            context_limit=self.get_context_limit(),
        )

    def get_context_limit(self) -> int:
        """Return context limit from config or known Claude model limits."""
        limit = self.config.get("context_limit", 0)
        if limit > 0:
            return limit
        model = self.config.get("model_name", "")
        known = {
            "claude-opus-4-6": 200000,
            "claude-sonnet-4-5-20250929": 200000,
            "claude-haiku-4-5-20251001": 200000,
        }
        if model in known:
            return known[model]
        for prefix, ctx_limit in known.items():
            if model.startswith(prefix.split("-202")[0]):
                return ctx_limit
        return 0
