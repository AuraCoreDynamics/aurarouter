from google import genai

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.Google")


class GoogleProvider(BaseProvider):
    """Google Generative AI provider (Gemini models)."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
        api_key = self.resolve_api_key()
        if not api_key:
            raise ValueError(
                f"No API key found for Google model '{self.config.get('model_name')}'. "
                "Set 'api_key' in config or use 'env_key' to reference an env var."
            )

        client = genai.Client(api_key=api_key)

        # Configure JSON output mode if requested
        config = None
        if json_mode:
            config = genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )

        resp = client.models.generate_content(
            model=self.config["model_name"],
            contents=prompt,
            generation_config=config,
        )

        input_tokens = 0
        output_tokens = 0
        try:
            meta = resp.usage_metadata
            if meta is not None:
                input_tokens = getattr(meta, "prompt_token_count", 0) or 0
                output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        except Exception:
            pass

        return GenerateResult(
            text=resp.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via Google GenAI with message history."""
        from google.genai import types

        api_key = self.resolve_api_key()
        if not api_key:
            raise RuntimeError("Google API key not configured")

        client = genai.Client(api_key=api_key)

        # Build content history
        history_contents = []
        for msg in messages[:-1]:
            role = "model" if msg["role"] == "assistant" else "user"
            history_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        current_msg = messages[-1]["content"] if messages else ""

        config_kwargs = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        resp = client.models.generate_content(
            model=self.config["model_name"],
            contents=history_contents + [
                types.Content(
                    role="user",
                    parts=[types.Part(text=current_msg)],
                )
            ],
            config=config,
        )

        text = resp.text or ""
        input_tokens = 0
        output_tokens = 0
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            input_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0

        return GenerateResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=self.config.get("model_name", ""),
            provider="google",
            context_limit=self.get_context_limit(),
        )

    def get_context_limit(self) -> int:
        """Return context limit from config or known Gemini model limits."""
        limit = self.config.get("context_limit", 0)
        if limit > 0:
            return limit
        model = self.config.get("model_name", "")
        known = {
            "gemini-2.0-flash": 1048576,
            "gemini-2.0-pro": 1048576,
            "gemini-2.5-flash": 1048576,
            "gemini-2.5-pro": 1048576,
        }
        return known.get(model, 0)
