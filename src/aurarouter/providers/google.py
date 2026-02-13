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
