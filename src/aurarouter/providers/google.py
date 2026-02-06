from google import genai

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider

logger = get_logger("AuraRouter.Google")


class GoogleProvider(BaseProvider):
    """Google Generative AI provider (Gemini models)."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        api_key = self.resolve_api_key()
        if not api_key:
            raise ValueError(
                f"No API key found for Google model '{self.config.get('model_name')}'. "
                "Set 'api_key' in config or use 'env_key' to reference an env var."
            )

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=self.config["model_name"],
            contents=prompt,
        )
        return resp.text
