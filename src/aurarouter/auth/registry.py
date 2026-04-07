"""AuraRouter Auth Registry: Maps providers to their authentication metadata."""

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AuthMetadata:
    provider: str
    display_name: str
    auth_url: str
    key_regex: str
    help_text: str

# Registry of known cloud/managed providers
AUTH_REGISTRY: dict[str, AuthMetadata] = {
    "anthropic": AuthMetadata(
        provider="anthropic",
        display_name="Anthropic (Claude)",
        auth_url="https://console.anthropic.com/settings/keys",
        key_regex=r"sk-ant-api03-[a-zA-Z0-9\-_]{90,100}",
        help_text="Go to the Anthropic Console, create a key, and copy it."
    ),
    "openai": AuthMetadata(
        provider="openai",
        display_name="OpenAI (GPT-4o)",
        auth_url="https://platform.openai.com/api-keys",
        key_regex=r"sk-[a-zA-Z0-9]{32,100}", # Matches both legacy and project keys
        help_text="Create a new secret key on the OpenAI Platform."
    ),
    "huggingface": AuthMetadata(
        provider="huggingface",
        display_name="Hugging Face",
        auth_url="https://huggingface.co/settings/tokens",
        key_regex=r"(hf_[a-zA-Z0-9]{34,40}|hf_token_[a-zA-Z0-9]{34,40})",
        help_text="Generate a 'Read' token in your Hugging Face settings."
    ),
    "groq": AuthMetadata(
        provider="groq",
        display_name="GroqCloud",
        auth_url="https://console.groq.com/keys",
        key_regex=r"gsk_[a-zA-Z0-9]{48,60}",
        help_text="Generate a new API key in the Groq Console."
    ),
    "google": AuthMetadata(
        provider="google",
        display_name="Google (Gemini)",
        auth_url="https://aistudio.google.com/app/apikey",
        key_regex=r"AIzaSy[a-zA-Z0-9\-_]{33}",
        help_text="Get an API key from Google AI Studio."
    ),
}

def get_auth_metadata(provider_id: str) -> Optional[AuthMetadata]:
    """Retrieve metadata for a specific provider, or None if unknown."""
    return AUTH_REGISTRY.get(provider_id.lower())

def find_by_url(url: str) -> Optional[AuthMetadata]:
    """Identify a provider based on a URL substring."""
    for meta in AUTH_REGISTRY.values():
        if meta.auth_url in url:
            return meta
    return None
