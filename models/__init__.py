"""Model client implementations, selected by a "provider:model" string."""

from models.anthropic_client import AnthropicClient
from models.ollama_client import OllamaClient


def get_client(model_spec: str):
    """Build a ModelClient from a spec like 'anthropic:claude-haiku-4-5-20251001'
    or 'ollama:llama3'."""
    provider, _, model_name = model_spec.partition(":")
    if not model_name:
        raise ValueError(f"Model spec must be 'provider:model_name', got {model_spec!r}")

    if provider == "anthropic":
        return AnthropicClient(model_name)
    if provider == "ollama":
        return OllamaClient(model_name)
    raise ValueError(f"Unknown provider {provider!r}. Supported: anthropic, ollama")
