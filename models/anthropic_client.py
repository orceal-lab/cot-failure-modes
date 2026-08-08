import os

from anthropic import Anthropic

from models.base import ModelClient

MAX_TOKENS = 2048


class AnthropicClient(ModelClient):
    """Wraps the Anthropic Messages API. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export it in your shell."
            )
        self._client = Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
