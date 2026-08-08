import os

import requests

from models.base import ModelClient


class OllamaClient(ModelClient):
    """Wraps a local Ollama server's /api/generate endpoint.

    Requires `ollama serve` running locally and the target model pulled
    (e.g. `ollama pull llama3`).
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self._host}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
