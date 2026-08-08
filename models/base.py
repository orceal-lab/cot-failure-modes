"""Common interface all model clients implement, so run_experiment.py
doesn't need to care which provider it's talking to."""

from abc import ABC, abstractmethod


class ModelClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the model and return the raw text response."""
        raise NotImplementedError

    @property
    def label(self) -> str:
        """Identifier used in result filenames, e.g. 'anthropic-claude-haiku-4-5'."""
        provider = self.__class__.__name__.replace("Client", "").lower()
        safe_model = self.model_name.replace("/", "-").replace(":", "-")
        return f"{provider}-{safe_model}"
