from typing import Protocol

from buildingmind.rag.models import AnswerDraft
from buildingmind.rag.prompt_builder import Prompt


class ProviderError(RuntimeError):
    """Provider failure with no credentials or raw payload exposed to callers."""


class LLMProvider(Protocol):
    def generate(self, prompt: Prompt) -> AnswerDraft: ...
