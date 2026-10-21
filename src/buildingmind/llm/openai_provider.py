"""Optional Responses API adapter. SDK import and credentials are lazy."""

import math
from typing import Any

from buildingmind.llm.base import ProviderError
from buildingmind.rag.models import AnswerDraft
from buildingmind.rag.prompt_builder import Prompt


class OpenAIProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any = None,
        timeout: float = 30,
        max_output_tokens: int = 3000,
    ):
        if (
            not model.strip()
            or not math.isfinite(timeout)
            or timeout <= 0
            or max_output_tokens <= 0
        ):
            raise ValueError("Explicit model and positive finite limits are required")
        self.model = model
        self.api_key = api_key
        self.client = client
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: Prompt) -> AnswerDraft:
        try:
            if self.client is None:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=0)
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                text_format=AnswerDraft,
                max_output_tokens=self.max_output_tokens,
                timeout=self.timeout,
                store=False,
            )
            if response.status != "completed" or response.output_parsed is None:
                raise ProviderError("Model refused or returned incomplete structured output")
            return AnswerDraft.model_validate(response.output_parsed.model_dump())
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                "OpenAI provider failed; check local configuration and service status"
            ) from exc
