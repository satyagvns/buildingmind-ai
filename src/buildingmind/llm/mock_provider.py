"""Deterministic extractive provider; no network access or randomness."""

from buildingmind.rag.grounding import supported_hypotheses
from buildingmind.rag.models import AnswerDraft, Claim
from buildingmind.rag.prompt_builder import Prompt


class MockLLMProvider:
    def generate(self, prompt: Prompt) -> AnswerDraft:
        return AnswerDraft(
            observations=tuple(
                Claim(evidence_id=e.id, quote=e.text)
                for e in prompt.context.evidence
                if e.kind == "telemetry"
            ),
            historical_evidence=tuple(
                Claim(evidence_id=e.id, quote=e.text)
                for e in prompt.context.evidence
                if e.kind != "telemetry"
            ),
            hypotheses=supported_hypotheses(prompt.context),
        )
