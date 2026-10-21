"""Read-only, provider-independent question-to-diagnostic orchestration."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from buildingmind.context.base import HistoricalContextProvider
from buildingmind.llm.base import LLMProvider
from buildingmind.rag.context_builder import ContextBuilder
from buildingmind.rag.grounding import GroundingError, detect_conflicts, ground_answer
from buildingmind.rag.models import AnswerDraft, DiagnosticResponse
from buildingmind.rag.prompt_builder import PromptBuilder
from buildingmind.rag.query_understanding import understand_query
from buildingmind.retrieval.base import Retriever
from buildingmind.telemetry.base import TelemetryProvider

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        telemetry: TelemetryProvider,
        historical: HistoricalContextProvider,
        llm: LLMProvider,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        top_k: int = 5,
        min_score: float = 0.15,
    ):
        if top_k <= 0 or not 0 <= min_score <= 1:
            raise ValueError("Invalid retrieval limits")
        self.retriever, self.telemetry, self.historical, self.llm = (
            retriever,
            telemetry,
            historical,
            llm,
        )
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.clock, self.top_k, self.min_score = clock, top_k, min_score

    def answer(self, question: str) -> DiagnosticResponse:
        query = understand_query(question)
        if query.floor is None and not query.equipment:
            return DiagnosticResponse(
                question=query.question,
                status="insufficient_evidence",
                evidence_gaps=("Specify a floor or equipment identifier.",),
            )
        try:
            retrieved = self.retriever.search(
                query.question,
                top_k=self.top_k,
                min_score=self.min_score,
                floor=query.floor,
                equipment=query.equipment,
            )
            context = self.context_builder.build(
                query,
                telemetry=self.telemetry.read(query),
                retrieved=tuple(hit.evidence for hit in retrieved),
                historical=self.historical.read(query),
                now=self.clock(),
            )
        except Exception:
            # Do not log exception payloads: providers may include secrets or source content.
            logger.warning("Evidence provider or context validation failed")
            return DiagnosticResponse(
                question=query.question,
                status="provider_error",
                evidence_gaps=("Evidence providers failed or returned invalid data.",),
            )
        empty = AnswerDraft(observations=(), historical_evidence=(), hypotheses=())
        if not context.evidence or detect_conflicts(context):
            return ground_answer(empty, context)
        try:
            prompt = self.prompt_builder.build(context)
            raw = self.llm.generate(prompt)
            draft = AnswerDraft.model_validate(
                raw.model_dump() if isinstance(raw, AnswerDraft) else raw
            )
        except Exception:
            logger.warning("LLM provider or structured output validation failed")
            return DiagnosticResponse(
                question=query.question,
                status="provider_error",
                evidence_gaps=(*context.gaps, "LLM output was unavailable or invalid."),
            )
        try:
            return ground_answer(draft, context)
        except GroundingError:
            logger.warning("Generated answer failed grounding validation")
            response = ground_answer(empty, context)
            return response.model_copy(
                update={
                    "evidence_gaps": (
                        *response.evidence_gaps,
                        "Generated answer failed citation or grounding validation.",
                    )
                }
            )
