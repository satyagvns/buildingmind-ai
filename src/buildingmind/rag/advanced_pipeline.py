"""Advanced retrieval composes the existing v0.4 generation/grounding path."""

import logging
from datetime import UTC, datetime

from buildingmind.query.understanding import QueryUnderstanding
from buildingmind.rag.context_builder import ContextBuilder
from buildingmind.rag.models import DiagnosticResponse, Query
from buildingmind.rag.pipeline import RAGPipeline
from buildingmind.retrieval.base import SearchHit
from buildingmind.retrieval.context_optimizer import ContextOptimizer
from buildingmind.retrieval.evidence_validation import EvidenceValidator
from buildingmind.retrieval.filters import RetrievalFilter
from buildingmind.retrieval.models import RetrievalReport
from buildingmind.retrieval.reranking import rerank

logger = logging.getLogger(__name__)


class AdvancedRAGPipeline:
    def __init__(
        self,
        *,
        retriever,
        access,
        telemetry,
        historical,
        llm,
        building_context=None,
        understanding=None,
        reranker=None,
        validator=None,
        optimizer=None,
        clock=lambda: datetime.now(UTC),
        candidate_count=12,
        final_top_k=5,
    ):
        if candidate_count <= 0 or not 0 < final_top_k <= candidate_count:
            raise ValueError("Invalid candidate limits")
        self.retriever, self.access = retriever, access
        self.telemetry, self.historical, self.llm = telemetry, historical, llm
        self.building_context = building_context
        self.understanding = understanding or QueryUnderstanding()
        self.reranker = reranker
        self.validator = validator or EvidenceValidator()
        self.optimizer = optimizer or ContextOptimizer()
        self.clock = clock
        self.candidate_count, self.final_top_k = candidate_count, final_top_k

    def answer(self, question: str):
        query = self.understanding.understand(question, self.building_context)
        report = RetrievalReport()
        if query.floor is None and not query.equipment_ids:
            return DiagnosticResponse(
                question=question,
                status="insufficient_evidence",
                evidence_gaps=("Specify a floor or equipment identifier.",),
            ), report
        filters = RetrievalFilter(
            access=self.access,
            building=query.building,
            floor=query.floor,
            zone=query.zone,
            system=query.system,
            equipment=query.equipment_ids,
        )
        try:
            raw_report = self.retriever.retrieve(query, filters)
            # Recheck the injected retriever boundary before reranking or returning diagnostics.
            # Never expose a rejected candidate in the public retrieval report.
            if any(not filters.matches(c.evidence) for c in raw_report.candidates):
                report = RetrievalReport(
                    complete=False, errors=("Retriever authorization validation failed.",)
                )
            else:
                report = raw_report
            if not report.complete:
                return DiagnosticResponse(
                    question=question, status="provider_error", evidence_gaps=report.errors
                ), report
            candidates = report.candidates
            basic = Query(question=query.question, floor=query.floor, equipment=query.equipment_ids)
            # Validate the entire fused pool before final top-k can hide contradictory findings.
            validation = self.validator.validate(candidates, basic, now=self.clock())
            candidates = validation.candidates
            if self.reranker:
                selected, latency = rerank(
                    question,
                    candidates,
                    self.reranker,
                    candidate_count=self.candidate_count,
                    top_k=self.final_top_k,
                )
                report = report.model_copy(update={"rerank_seconds": latency})
            else:
                selected = candidates[: self.final_top_k]
            pinned = set(validation.conflict_ids)
            pinned.update(validation.valid_procedure_ids[:1])
            selected_ids = {c.evidence.id for c in selected}
            selected = (
                *selected,
                *(
                    c
                    for c in candidates
                    if c.evidence.id in pinned and c.evidence.id not in selected_ids
                ),
            )
            report = report.model_copy(update={"candidates": selected})
            outer = self

            class SelectedRetriever:
                def search(self, *args, **kwargs):
                    return tuple(
                        SearchHit(evidence=c.evidence, score=min(c.score, 1)) for c in selected
                    )

            class ScopedProvider:
                def __init__(self, provider):
                    self.provider = provider

                def read(self, ignored):
                    return tuple(e for e in self.provider.read(basic) if filters.matches(e))

            class OptimizedBuilder:
                def build(self, ignored, **kwargs):
                    context = ContextBuilder(max_evidence_chars=10**9).build(basic, **kwargs)
                    context = context.model_copy(
                        update={"gaps": (*context.gaps, *validation.issues)}
                    )
                    optimized = outer.optimizer.optimize(context)
                    return optimized

            response = RAGPipeline(
                retriever=SelectedRetriever(),
                telemetry=ScopedProvider(self.telemetry),
                historical=ScopedProvider(self.historical),
                llm=self.llm,
                context_builder=OptimizedBuilder(),
                clock=self.clock,
            ).answer(question)
            if response.status == "answered":
                procedures = tuple(
                    c for c in response.citations if c.evidence_id in validation.valid_procedure_ids
                )
                if not procedures:
                    response = response.model_copy(
                        update={
                            "status": "insufficient_evidence",
                            "hypotheses": (),
                            "verification_steps": (),
                            "evidence_gaps": (
                                *response.evidence_gaps,
                                "Generated answer omitted approved procedure citations.",
                            ),
                        }
                    )
                else:
                    refs = ", ".join(sorted({c.source_id for c in procedures}))
                    response = response.model_copy(
                        update={
                            "verification_steps": (
                                "Have qualified staff verify airflow, actuator/linkage "
                                "and position feedback "
                                f"using approved procedure {refs}.",
                                "Review time-aligned measurements before confirming a root cause.",
                            )
                        }
                    )
            return response, report
        except Exception:
            logger.warning("Advanced retrieval or evidence validation failed")
            return DiagnosticResponse(
                question=question,
                status="provider_error",
                evidence_gaps=("Advanced evidence processing failed safely.",),
            ), report
