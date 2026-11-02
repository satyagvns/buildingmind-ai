"""Synthetic advanced RAG demo; no external models or building connections."""

import json
from datetime import UTC, datetime
from pathlib import Path

from buildingmind.context.mock_provider import MockHistoricalContextProvider
from buildingmind.embeddings.mock_provider import MockEmbeddingProvider
from buildingmind.ingestion.chunker import chunk_document
from buildingmind.ingestion.models import Document
from buildingmind.llm.mock_provider import MockLLMProvider
from buildingmind.query.understanding import BuildingContext
from buildingmind.rag.advanced_pipeline import AdvancedRAGPipeline
from buildingmind.retrieval.bm25 import BM25Retriever
from buildingmind.retrieval.filters import AccessScope
from buildingmind.retrieval.hybrid import HybridRetriever
from buildingmind.retrieval.in_memory import InMemoryVectorStore
from buildingmind.retrieval.reranking import LexicalReranker
from examples.floor7_hvac import QUESTION
from examples.floor7_hvac import create_pipeline as legacy_pipeline

NOW = datetime(2026, 10, 21, 8, tzinfo=UTC)
DATA = Path(__file__).parent / "data" / "advanced_records.json"


def corpus():
    return tuple(
        chunk
        for record in json.loads(DATA.read_text())
        for chunk in chunk_document(Document.model_validate(record))
    )


def create_pipeline(*, reranking=True):
    records = corpus()
    vector = InMemoryVectorStore(MockEmbeddingProvider())
    vector.add(records)
    legacy = legacy_pipeline()

    def tagged(e):
        return e.model_copy(update={"building": "Tower A", "zone": "East", "system": "HVAC"})

    legacy.telemetry.evidence = tuple(tagged(e) for e in legacy.telemetry.evidence)
    history = tuple(tagged(e) for e in legacy.historical.evidence)
    return AdvancedRAGPipeline(
        retriever=HybridRetriever(keyword=BM25Retriever(records), vector=vector),
        access=AccessScope(buildings=("Tower A",)),
        building_context=BuildingContext(
            building="Tower A",
            floor="7",
            zone="East",
            system="HVAC",
            equipment=("AHU-07", "VAV-7E-14"),
        ),
        telemetry=legacy.telemetry,
        historical=MockHistoricalContextProvider(history),
        llm=MockLLMProvider(),
        reranker=LexicalReranker() if reranking else None,
        clock=lambda: NOW,
    )


if __name__ == "__main__":
    answer, report = create_pipeline().answer(QUESTION)
    print(
        json.dumps(
            {"answer": answer.model_dump(mode="json"), "retrieval": report.model_dump(mode="json")},
            indent=2,
        )
    )
