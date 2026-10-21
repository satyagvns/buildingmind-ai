"""Offline example using explicitly synthetic records and a fixed snapshot clock."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from buildingmind.context.mock_provider import MockHistoricalContextProvider
from buildingmind.embeddings.mock_provider import MockEmbeddingProvider
from buildingmind.ingestion.chunker import chunk_document
from buildingmind.ingestion.loader import load_document
from buildingmind.llm.factory import create_llm_provider
from buildingmind.rag.pipeline import RAGPipeline
from buildingmind.retrieval.in_memory import InMemoryVectorStore
from buildingmind.telemetry.mock_provider import MockTelemetryProvider, measurement

DATA = Path(__file__).parent / "data"
QUESTION = "Why is Floor 7 getting warmer even though AHU-07 is running?"


def create_pipeline(*, llm=None) -> RAGPipeline:
    fixture = json.loads((DATA / "telemetry.json").read_text())
    now = datetime.fromisoformat(fixture["observed_at"])
    metadata = {"floor": "7", "equipment": ("AHU-07", "VAV-7E-14")}
    store = InMemoryVectorStore(MockEmbeddingProvider())
    for filename, kind in (("ahu07_manual.md", "manual"), ("maintenance.md", "maintenance")):
        store.add(
            chunk_document(
                load_document(
                    DATA / filename, source_id=f"synthetic:{filename}", kind=kind, **metadata
                )
            )
        )
    history = chunk_document(
        load_document(
            DATA / "history.md", source_id="synthetic:history", kind="history", **metadata
        )
    )
    sensors = tuple(
        measurement(
            **item,
            floor="7",
            observed_at=now,
            related_equipment=("AHU-07",),
        )
        for item in fixture["measurements"]
    )
    return RAGPipeline(
        retriever=store,
        telemetry=MockTelemetryProvider(sensors),
        historical=MockHistoricalContextProvider(history),
        llm=llm or create_llm_provider(),
        clock=lambda: now,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "openai"), default="mock")
    parser.add_argument("--model", help="Explicit model supporting structured Responses output")
    args = parser.parse_args()
    if args.provider == "openai" and not args.model:
        parser.error("--model is required for the optional OpenAI provider")
    provider = (
        create_llm_provider("openai", model=args.model)
        if args.provider == "openai"
        else create_llm_provider()
    )
    answer = create_pipeline(llm=provider).answer(QUESTION)
    print(answer.model_dump_json(indent=2))
    if answer.status == "provider_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
