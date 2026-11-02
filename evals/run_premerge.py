"""Targeted retrieval comparison and repeated warm latency measurements; offline."""

import json
import math
import platform
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from buildingmind.embeddings.mock_provider import MockEmbeddingProvider
from buildingmind.ingestion.chunker import chunk_document
from buildingmind.ingestion.models import Document
from buildingmind.query.understanding import QueryUnderstanding
from buildingmind.retrieval.bm25 import BM25Retriever
from buildingmind.retrieval.evaluation import retrieval_metrics
from buildingmind.retrieval.filters import AccessScope, RetrievalFilter
from buildingmind.retrieval.hybrid import HybridRetriever
from buildingmind.retrieval.in_memory import InMemoryVectorStore
from buildingmind.retrieval.reranking import LexicalReranker, rerank
from examples.advanced_floor7 import corpus

HERE = Path(__file__).parent
MODES = ("keyword", "vector", "hybrid", "hybrid_reranked")


def review_corpus():
    return (
        *corpus(),
        *(
            e
            for d in json.loads((HERE / "premerge_records.json").read_text())
            for e in chunk_document(Document.model_validate(d))
        ),
    )


def make_retriever(records):
    vector = InMemoryVectorStore(MockEmbeddingProvider())
    vector.add(records)
    return HybridRetriever(keyword=BM25Retriever(records), vector=vector, top_k=24)


def evaluate_premerge(*, repeats=30, warmups=5, k=3):
    if repeats < 1 or warmups < 0:
        raise ValueError("Invalid timing sample counts")
    records = review_corpus()
    retriever = make_retriever(records)
    cases = json.loads((HERE / "premerge_cases.json").read_text())
    rows, timing = [], {mode: [] for mode in MODES}
    for case in cases:
        query = QueryUnderstanding(expand=False).understand(case["question"])
        filters = RetrievalFilter(
            access=AccessScope(buildings=("Tower A",)),
            building="Tower A",
            floor="7",
            equipment=query.equipment_ids,
        )
        labels = {
            e.id: (
                case["relevant_sources"].get(e.source_id, 0)
                if case.get("required_relevance_quotes", {}).get(e.source_id, "") in e.text
                else 0
            )
            for e in records
            if filters.matches(e)
        }
        assert all(
            any(e.source_id == source and filters.matches(e) for e in records)
            for source in case["relevant_sources"]
        ), "Relevant source not in scoped corpus"
        per_mode = {}
        # Rotate execution order each repeat to reduce systematic warm/order bias.
        for sample in range(warmups + repeats):
            order = MODES[sample % len(MODES) :] + MODES[: sample % len(MODES)]
            for mode in order:
                started = perf_counter()
                report = retriever.retrieve(
                    query, filters, mode="hybrid" if mode == "hybrid_reranked" else mode
                )
                if not report.complete:
                    raise RuntimeError("Incomplete benchmark retrieval")
                candidates, rerank_time = report.candidates, 0.0
                if mode == "hybrid_reranked":
                    candidates, rerank_time = rerank(
                        query.question, candidates, LexicalReranker(), candidate_count=24, top_k=24
                    )
                elapsed = perf_counter() - started
                if sample >= warmups:
                    timing[mode].append(
                        {
                            "case": case["id"],
                            "elapsed_ms": elapsed * 1000,
                            "rerank_ms": rerank_time * 1000,
                        }
                    )
                per_mode[mode] = candidates
        for mode, candidates in per_mode.items():
            ids = [c.evidence.id for c in candidates]
            rows.append(
                {
                    "case": case["id"],
                    "category": case["category"],
                    "mode": mode,
                    "metrics": retrieval_metrics(ids, labels, k=k),
                    "relevance": labels,
                    "ranking": [
                        {
                            "chunk_id": c.evidence.id,
                            "source_id": c.evidence.source_id,
                            "rankings": [r.model_dump() for r in c.rankings],
                        }
                        for c in candidates
                    ],
                }
            )

    def percentiles(values):
        ordered = sorted(values)
        return {
            "median_ms": median(ordered),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "mean_ms": mean(ordered),
            "samples": len(ordered),
        }

    comparisons = []
    for case in cases:

        def metrics(mode, case_id=case["id"]):
            return next(
                row["metrics"] for row in rows if row["case"] == case_id and row["mode"] == mode
            )

        comparisons.append(
            {
                "case": case["id"],
                "category": case["category"],
                "hybrid_minus_vector": {
                    key: metrics("hybrid")[key] - metrics("vector")[key]
                    for key in metrics("vector")
                },
            }
        )
    return {
        "k": k,
        "repeats": repeats,
        "warmups_per_case": warmups,
        "environment": platform.python_version(),
        "corpus_chunks": len(records),
        "limitations": (
            "Synthetic labels; lexical hash vector test double; warm local timings exclude "
            "indexing, network and generation. No tuning to guarantee hybrid wins."
        ),
        "summary": {
            mode: {
                m: mean(r["metrics"][m] for r in rows if r["mode"] == mode)
                for m in rows[0]["metrics"]
            }
            for mode in MODES
        },
        "comparisons": comparisons,
        "latency": {
            mode: percentiles([t["elapsed_ms"] for t in samples])
            for mode, samples in timing.items()
        },
        "reranking_only": percentiles([t["rerank_ms"] for t in timing["hybrid_reranked"]]),
        "cases": rows,
        "timing_samples": timing,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_premerge(), indent=2))
