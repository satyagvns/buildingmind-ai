"""Run with python -m evals.run_advanced; source labels resolve to revision-specific chunk IDs."""

import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from buildingmind.query.understanding import QueryUnderstanding
from buildingmind.retrieval.evaluation import retrieval_metrics
from buildingmind.retrieval.filters import RetrievalFilter
from buildingmind.retrieval.reranking import LexicalReranker, rerank
from examples.advanced_floor7 import corpus, create_pipeline


def evaluate(k=3):
    pipeline = create_pipeline()
    cases = json.loads(Path(__file__).with_name("advanced_rag_cases.json").read_text())
    rows = []
    for case in cases:
        query = QueryUnderstanding(expand=False).understand(
            case["question"], pipeline.building_context
        )
        filters = RetrievalFilter(
            access=pipeline.access, building="Tower A", floor="7", equipment=query.equipment_ids
        )
        labels = {e.id: case["relevant_sources"].get(e.source_id, 0) for e in corpus()}
        for mode in ("keyword", "vector", "hybrid", "hybrid_reranked"):
            started = perf_counter()
            report = pipeline.retriever.retrieve(
                query, filters, mode="hybrid" if mode == "hybrid_reranked" else mode
            )
            if not report.complete:
                raise RuntimeError("Evaluation retriever failure")
            candidates, latency = report.candidates, 0.0
            if mode == "hybrid_reranked":
                candidates, latency = rerank(
                    case["question"], candidates, LexicalReranker(), top_k=12
                )
            ids = [c.evidence.id for c in candidates]
            rows.append(
                {
                    "case": case["id"],
                    "mode": mode,
                    "k": k,
                    "predicted_chunk_ids": ids,
                    "relevance": labels,
                    "metrics": retrieval_metrics(ids, labels, k=k),
                    "elapsed_seconds": perf_counter() - started,
                    "rerank_seconds": latency,
                }
            )
    return {
        "fixture": "synthetic; lexical hash embeddings; expansion disabled for comparison",
        "mrr_definition": (
            "reciprocal first-relevant rank in returned ranking; "
            "reranked list limited to candidate_count"
        ),
        "summary": {
            mode: {
                metric: mean(row["metrics"][metric] for row in rows if row["mode"] == mode)
                for metric in rows[0]["metrics"]
            }
            for mode in ("keyword", "vector", "hybrid", "hybrid_reranked")
        },
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
