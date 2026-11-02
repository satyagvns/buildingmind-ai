"""Fuse ranks only; BM25 and cosine scores remain uncalibrated provenance."""

import math

from buildingmind.retrieval.models import Candidate, Ranking


def reciprocal_rank_fusion(
    lists, *, k: float = 60, weights: dict[str, float] | None = None
) -> tuple[Candidate, ...]:
    if not math.isfinite(k) or k < 0:
        raise ValueError("Invalid rank smoothing constant")
    weights = weights or {}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Invalid fusion weight")
    entries, scores, rankings = {}, {}, {}
    for backend, query_index, hits in lists:
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = hit.evidence.id
            if key in entries and entries[key] != hit.evidence:
                raise ValueError("Evidence ID collision across retrievers")
            if key in seen:
                continue
            seen.add(key)
            entries[key] = hit.evidence
            scores[key] = scores.get(key, 0) + weights.get(backend, 1) / (k + rank)
            rankings.setdefault(key, []).append(
                Ranking(backend=backend, query_index=query_index, rank=rank, score=hit.score)
            )
    return tuple(
        Candidate(evidence=entries[key], score=scores[key], rankings=tuple(rankings[key]))
        for key in sorted(entries, key=lambda key: (-scores[key], key))
    )
