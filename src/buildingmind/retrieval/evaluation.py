"""Labeled ranking metrics; duplicate predictions cannot inflate scores."""

import math


def retrieval_metrics(predicted, relevance: dict[str, int], *, k: int):
    if k <= 0 or any(v < 0 for v in relevance.values()):
        raise ValueError("Positive k and nonnegative relevance required")
    full_ranking = tuple(dict.fromkeys(predicted))
    ranked = full_ranking[:k]
    relevant = {key for key, grade in relevance.items() if grade > 0}
    hits = sum(key in relevant for key in ranked)
    dcg = sum((2 ** relevance.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranked))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    return {
        "recall_at_k": hits / len(relevant) if relevant else 0.0,
        "precision_at_k": hits / k,
        "mrr": next((1 / i for i, key in enumerate(full_ranking, 1) if key in relevant), 0.0),
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "hit_rate_at_k": float(hits > 0),
    }
