"""Provider-independent reranking against the original user question."""

import math
from time import perf_counter
from typing import Protocol

from buildingmind.retrieval.bm25 import tokenize
from buildingmind.retrieval.models import Candidate


class Reranker(Protocol):
    def score(self, question: str, candidates: tuple[Candidate, ...]) -> tuple[float, ...]: ...


class LexicalReranker:
    def score(self, question, candidates):
        tokens = set(tokenize(question))
        identifiers = {t for t in tokens if t.startswith(("ahu-", "vav-"))}
        return tuple(
            float(
                len(tokens & set(tokenize(c.evidence.text)))
                + 2 * len(identifiers & set(tokenize(c.evidence.text)))
            )
            for c in candidates
        )


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, model=None):
        self.model_name, self.model = model_name, model

    def score(self, question, candidates):
        if not candidates:
            return ()
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        return tuple(
            float(v) for v in self.model.predict([(question, c.evidence.text) for c in candidates])
        )


def rerank(question, candidates, provider: Reranker, *, candidate_count=12, top_k=5):
    if candidate_count <= 0 or not 0 < top_k <= candidate_count:
        raise ValueError("Invalid reranking limits")
    started = perf_counter()
    selected = candidates[:candidate_count]
    scores = provider.score(question, selected)
    if len(scores) != len(selected) or any(not math.isfinite(v) for v in scores):
        raise ValueError("Invalid reranker scores")
    ranked = [
        c.model_copy(update={"rerank_score": s}) for c, s in zip(selected, scores, strict=True)
    ]
    return tuple(sorted(ranked, key=lambda c: -c.rerank_score)[:top_k]), perf_counter() - started
