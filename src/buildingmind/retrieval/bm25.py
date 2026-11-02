"""Small BM25 index with technical identifier tokens and early access filtering."""

import math
import re
from collections import Counter

from pydantic import Field

from buildingmind.context.models import Evidence
from buildingmind.models import Model
from buildingmind.retrieval.filters import RetrievalFilter


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold()))


class KeywordHit(Model):
    evidence: Evidence
    score: float = Field(ge=0)


class BM25Retriever:
    def __init__(self, evidence: tuple[Evidence, ...] = (), *, k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        self._entries: dict[str, Evidence] = {}
        self.add(evidence)

    def add(self, evidence: tuple[Evidence, ...]) -> None:
        if len({e.id for e in evidence}) != len(evidence):
            raise ValueError("Duplicate evidence IDs")
        if any(e.id in self._entries and self._entries[e.id] != e for e in evidence):
            raise ValueError("Evidence ID collision")
        self._entries.update({e.id: e for e in evidence})

    def search(
        self, query: str, *, filters: RetrievalFilter, top_k: int = 5
    ) -> tuple[KeywordHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = set(tokenize(query))
        docs = [
            (e, Counter(tokenize(e.text))) for e in self._entries.values() if filters.matches(e)
        ]
        if not terms or not docs:
            return ()
        avg = sum(sum(c.values()) for _, c in docs) / len(docs) or 1
        df = {t: sum(t in c for _, c in docs) for t in terms}
        hits = []
        for evidence, counts in docs:
            length = sum(counts.values())
            score = sum(
                math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                * counts[t]
                * (self.k1 + 1)
                / (counts[t] + self.k1 * (1 - self.b + self.b * length / avg))
                for t in terms
                if counts[t]
            )
            if score > 0:
                hits.append(KeywordHit(evidence=evidence, score=score))
        return tuple(sorted(hits, key=lambda h: (-h.score, h.evidence.id))[:top_k])
