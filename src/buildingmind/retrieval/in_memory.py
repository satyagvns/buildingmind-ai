"""Injectable cosine vector index with atomic upserts and strict scope filters."""

import math

from buildingmind.context.models import Evidence
from buildingmind.embeddings.base import EmbeddingProvider
from buildingmind.retrieval.base import SearchHit


class InMemoryVectorStore:
    def __init__(self, embeddings: EmbeddingProvider):
        self.embeddings = embeddings
        self._entries: dict[str, tuple[Evidence, tuple[float, ...]]] = {}
        self._dimensions: int | None = None

    @staticmethod
    def _normalize(vector):
        if not vector or any(not math.isfinite(v) for v in vector):
            raise ValueError("Embeddings must be nonempty finite vectors")
        norm = math.hypot(*vector)
        return tuple(v / norm if norm else 0.0 for v in vector)

    def add(self, evidence: tuple[Evidence, ...]) -> None:
        if not evidence:
            return
        if len({e.id for e in evidence}) != len(evidence):
            raise ValueError("Duplicate evidence IDs in batch")
        if any(e.id in self._entries and self._entries[e.id][0] != e for e in evidence):
            raise ValueError("Existing evidence ID cannot be reused for different content")
        vectors = self.embeddings.embed(tuple(e.text for e in evidence))
        if len(vectors) != len(evidence):
            raise ValueError("Embedding count mismatch")
        normalized = tuple(self._normalize(v) for v in vectors)
        dimensions = self._dimensions or len(normalized[0])
        if any(len(v) != dimensions for v in normalized):
            raise ValueError("Embedding dimension mismatch")
        self._entries.update({e.id: (e, v) for e, v in zip(evidence, normalized, strict=True)})
        self._dimensions = dimensions

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.15,
        floor: str | None = None,
        equipment: tuple[str, ...] = (),
    ) -> tuple[SearchHit, ...]:
        if top_k <= 0 or not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("Invalid search limits")
        if not query.strip() or not self._entries:
            return ()
        vectors = self.embeddings.embed((query,))
        if len(vectors) != 1:
            raise ValueError("Query embedding count mismatch")
        vector = self._normalize(vectors[0])
        if len(vector) != self._dimensions:
            raise ValueError("Query embedding dimension mismatch")
        hits = []
        for evidence, stored in self._entries.values():
            if floor is not None and evidence.floor != floor:
                continue
            if equipment and not set(equipment).intersection(evidence.equipment):
                continue
            score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, stored, strict=True))))
            if score >= min_score and score > 0:
                hits.append(SearchHit(evidence=evidence, score=score))
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.evidence.id))[:top_k])
