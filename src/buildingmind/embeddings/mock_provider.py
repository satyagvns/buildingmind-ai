"""Offline lexical hash vectors. A test double, not a learned semantic model."""

import hashlib
import math
import re


class MockEmbeddingProvider:
    def __init__(self, dimensions: int = 1024):
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        self.dimensions = dimensions

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8])
                vector[index % self.dimensions] += 1
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append(tuple(value / norm if norm else 0.0 for value in vector))
        return tuple(vectors)
