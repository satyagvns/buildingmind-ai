from typing import Protocol

from pydantic import Field

from buildingmind.context.models import Evidence
from buildingmind.models import Model


class SearchHit(Model):
    evidence: Evidence
    score: float = Field(ge=-1, le=1)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.15,
        floor: str | None = None,
        equipment: tuple[str, ...] = (),
    ) -> tuple[SearchHit, ...]: ...
