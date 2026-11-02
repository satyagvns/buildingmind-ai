from pydantic import Field

from buildingmind.context.models import Evidence
from buildingmind.models import Model


class Ranking(Model):
    backend: str
    query_index: int = Field(ge=0)
    rank: int = Field(ge=1)
    score: float


class Candidate(Model):
    evidence: Evidence
    score: float = Field(ge=0)
    rankings: tuple[Ranking, ...] = ()
    rerank_score: float | None = None


class RetrievalReport(Model):
    candidates: tuple[Candidate, ...] = ()
    errors: tuple[str, ...] = ()
    complete: bool = True
    elapsed_seconds: float = Field(default=0, ge=0)
    rerank_seconds: float = Field(default=0, ge=0)
