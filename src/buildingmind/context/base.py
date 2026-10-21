from typing import Protocol

from buildingmind.context.models import Evidence
from buildingmind.rag.models import Query


class HistoricalContextProvider(Protocol):
    def read(self, query: Query) -> tuple[Evidence, ...]: ...


def matches_scope(evidence: Evidence, query: Query) -> bool:
    if query.floor is None and not query.equipment:
        return False
    return (query.floor is None or evidence.floor == query.floor) and (
        not query.equipment or bool(set(query.equipment) & set(evidence.equipment))
    )
