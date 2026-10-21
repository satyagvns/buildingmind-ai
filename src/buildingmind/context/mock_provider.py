from buildingmind.context.base import matches_scope
from buildingmind.context.models import Evidence
from buildingmind.rag.models import Query


class MockHistoricalContextProvider:
    def __init__(self, evidence: tuple[Evidence, ...] = ()):
        if any(e.kind != "history" for e in evidence):
            raise ValueError("Historical provider accepts only history evidence")
        self.evidence = evidence

    def read(self, query: Query) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if matches_scope(e, query))
