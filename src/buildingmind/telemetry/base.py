from typing import Protocol

from buildingmind.context.models import Evidence
from buildingmind.rag.models import Query


class TelemetryProvider(Protocol):
    def read(self, query: Query) -> tuple[Evidence, ...]: ...
