"""Trusted authorization is injected by the caller, never derived from a question."""

from buildingmind.context.models import Evidence
from buildingmind.models import Model


class AccessScope(Model):
    buildings: tuple[str, ...]
    source_ids: tuple[str, ...] | None = None
    access_tags: tuple[str, ...] = ()

    def permits(self, evidence: Evidence) -> bool:
        return (
            evidence.building in self.buildings
            and (self.source_ids is None or evidence.source_id in self.source_ids)
            and set(evidence.access_tags).issubset(self.access_tags)
        )


class RetrievalFilter(Model):
    access: AccessScope
    building: str | None = None
    floor: str | None = None
    zone: str | None = None
    system: str | None = None
    equipment: tuple[str, ...] = ()

    def matches(self, evidence: Evidence) -> bool:
        return (
            self.access.permits(evidence)
            and all(
                value is None or getattr(evidence, name) == value
                for name, value in (
                    ("building", self.building),
                    ("floor", self.floor),
                    ("zone", self.zone),
                    ("system", self.system),
                )
            )
            and (not self.equipment or bool(set(self.equipment) & set(evidence.equipment)))
        )
