"""Read-only deterministic telemetry snapshots, never building control commands."""

import hashlib
from datetime import datetime

from buildingmind.context.base import matches_scope
from buildingmind.context.models import Evidence, Fact
from buildingmind.rag.models import Query


def measurement(
    subject: str,
    metric: str,
    value: str,
    unit: str,
    *,
    floor: str,
    observed_at: datetime,
    related_equipment: tuple[str, ...] = (),
) -> Evidence:
    text = f"{subject}: {metric} = {value}{(' ' + unit) if unit else ''}"
    source_id = f"telemetry:{subject}:{metric}"
    identity = f"{source_id}:{floor}:{observed_at.isoformat()}:{text}:{related_equipment}"
    fact = Fact(
        subject=subject,
        metric=metric,
        value=value,
        unit=unit,
        scope=observed_at.isoformat(),
        quote=text,
    )
    return Evidence(
        id="sensor-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        source_id=source_id,
        title=f"{subject} {metric}",
        kind="telemetry",
        text=text,
        floor=floor,
        equipment=tuple(dict.fromkeys((subject, *related_equipment))),
        observed_at=observed_at,
        end_char=len(text),
        facts=(fact,),
    )


class MockTelemetryProvider:
    def __init__(self, evidence: tuple[Evidence, ...] = ()):
        if any(e.kind != "telemetry" for e in evidence):
            raise ValueError("Telemetry provider accepts only telemetry evidence")
        self.evidence = evidence

    def read(self, query: Query) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if matches_scope(e, query))
