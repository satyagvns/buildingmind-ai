"""Evidence contracts shared by ingestion, retrieval, telemetry and RAG."""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from buildingmind.models import Model

EvidenceKind = Literal["telemetry", "manual", "maintenance", "history"]


class Fact(Model):
    subject: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str = ""
    scope: str = "current"
    quote: str = Field(min_length=1)


class Evidence(Model):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    kind: EvidenceKind
    text: str = Field(min_length=1)
    floor: str | None = None
    equipment: tuple[str, ...] = ()
    observed_at: AwareDatetime | None = None
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(ge=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=1, ge=1)
    facts: tuple[Fact, ...] = ()

    @model_validator(mode="after")
    def validate_provenance(self):
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("Character offsets must match text length")
        if self.end_line < self.start_line:
            raise ValueError("Invalid line range")
        if self.kind == "telemetry" and self.observed_at is None:
            raise ValueError("Telemetry requires an aware timestamp")
        if self.kind == "telemetry":
            for fact in self.facts:
                suffix = f" {fact.unit}" if fact.unit else ""
                expected = f"{fact.subject}: {fact.metric} = {fact.value}{suffix}"
                if fact.quote != expected or fact.subject not in self.equipment:
                    raise ValueError("Telemetry facts must match canonical text and equipment")
        if any(fact.quote not in self.text for fact in self.facts):
            raise ValueError("Fact quote must occur verbatim in evidence")
        return self


def is_current(evidence: Evidence, now: datetime, max_age_seconds: float) -> bool:
    if evidence.kind != "telemetry":
        return True
    age = (now - evidence.observed_at).total_seconds()
    return 0 <= age <= max_age_seconds
