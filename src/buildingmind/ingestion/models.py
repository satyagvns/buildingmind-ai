from typing import Literal

from pydantic import Field, model_validator

from buildingmind.context.models import Fact
from buildingmind.models import Model


class Document(Model):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: Literal["manual", "maintenance", "history"] = "manual"
    floor: str | None = None
    equipment: tuple[str, ...] = ()
    facts: tuple[Fact, ...] = ()

    @model_validator(mode="after")
    def validate_facts(self):
        if any(f.quote not in self.text for f in self.facts):
            raise ValueError("Fact quote must occur verbatim in document")
        return self
