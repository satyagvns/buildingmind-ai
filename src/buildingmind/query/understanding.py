"""Deterministic rewriting; contextual relationships must come from validated adapters."""

import re
from typing import Protocol

from buildingmind.models import Model
from buildingmind.rag.query_understanding import understand_query


class BuildingContext(Model):
    building: str
    floor: str | None = None
    zone: str | None = None
    system: str | None = None
    equipment: tuple[str, ...] = ()


class StructuredQuery(Model):
    question: str
    intent: str
    building: str | None = None
    floor: str | None = None
    zone: str | None = None
    system: str | None = None
    original_equipment_ids: tuple[str, ...] = ()
    equipment_ids: tuple[str, ...] = ()
    semantic_queries: tuple[str, ...]
    keyword_queries: tuple[str, ...]


class QueryRewriter(Protocol):
    def rewrite(self, query: StructuredQuery) -> StructuredQuery: ...


class QueryUnderstanding:
    def __init__(
        self, *, expand: bool = True, max_queries: int = 3, rewriter: QueryRewriter | None = None
    ):
        if not 1 <= max_queries <= 8:
            raise ValueError("Require 1 to 8 queries per backend")
        self.expand, self.max_queries, self.rewriter = expand, max_queries, rewriter

    def understand(self, question: str, context: BuildingContext | None = None) -> StructuredQuery:
        basic = understand_query(question)
        originals = tuple(
            dict.fromkeys(re.findall(r"\b(?:AHU-\d+|VAV-\d+[A-Z]?-\d+)\b", question, re.I))
        )
        if context and basic.floor and context.floor and basic.floor != context.floor:
            raise ValueError("Question and validated context floors disagree")
        equipment = tuple(
            dict.fromkeys((*basic.equipment, *(context.equipment if context else ())))
        )
        hvac = bool(equipment) or bool(re.search(r"hvac|warm|cool|airflow", question, re.I))
        expansions = (
            "HVAC cooling performance running AHU low VAV airflow damper actuator feedback",
        )
        result = StructuredQuery(
            question=basic.question,
            intent="hvac_diagnosis" if hvac else "information",
            building=context.building if context else None,
            floor=basic.floor or (context.floor if context else None),
            zone=context.zone if context else None,
            system=(context.system if context else None) or ("HVAC" if hvac else None),
            original_equipment_ids=originals,
            equipment_ids=equipment,
            semantic_queries=(basic.question, *(expansions if self.expand and hvac else ())),
            keyword_queries=(basic.question, *(equipment if self.expand else ())),
        )
        if self.rewriter:
            rewritten = StructuredQuery.model_validate(self.rewriter.rewrite(result).model_dump())
            # Rewriting may change retrieval text, never trusted scope or the actual question.
            result = result.model_copy(
                update={
                    "semantic_queries": rewritten.semantic_queries,
                    "keyword_queries": rewritten.keyword_queries,
                }
            )
        return result.model_copy(
            update={
                "semantic_queries": result.semantic_queries[: self.max_queries],
                "keyword_queries": result.keyword_queries[: self.max_queries],
            }
        )
