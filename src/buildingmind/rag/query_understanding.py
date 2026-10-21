"""Deterministic scope extraction; no speculative equipment relationships."""

import re

from buildingmind.rag.models import Query


def understand_query(question: str) -> Query:
    floors = tuple(dict.fromkeys(re.findall(r"\bfloor\s+(\d+)\b", question, re.I)))
    if len(floors) > 1:
        raise ValueError("Ask about one floor at a time")
    equipment = tuple(
        dict.fromkeys(re.findall(r"\b(?:AHU-\d+|VAV-\d+[A-Z]?-\d+)\b", question.upper()))
    )
    return Query(
        question=question.strip(), floor=floors[0] if floors else None, equipment=equipment
    )
