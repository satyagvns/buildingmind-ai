"""Bounded evidence selection with provenance, freshness and scope checks."""

from datetime import datetime

from buildingmind.context.base import matches_scope
from buildingmind.context.models import Evidence, is_current
from buildingmind.rag.models import Context, Query


class ContextBuilder:
    def __init__(self, *, max_evidence_chars: int = 24000, max_age_seconds: float = 900):
        if max_evidence_chars < 2 or not 0 < max_age_seconds < float("inf"):
            raise ValueError("Invalid context limits")
        self.max_evidence_chars = max_evidence_chars
        self.max_age_seconds = max_age_seconds

    def build(
        self,
        query: Query,
        *,
        telemetry: tuple[Evidence, ...],
        retrieved: tuple[Evidence, ...],
        historical: tuple[Evidence, ...],
        now: datetime,
    ) -> Context:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Clock must return a timezone-aware datetime")
        if any(e.kind != "telemetry" for e in telemetry):
            raise ValueError("Telemetry channel contains non-telemetry evidence")
        if any(e.kind == "telemetry" for e in retrieved):
            raise ValueError("Retrieval channel cannot supply live telemetry")
        if any(e.kind != "history" for e in historical):
            raise ValueError("Historical channel contains non-history evidence")
        chosen, gaps = [], []
        identities = {}
        used = 2  # JSON array brackets; individual model JSON includes provenance.
        for raw in (*telemetry, *retrieved, *historical):
            # Revalidate at provider boundaries (also rejects malformed model_copy updates).
            evidence = Evidence.model_validate(raw.model_dump())
            if evidence.id in identities:
                if evidence != identities[evidence.id]:
                    raise ValueError("Evidence ID collision")
                continue
            identities[evidence.id] = evidence
            if not matches_scope(evidence, query):
                gaps.append("Evidence outside the requested scope was excluded.")
                continue
            if not is_current(evidence, now, self.max_age_seconds):
                gaps.append("Stale or future-dated telemetry was excluded.")
                continue
            cost = len(evidence.model_dump_json()) + (1 if chosen else 0)
            if used + cost > self.max_evidence_chars:
                gaps.append("Evidence was omitted because the context budget was reached.")
                continue
            chosen.append(evidence)
            used += cost
        if not any(e.kind == "telemetry" for e in chosen):
            gaps.append("No fresh telemetry is available for the requested scope.")
        if not any(e.kind in {"manual", "maintenance"} for e in chosen):
            gaps.append("No relevant manual or maintenance evidence is available.")
        if not any(e.kind == "history" for e in chosen):
            gaps.append("No historical operational context is available.")
        return Context(query=query, evidence=tuple(chosen), gaps=tuple(dict.fromkeys(gaps)))
