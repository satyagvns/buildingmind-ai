"""Resolve exact excerpts against the context actually supplied to the model."""

from buildingmind.context.models import Evidence
from buildingmind.rag.models import Citation, Claim


class CitationError(ValueError):
    pass


def resolve_citation(claim: Claim, evidence: tuple[Evidence, ...]) -> Citation:
    matches = [item for item in evidence if item.id == claim.evidence_id]
    if len(matches) != 1:
        raise CitationError("Citation must resolve to exactly one context evidence item")
    item = matches[0]
    offset = item.text.find(claim.quote)
    if not claim.quote.strip() or offset < 0:
        raise CitationError("Citation quote must match evidence verbatim")
    start_line = item.start_line + item.text.count("\n", 0, offset)
    end_line = item.start_line + item.text.count("\n", 0, offset + len(claim.quote) - 1)
    return Citation(
        evidence_id=item.id,
        source_id=item.source_id,
        title=item.title,
        kind=item.kind,
        quote=claim.quote,
        start_char=item.start_char + offset,
        end_char=item.start_char + offset + len(claim.quote),
        start_line=start_line,
        end_line=end_line,
        observed_at=item.observed_at.isoformat() if item.observed_at else None,
    )
