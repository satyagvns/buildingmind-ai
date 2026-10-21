"""Validated inputs, provider drafts, and public diagnostic responses."""

from typing import Literal

from pydantic import Field

from buildingmind.context.models import Evidence
from buildingmind.models import Model


class Query(Model):
    question: str = Field(min_length=1, max_length=2000)
    floor: str | None = None
    equipment: tuple[str, ...] = ()


class Context(Model):
    query: Query
    evidence: tuple[Evidence, ...] = ()
    constraints: tuple[str, ...] = ("Read-only building operations; never issue control commands.",)
    gaps: tuple[str, ...] = ()


class Claim(Model):
    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


HypothesisCode = Literal["damper_tracking_fault", "airflow_restriction"]


class HypothesisDraft(Model):
    code: HypothesisCode
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class AnswerDraft(Model):
    # Required fields also satisfy strict structured-output schemas.
    observations: tuple[Claim, ...]
    historical_evidence: tuple[Claim, ...]
    hypotheses: tuple[HypothesisDraft, ...]


class Citation(Model):
    evidence_id: str
    source_id: str
    title: str
    kind: str
    quote: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    observed_at: str | None


class GroundedClaim(Model):
    text: str
    citation: Citation


class Hypothesis(Model):
    code: HypothesisCode
    text: str
    confidence: Literal["tentative"] = "tentative"
    evidence_ids: tuple[str, ...]


class Conflict(Model):
    subject: str
    metric: str
    scope: str
    unit: str
    values: tuple[str, ...]
    citations: tuple[Citation, ...]


class DiagnosticResponse(Model):
    question: str
    status: Literal["answered", "insufficient_evidence", "conflicting_evidence", "provider_error"]
    observations: tuple[GroundedClaim, ...] = ()
    historical_evidence: tuple[GroundedClaim, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    verification_steps: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    evidence_gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = (
        "Diagnostic hypotheses are tentative; evidence does not establish a confirmed root cause.",
    )
