"""Conservative extractive grounding and a bounded, auditable HVAC rule set.

Free-form model reasoning is intentionally excluded from public factual claims.
Rule support establishes plausibility, never a confirmed physical root cause.
"""

from collections import defaultdict
from datetime import UTC
from decimal import Decimal, InvalidOperation

from buildingmind.rag.citations import CitationError, resolve_citation
from buildingmind.rag.diagnostics import diagnostic_candidates
from buildingmind.rag.models import (
    AnswerDraft,
    Claim,
    Conflict,
    Context,
    DiagnosticResponse,
    GroundedClaim,
    Hypothesis,
    HypothesisDraft,
)


class GroundingError(ValueError):
    pass


HYPOTHESIS_TEXT = {
    "damper_tracking_fault": (
        "A damper actuator, linkage, or position-feedback fault may explain the "
        "command/position mismatch and contribute to reduced cooling."
    ),
    "airflow_restriction": (
        "Reduced zone air delivery may be limiting cooling; a restriction or control "
        "fault remains a hypothesis even while the AHU is running."
    ),
}

VERIFICATION_STEPS = (
    "Verify the airflow sensor reading against an independent measurement.",
    "Have qualified staff inspect damper actuator/linkage and compare position feedback "
    "with the commanded position under the site's maintenance procedure.",
    "Review maintenance records and time-aligned trends before confirming a root cause.",
)


def detect_conflicts(context: Context) -> tuple[Conflict, ...]:
    groups = defaultdict(list)
    for evidence in context.evidence:
        for fact in evidence.facts:
            # Telemetry timestamps define the comparison period; prior readings are not conflicts.
            scope = (
                evidence.observed_at.astimezone(UTC).isoformat()
                if evidence.kind == "telemetry"
                else fact.scope
            )
            groups[(fact.subject, fact.metric, scope, fact.unit)].append((evidence, fact))

    def normalized(value: str):
        try:
            number = Decimal(value)
            if number.is_finite():
                return number
        except InvalidOperation:
            pass
        return value.strip().casefold()

    conflicts = []
    for (subject, metric, scope, unit), items in sorted(groups.items()):
        values = tuple(sorted({f.value for _, f in items}))
        if len({normalized(value) for value in values}) > 1:
            citations = tuple(
                dict.fromkeys(
                    resolve_citation(Claim(evidence_id=e.id, quote=f.quote), context.evidence)
                    for e, f in items
                )
            )
            conflicts.append(
                Conflict(
                    subject=subject,
                    metric=metric,
                    scope=scope,
                    unit=unit,
                    values=values,
                    citations=citations,
                )
            )
    return tuple(conflicts)


def supported_hypotheses(context: Context) -> tuple[HypothesisDraft, ...]:
    return () if detect_conflicts(context) else diagnostic_candidates(context)


def ground_answer(draft: AnswerDraft, context: Context) -> DiagnosticResponse:
    draft = AnswerDraft.model_validate(draft.model_dump())
    observations, historical = [], []
    for claims, output, telemetry in (
        (draft.observations, observations, True),
        (draft.historical_evidence, historical, False),
    ):
        for claim in claims:
            try:
                citation = resolve_citation(claim, context.evidence)
            except CitationError as exc:
                raise GroundingError(str(exc)) from exc
            if (citation.kind == "telemetry") != telemetry:
                raise GroundingError("Evidence was placed in the wrong response category")
            if telemetry:
                source = next(e for e in context.evidence if e.id == claim.evidence_id)
                if claim.quote != source.text:
                    raise GroundingError(
                        "Telemetry observations must preserve the complete measurement"
                    )
            output.append(GroundedClaim(text=claim.quote, citation=citation))
    conflicts = detect_conflicts(context)
    allowed = supported_hypotheses(context)
    hypotheses = []
    for proposed in draft.hypotheses:
        if not any(
            proposed.code == item.code and set(proposed.evidence_ids) == set(item.evidence_ids)
            for item in allowed
        ):
            raise GroundingError("Hypothesis lacks validated measurement support")
        observation_ids = {claim.citation.evidence_id for claim in observations}
        if not set(proposed.evidence_ids) <= observation_ids:
            raise GroundingError("Hypothesis support must also appear in cited observations")
        hypotheses.append(
            Hypothesis(
                code=proposed.code,
                text=HYPOTHESIS_TEXT[proposed.code],
                evidence_ids=proposed.evidence_ids,
            )
        )
    hypotheses = tuple(dict.fromkeys(hypotheses))
    citations = [claim.citation for claim in (*observations, *historical)]
    for hypothesis in hypotheses:
        for evidence_id in hypothesis.evidence_ids:
            evidence = next(e for e in context.evidence if e.id == evidence_id)
            citations.append(
                resolve_citation(
                    Claim(evidence_id=evidence_id, quote=evidence.text), context.evidence
                )
            )
    citations.extend(citation for conflict in conflicts for citation in conflict.citations)
    gaps = list(context.gaps)
    if not hypotheses:
        gaps.append(
            "Generated answer did not include a supported diagnostic hypothesis."
            if allowed
            else "Measurements do not support a validated HVAC hypothesis in the current rule set."
        )
    has_documents = any(c.citation.kind in {"manual", "maintenance"} for c in historical)
    if not has_documents and any(e.kind in {"manual", "maintenance"} for e in context.evidence):
        gaps.append("Generated answer did not include cited manual or maintenance evidence.")
    status = "answered" if hypotheses and has_documents else "insufficient_evidence"
    if conflicts:
        status = "conflicting_evidence"
    return DiagnosticResponse(
        question=context.query.question,
        status=status,
        observations=tuple(dict.fromkeys(observations)),
        historical_evidence=tuple(dict.fromkeys(historical)),
        hypotheses=hypotheses,
        verification_steps=(
            VERIFICATION_STEPS
            if hypotheses
            else (
                "Collect fresh, time-aligned telemetry and relevant equipment records "
                "before diagnosis.",
            )
        ),
        citations=tuple(dict.fromkeys(citations)),
        conflicts=conflicts,
        evidence_gaps=tuple(dict.fromkeys(gaps)),
    )
