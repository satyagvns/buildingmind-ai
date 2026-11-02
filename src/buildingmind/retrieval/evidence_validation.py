"""Explicit source lifecycle policy; newest is not automatically authoritative."""

from datetime import datetime

from pydantic import Field

from buildingmind.models import Model
from buildingmind.rag.grounding import detect_conflicts
from buildingmind.rag.models import Context, Query
from buildingmind.retrieval.models import Candidate


class EvidencePolicy(Model):
    minimum_authority: int = Field(default=50, ge=0, le=100)
    require_metadata: bool = True
    approved_versions: tuple[tuple[str, str], ...] = ()


class ValidationResult(Model):
    candidates: tuple[Candidate, ...]
    issues: tuple[str, ...]
    conflict_ids: tuple[str, ...] = ()
    valid_procedure_ids: tuple[str, ...] = ()


class EvidenceValidator:
    def __init__(self, policy: EvidencePolicy | None = None):
        self.policy = policy or EvidencePolicy()

    def validate(self, candidates, query: Query, *, now: datetime):
        issues, valid, seen, unique = [], [], {}, []
        for c in candidates:
            e = c.evidence
            if e.id in seen:
                if seen[e.id] != e:
                    raise ValueError("Evidence ID collision")
                issues.append(f"{e.source_id}: duplicate evidence")
                continue
            seen[e.id] = e
            unique.append(c)
            missing = not e.version or e.updated_at is None or e.effective_at is None
            if missing:
                issues.append(f"{e.source_id}: missing version/date metadata")
            approved = (
                e.approval == "approved"
                and e.authority >= self.policy.minimum_authority
                and e.effective_at is not None
                and e.effective_at <= now
                and e.updated_at is not None
                and e.updated_at <= now
                and (
                    not self.policy.approved_versions
                    or (e.source_id, e.version) in self.policy.approved_versions
                )
            )
            if not approved:
                issues.append(f"{e.source_id}: not an authoritative current approved source")
            if query.equipment and not set(query.equipment).intersection(e.equipment):
                issues.append(f"{e.source_id}: equipment relevance not established")
                approved = False
            if (
                e.kind == "maintenance"
                and approved
                and (not missing or not self.policy.require_metadata)
            ):
                valid.append(e.id)
        context = Context(query=query, evidence=tuple(seen.values()))
        conflicts = detect_conflicts(context)
        ids = tuple(
            dict.fromkeys(c.evidence_id for conflict in conflicts for c in conflict.citations)
        )
        if conflicts:
            issues.append("Contradictory structured evidence retained; diagnosis must abstain.")
        return ValidationResult(
            candidates=tuple(unique),
            issues=tuple(issues),
            conflict_ids=ids,
            valid_procedure_ids=tuple(valid),
        )
