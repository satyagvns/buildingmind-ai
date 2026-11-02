"""Bounded generation regression checks, not a factual correctness judge."""

import json

from buildingmind.retrieval.filters import AccessScope
from examples.advanced_floor7 import corpus, create_pipeline
from examples.floor7_hvac import QUESTION


def evaluate_generation():
    rows = []
    for case in ("full", "missing-telemetry", "denied-building", "missing-procedure"):
        pipeline = create_pipeline()
        if case == "missing-telemetry":
            pipeline.telemetry.evidence = ()
        elif case == "denied-building":
            pipeline.access = AccessScope(buildings=("Other",))
        elif case == "missing-procedure":
            from buildingmind.retrieval.evidence_validation import EvidencePolicy, EvidenceValidator

            pipeline.validator = EvidenceValidator(
                EvidencePolicy(approved_versions=(("absent", "0"),))
            )
        response, report = pipeline.answer(QUESTION)
        source = {
            e.id: e
            for e in (*corpus(), *pipeline.telemetry.evidence, *pipeline.historical.evidence)
        }
        valid_citations = all(
            c.evidence_id in source and c.quote in source[c.evidence_id].text
            for c in response.citations
        )
        cited = {c.evidence_id for c in response.citations}
        supported = all(
            set(h.evidence_ids) <= cited and h.confidence == "tentative"
            for h in response.hypotheses
        )
        expected_codes = (
            {"airflow_restriction", "damper_tracking_fault"} if case == "full" else set()
        )
        expected_status = "answered" if case == "full" else "insufficient_evidence"
        rows.append(
            {
                "case": case,
                "status": response.status,
                "answer_relevance_rule_codes": {h.code for h in response.hypotheses}
                == expected_codes,
                "citation_validity": valid_citations,
                "grounding_reference_check": supported,
                "evidence_completeness": len(response.observations) == 7
                if case == "full"
                else None,
                "unsupported_hypothesis_count": sum(
                    not set(h.evidence_ids) <= cited for h in response.hypotheses
                ),
                "expected_status": response.status == expected_status,
                "retrieval_complete": report.complete,
            }
        )
    return {
        "limitations": (
            "Extractive attribution and bounded rule checks do not prove source truth, "
            "causal correctness, or general answer relevance."
        ),
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_generation(), indent=2))
