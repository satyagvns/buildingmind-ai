"""Small synthetic regression evaluations, not a production quality benchmark."""

import json
from pathlib import Path

import pytest

from buildingmind.telemetry.mock_provider import measurement
from examples.floor7_hvac import QUESTION

EVALS = Path(__file__).resolve().parents[2] / "evals"


@pytest.mark.parametrize(
    "case", json.loads((EVALS / "retrieval_cases.json").read_text()), ids=lambda case: case["id"]
)
def test_retrieval_top1_and_scope(pipeline, case):
    hits = pipeline.retriever.search(
        case["query"], floor=case["floor"], equipment=tuple(case["equipment"]), top_k=1
    )
    if case["expected_top_source"] is None:
        assert hits == ()
    else:
        assert hits[0].evidence.source_id == case["expected_top_source"]


@pytest.mark.parametrize(
    "case", json.loads((EVALS / "floor7_cases.json").read_text()), ids=lambda case: case["id"]
)
def test_diagnostic_regressions(pipeline, case):
    changed = []
    for evidence in pipeline.telemetry.evidence:
        fact = evidence.facts[0]
        changed.append(
            measurement(
                fact.subject,
                fact.metric,
                case["changes"].get(fact.metric, fact.value),
                fact.unit,
                floor=evidence.floor,
                related_equipment=("AHU-07",),
                observed_at=evidence.observed_at,
            )
        )
    pipeline.telemetry.evidence = tuple(changed)
    answer = pipeline.answer(QUESTION)
    assert answer.status == case["status"]
    assert {hypothesis.code for hypothesis in answer.hypotheses} == set(case["hypotheses"])
