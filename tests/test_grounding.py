from datetime import UTC, datetime, timedelta, timezone

import pytest

from buildingmind.rag.grounding import GroundingError, detect_conflicts, ground_answer
from buildingmind.rag.models import AnswerDraft, Context
from buildingmind.rag.prompt_builder import PromptBuilder
from buildingmind.rag.query_understanding import understand_query
from buildingmind.telemetry.mock_provider import measurement
from examples.floor7_hvac import QUESTION


def test_same_reading_format_and_timezone_do_not_conflict():
    now = datetime(2026, 10, 21, 8, tzinfo=UTC)
    a = measurement("AHU-07", "damper_actual", "31", "%", floor="7", observed_at=now)
    b = measurement(
        "AHU-07",
        "damper_actual",
        "31.0",
        "%",
        floor="7",
        observed_at=now.astimezone(timezone(timedelta(hours=5, minutes=30))),
    )
    context = Context(query=understand_query("Floor 7 AHU-07"), evidence=(a, b))
    assert detect_conflicts(context) == ()
    c = measurement("AHU-07", "damper_actual", "90", "%", floor="7", observed_at=b.observed_at)
    conflicts = detect_conflicts(context.model_copy(update={"evidence": (a, c)}))
    assert len(conflicts) == 1


def test_same_equipment_different_timestamps_are_not_conflicting():
    now = datetime(2026, 10, 21, 8, tzinfo=UTC)
    a = measurement("AHU-07", "damper_actual", "31", "%", floor="7", observed_at=now)
    b = measurement(
        "AHU-07", "damper_actual", "90", "%", floor="7", observed_at=now - timedelta(minutes=1)
    )
    assert detect_conflicts(Context(query=understand_query("AHU-07"), evidence=(a, b))) == ()


def test_hypothesis_must_also_expose_supporting_observations(pipeline):
    real = pipeline.llm
    contexts = []

    class Capture:
        def generate(self, prompt):
            contexts.append(prompt.context)
            return real.generate(prompt)

    pipeline.llm = Capture()
    assert pipeline.answer(QUESTION).status == "answered"
    draft = real.generate(PromptBuilder().build(contexts[0]))
    incomplete = AnswerDraft(
        observations=(), historical_evidence=draft.historical_evidence, hypotheses=draft.hypotheses
    )
    with pytest.raises(GroundingError, match="cited observations"):
        ground_answer(incomplete, contexts[0])
