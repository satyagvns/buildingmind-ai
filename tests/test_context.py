from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from buildingmind.context.models import Evidence, Fact
from buildingmind.ingestion.chunker import chunk_document
from buildingmind.ingestion.models import Document
from buildingmind.rag.context_builder import ContextBuilder
from buildingmind.rag.query_understanding import understand_query
from buildingmind.telemetry.mock_provider import MockTelemetryProvider, measurement

NOW = datetime(2026, 10, 21, 8, tzinfo=UTC)


def sensor(**changes):
    args = dict(
        subject="VAV-7E-14",
        metric="damper_actual",
        value="31",
        unit="%",
        floor="7",
        related_equipment=("AHU-07",),
        observed_at=NOW,
    )
    args.update(changes)
    return measurement(**args)


def build(*, telemetry=(), retrieved=(), historical=(), builder=None, now=NOW):
    return (builder or ContextBuilder()).build(
        understand_query("Why is Floor 7 warmer with AHU-07 running?"),
        telemetry=telemetry,
        retrieved=retrieved,
        historical=historical,
        now=now,
    )


def test_query_understanding_and_ambiguous_scope():
    query = understand_query(" Why is Floor 7 warmer with ahu-07 and VAV-7E-14? ")
    assert query.floor == "7"
    assert query.equipment == ("AHU-07", "VAV-7E-14")
    assert not query.question.startswith(" ")
    with pytest.raises(ValueError):
        understand_query("Compare Floor 7 and Floor 8")
    with pytest.raises(ValidationError):
        understand_query("  ")
    with pytest.raises(ValidationError):
        understand_query("x" * 2001)


def test_telemetry_read_is_scoped_and_read_only():
    provider = MockTelemetryProvider((sensor(), sensor(floor="8")))
    assert provider.read(understand_query("Floor 7 AHU-07")) == (sensor(),)
    assert provider.read(understand_query("What is happening?")) == ()
    assert not hasattr(provider, "write")


def test_context_combines_sources_deduplicates_and_retains_constraints():
    manual = chunk_document(
        Document(
            source_id="manual",
            title="Manual",
            text="AHU-07 guide",
            floor="7",
            equipment=("AHU-07",),
        )
    )[0]
    history = chunk_document(
        Document(
            source_id="history",
            title="History",
            text="Trend log",
            kind="history",
            floor="7",
            equipment=("AHU-07",),
        )
    )[0]
    context = build(telemetry=(sensor(), sensor()), retrieved=(manual,), historical=(history,))
    assert len(context.evidence) == 3
    assert context.gaps == ()
    assert "Read-only" in context.constraints[0]


@pytest.mark.parametrize("timestamp", [NOW - timedelta(minutes=16), NOW + timedelta(seconds=1)])
def test_stale_and_future_telemetry_excluded(timestamp):
    context = build(telemetry=(sensor(observed_at=timestamp),))
    assert context.evidence == ()
    assert any("Stale or future" in gap for gap in context.gaps)


def test_budget_scope_collision_and_invalid_channel():
    assert build(telemetry=(sensor(),), builder=ContextBuilder(max_evidence_chars=2)).evidence == ()
    assert build(telemetry=(sensor(floor="8"),)).evidence == ()
    with pytest.raises(ValueError, match="collision"):
        build(telemetry=(sensor(), sensor().model_copy(update={"title": "changed"})))
    with pytest.raises(ValueError, match="Retrieval channel"):
        build(retrieved=(sensor(),))
    with pytest.raises(ValueError, match="timezone-aware"):
        build(now=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError):
        ContextBuilder(max_age_seconds=float("nan"))


def test_evidence_rejects_missing_timestamps_bad_quotes_and_offsets():
    valid = sensor().model_dump()
    for changes in ({"observed_at": None}, {"end_char": 10000}):
        with pytest.raises(ValidationError):
            Evidence.model_validate(valid | changes)
    with pytest.raises(ValidationError):
        Evidence.model_validate(
            valid | {"facts": [Fact(subject="x", metric="y", value="z", quote="invented")]}
        )
    with pytest.raises(ValidationError):
        sensor(observed_at=NOW.replace(tzinfo=None))
