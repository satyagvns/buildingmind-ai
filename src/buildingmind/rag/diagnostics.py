"""Small deterministic HVAC plausibility rules; thresholds are demonstration defaults."""

import math
from collections import defaultdict

from buildingmind.context.models import Evidence, Fact
from buildingmind.rag.models import Context, HypothesisDraft

FactPair = tuple[Evidence, Fact]


def _number(pair: FactPair | None) -> float | None:
    if pair is None:
        return None
    try:
        value = float(pair[1].value)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def _snapshot_hypotheses(evidence: list[Evidence]) -> list[HypothesisDraft]:
    facts = {(f.subject, f.metric, f.unit): (e, f) for e in evidence for f in e.facts}
    candidates = []
    for (subject, metric, unit), flow in facts.items():
        if not subject.startswith("VAV-") or metric != "airflow_deficit" or unit != "%":
            continue
        deficit = _number(flow)
        if deficit is None or not 30 <= deficit <= 100:
            continue
        for ahu in flow[0].equipment:
            status = facts.get((ahu, "status", ""))
            supply = facts.get((ahu, "supply_air_temperature", ""))
            if (
                not ahu.startswith("AHU-")
                or status is None
                or supply is None
                or status[1].value.lower() != "running"
                or supply[1].value.lower() != "normal"
            ):
                continue
            for (_, key, u), temperature in facts.items():
                if key != "temperature" or u != "°C" or ahu not in temperature[0].equipment:
                    continue
                setpoint = facts.get((temperature[1].subject, "setpoint", "°C"))
                t, s = _number(temperature), _number(setpoint)
                if t is None or s is None or t <= s:
                    continue
                ids = tuple(
                    dict.fromkeys(
                        item[0].id for item in (temperature, setpoint, status, supply, flow)
                    )
                )
                candidates.append(HypothesisDraft(code="airflow_restriction", evidence_ids=ids))
                command = facts.get((subject, "damper_command", "%"))
                actual = facts.get((subject, "damper_actual", "%"))
                c, a = _number(command), _number(actual)
                if c is not None and a is not None and 0 <= a < c <= 100 and c - a >= 20:
                    candidates.append(
                        HypothesisDraft(
                            code="damper_tracking_fault",
                            evidence_ids=tuple(dict.fromkeys((*ids, command[0].id, actual[0].id))),
                        )
                    )
    return candidates


def diagnostic_candidates(context: Context) -> tuple[HypothesisDraft, ...]:
    snapshots = defaultdict(list)
    for evidence in context.evidence:
        if evidence.kind == "telemetry":
            snapshots[(evidence.floor, evidence.observed_at)].append(evidence)
    return tuple(
        dict.fromkeys(
            candidate
            for snapshot in snapshots.values()
            for candidate in _snapshot_hypotheses(snapshot)
        )
    )
