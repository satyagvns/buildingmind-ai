"""Convert authorized schema-validated tool results into existing evidence contracts."""

import hashlib
import json
from datetime import datetime

from buildingmind.context.models import Evidence, Fact
from buildingmind.telemetry.mock_provider import measurement
from buildingmind.tools.authorization import authorize
from buildingmind.tools.models import EquipmentData, MaintenanceData, TelemetryData


class ToolEvidenceMapper:
    def __init__(self, catalog):
        self.catalog = catalog

    def convert(self, results, principal, *, now: datetime):
        telemetry, history, links, gaps = [], [], {}, []
        equipment_results = [
            r for r in results if not r.error and r.tool_name == "get_equipment_details"
        ]
        relationships = {}
        for result in equipment_results:
            data = EquipmentData.model_validate(result.data)
            authorize(principal, data.resource, ("equipment:read",))
            for related in data.related_equipment:
                if not self.catalog.related(principal, data.resource, related):
                    raise PermissionError("Unauthorized relationship")
                relationships.setdefault(related, set()).add(data.resource.equipment_id)
                relationships.setdefault(data.resource.equipment_id, set()).add(related)
        for result in results:
            if result.error:
                gaps.append(
                    f"Tool {result.tool_name}: {result.error.code}; operational data unavailable."
                )
                continue
            if result.provenance is None:
                raise ValueError("Missing tool provenance")
            output = {
                "get_live_telemetry": TelemetryData,
                "get_equipment_details": EquipmentData,
                "get_maintenance_history": MaintenanceData,
            }.get(result.tool_name)
            if output is None:
                raise ValueError("Unknown tool evidence type")
            data = output.model_validate(result.data)
            if (
                data.source_id != result.provenance.source_id
                or data.simulated != result.provenance.simulated
            ):
                raise ValueError("Tool provenance mismatch")
            resource = data.resource
            scope = {
                "get_live_telemetry": "telemetry:read",
                "get_equipment_details": "equipment:read",
                "get_maintenance_history": "maintenance:read",
            }[result.tool_name]
            authorize(principal, resource, (scope,))
            tags = {
                "building": self.catalog.building_name(resource.building_id),
                "floor": str(resource.floor),
                "zone": resource.zone,
                "system": "HVAC",
                "equipment": (resource.equipment_id,),
            }

            def evidence(
                text, *, source_id=data.source_id, facts=(), result=result, data=data, tags=tags
            ):
                key = f"{result.provenance.execution_id}:{source_id}:{text}"
                item = Evidence(
                    id="tool-" + hashlib.sha256(key.encode()).hexdigest()[:24],
                    source_id=source_id,
                    title=f"{result.tool_name} ({'simulated' if data.simulated else 'external'})",
                    kind="history",
                    text=text,
                    end_char=len(text),
                    end_line=text.count("\n") + 1,
                    facts=facts,
                    **tags,
                )
                links[item.id] = result.provenance
                history.append(item)

            if isinstance(data, TelemetryData):
                if data.data_quality != "valid":
                    gaps.append(f"{data.source_id}: operational measurements unavailable.")
                    continue
                if not 0 <= (now - data.observed_at).total_seconds() <= 900:
                    gaps.append(
                        f"{data.source_id}: stale or future operational measurements excluded."
                    )
                    continue
                for reading in data.readings:
                    subject = (
                        f"Floor {resource.floor}"
                        if reading.metric in {"temperature", "setpoint"}
                        else resource.equipment_id
                    )
                    related = tuple(
                        sorted(
                            {resource.equipment_id, *relationships.get(resource.equipment_id, ())}
                        )
                    )
                    item = measurement(
                        subject,
                        reading.metric,
                        reading.value,
                        reading.unit,
                        floor=str(resource.floor),
                        observed_at=data.observed_at,
                        related_equipment=related,
                    )
                    text = (
                        item.text + f"\n{'SIMULATED' if data.simulated else 'External'} telemetry; "
                        f"observed_at={data.observed_at.isoformat()}; source={data.source_id}; "
                        f"tool={result.tool_name}; version={result.provenance.tool_version}."
                    )
                    identity = hashlib.sha256(
                        (result.provenance.execution_id + item.id).encode()
                    ).hexdigest()[:24]
                    item = Evidence.model_validate(
                        {
                            **item.model_dump(),
                            "id": "tool-" + identity,
                            "text": text,
                            "end_char": len(text),
                            "end_line": 2,
                            "source_id": data.source_id,
                            **{k: v for k, v in tags.items() if k != "equipment"},
                        }
                    )
                    links[item.id] = result.provenance
                    telemetry.append(item)
            elif isinstance(data, EquipmentData):
                evidence(
                    "Equipment registry ("
                    + ("synthetic" if data.simulated else "external")
                    + "): "
                    + data.model_dump_json()
                )
            else:
                if not data.events:
                    gaps.append(f"{data.source_id}: no maintenance history available.")
                for event in data.events:
                    text = (
                        "Historical maintenance "
                        f"({'synthetic' if data.simulated else 'external'}): "
                        + json.dumps(event.model_dump(mode="json"), sort_keys=True)
                    )
                    facts = ()
                    if (
                        event.inspection_metric
                        and event.inspection_value
                        and event.inspection_scope
                    ):
                        quote = (
                            f"{resource.equipment_id}: {event.inspection_metric} "
                            f"= {event.inspection_value}"
                        )
                        text += "\n" + quote
                        facts = (
                            Fact(
                                subject=resource.equipment_id,
                                metric=event.inspection_metric,
                                value=event.inspection_value,
                                scope=event.inspection_scope,
                                quote=quote,
                            ),
                        )
                    if event.event_at > now:
                        gaps.append(f"{event.source_id}: future maintenance event excluded.")
                        continue
                    evidence(text, source_id=event.source_id, facts=facts)
        return tuple(telemetry), tuple(history), links, tuple(dict.fromkeys(gaps))
