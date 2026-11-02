"""Bounded deterministic tool selection; no generated prose becomes source evidence."""

import re

from buildingmind.llm.tool_calling_models import ToolProposal
from buildingmind.tools.models import ToolCall


class MockToolCallingProvider:
    def propose(self, request):
        allowed = {t.name for t in request.tools}
        q = request.question.lower()
        if not request.resources or not re.search(
            r"warm|cool|diagnos|telemetry|airflow|equipment|maintenance", q
        ):
            return ToolProposal()
        if request.round_index == 0:
            asset = request.resources[0]
            name = (
                "get_live_telemetry"
                if "telemetry" in q
                else "get_maintenance_history"
                if "maintenance history" in q
                else "get_equipment_details"
            )
            if name not in allowed:
                return ToolProposal()
            args = {"building_id": asset.building_id, "equipment_id": asset.equipment_id}
            if name == "get_live_telemetry":
                args["metrics"] = (
                    ["status", "supply_air_temperature"]
                    if asset.equipment_id.startswith("AHU-")
                    else [
                        "temperature",
                        "setpoint",
                        "airflow_deficit",
                        "damper_command",
                        "damper_actual",
                    ]
                )
            return ToolProposal(calls=(ToolCall(call_id="round0", name=name, arguments=args),))
        if any(word in q for word in ("telemetry", "maintenance history", "equipment details")):
            return ToolProposal()
        equipment_results = [
            r
            for turn in request.history
            for r in turn.results
            if not r.error and r.tool_name == "get_equipment_details"
        ]
        if not equipment_results:
            return ToolProposal()
        data = equipment_results[0].data
        assets = [data["resource"]["equipment_id"], *data["related_equipment"]]
        building = data["resource"]["building_id"]
        calls = []
        for index, asset in enumerate(assets[:2]):
            if "get_live_telemetry" in allowed:
                metrics = (
                    ["status", "supply_air_temperature"]
                    if asset.startswith("AHU-")
                    else [
                        "temperature",
                        "setpoint",
                        "airflow_deficit",
                        "damper_command",
                        "damper_actual",
                    ]
                )
                calls.append(
                    ToolCall(
                        call_id=f"telemetry{index}",
                        name="get_live_telemetry",
                        arguments={
                            "building_id": building,
                            "equipment_id": asset,
                            "metrics": metrics,
                        },
                    )
                )
        vav = next((asset for asset in assets if asset.startswith("VAV-")), None)
        if vav and "get_maintenance_history" in allowed:
            calls.append(
                ToolCall(
                    call_id="maintenance",
                    name="get_maintenance_history",
                    arguments={"building_id": building, "equipment_id": vav, "limit": 5},
                )
            )
        return ToolProposal(calls=tuple(calls))
