"""Real optional SDK, in-process HTTP only: schema, native calls and result continuation."""

import json

import pytest

openai = pytest.importorskip("openai", reason="Optional native tool SDK test: install .[openai]")
httpx = pytest.importorskip("httpx")

from buildingmind.llm.openai_tools import OpenAIToolCallingProvider  # noqa: E402
from examples.floor7_hvac import QUESTION  # noqa: E402
from examples.tool_calling_floor7 import create_demo  # noqa: E402


def test_native_tool_roundtrip_uses_registered_schemas_and_forwards_results():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            calls = [
                (
                    "asset",
                    "get_equipment_details",
                    {
                        "building_id": "tower-a",
                        "equipment_id": "AHU-07",
                        "floor": None,
                        "zone": None,
                    },
                )
            ]
        else:
            calls = [
                (
                    "ahu",
                    "get_live_telemetry",
                    {
                        "building_id": "tower-a",
                        "equipment_id": "AHU-07",
                        "floor": 7,
                        "zone": "East",
                        "metrics": ["status", "supply_air_temperature"],
                    },
                ),
                (
                    "vav",
                    "get_live_telemetry",
                    {
                        "building_id": "tower-a",
                        "equipment_id": "VAV-7E-14",
                        "floor": 7,
                        "zone": "East",
                        "metrics": [
                            "temperature",
                            "setpoint",
                            "airflow_deficit",
                            "damper_command",
                            "damper_actual",
                        ],
                    },
                ),
                (
                    "history",
                    "get_maintenance_history",
                    {
                        "building_id": "tower-a",
                        "equipment_id": "VAV-7E-14",
                        "floor": 7,
                        "zone": "East",
                        "limit": 5,
                    },
                ),
            ]
        return httpx.Response(
            200,
            json={
                "id": "response_" + str(len(requests)),
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "offline-model",
                "output": [
                    {
                        "id": "fc_" + id,
                        "type": "function_call",
                        "status": "completed",
                        "call_id": id,
                        "name": name,
                        "arguments": json.dumps(args),
                    }
                    for id, name, args in calls
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = openai.OpenAI(api_key="offline-placeholder", http_client=transport, max_retries=0)
        o, p, _, _ = create_demo(
            selector=OpenAIToolCallingProvider(model="offline-model", client=client)
        )
        try:
            response = o.answer(QUESTION, p)
            assert response.answer.status == "answered" and response.executions == 4
            assert len(response.answer.observations) == 7
        finally:
            o.gateway.close()
    assert len(requests) == 2 and all(body["store"] is False for body in requests)
    for definition in requests[0]["tools"]:
        assert definition["type"] == "function" and definition["strict"] is True
        assert definition["parameters"]["additionalProperties"] is False
    outputs = [item for item in requests[1]["input"] if item.get("type") == "function_call_output"]
    assert len(outputs) == 1 and outputs[0]["call_id"] == "asset"
    data = json.loads(outputs[0]["output"])
    assert data["data"]["related_equipment"] == ["VAV-7E-14"]
    assert data["provenance"]["simulated"] is True
