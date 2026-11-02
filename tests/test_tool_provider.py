"""Provider-neutral proposals and optional adapter behavior without real API calls."""

from types import SimpleNamespace

import pytest

from buildingmind.llm.base import ProviderError
from buildingmind.llm.openai_tools import OpenAIToolCallingProvider, strict_schema
from buildingmind.llm.tool_calling_models import ToolSelectionRequest
from examples.tool_calling_floor7 import create_demo


def request_for(o, p):
    from buildingmind.tools.models import ToolArguments

    resource = o.catalog.resolve(ToolArguments(building_id="tower-a", equipment_id="AHU-07"))
    return ToolSelectionRequest(
        question="Floor 7 AHU-07",
        resources=(resource,),
        tools=o.gateway.registry.discover(p),
        round_index=0,
    )


@pytest.mark.parametrize(
    "status,items,expected",
    [
        ("incomplete", [], "incomplete"),
        ("completed", [{"type": "message", "content": [{"type": "refusal"}]}], "refused"),
        ("completed", [], "completed"),
    ],
)
def test_provider_refusal_incomplete_and_no_calls(status, items, expected):
    class Item:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            return self.data

    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                status=status, output=[Item(item) for item in items]
            )
        )
    )
    o, p, _, _ = create_demo()
    try:
        proposal = OpenAIToolCallingProvider(model="offline", client=client).propose(
            request_for(o, p)
        )
        assert proposal.status == expected and not proposal.calls
        for definition in o.gateway.registry.discover(p):
            schema = strict_schema(definition.input_schema)
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
    finally:
        o.gateway.close()


def test_invalid_native_arguments_and_provider_exceptions_are_sanitized():
    class Item:
        def model_dump(self, **kwargs):
            return {
                "type": "function_call",
                "call_id": "call",
                "name": "get_live_telemetry",
                "arguments": "{secret",
            }

    for mode in ("exception", "invalid"):

        def create(mode=mode, **kwargs):
            if mode == "exception":
                raise RuntimeError("SECRET_KEY")
            return SimpleNamespace(status="completed", output=[Item()])

        o, p, _, _ = create_demo()
        try:
            with pytest.raises(ProviderError) as error:
                OpenAIToolCallingProvider(
                    model="offline",
                    client=SimpleNamespace(responses=SimpleNamespace(create=create)),
                ).propose(request_for(o, p))
            assert "SECRET_KEY" not in str(error.value)
        finally:
            o.gateway.close()
