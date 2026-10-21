"""Optional SDK contract test using an in-process HTTP transport, never a real API."""

import json

import pytest

openai = pytest.importorskip("openai", reason="Optional SDK contract test: install .[openai]")
httpx = pytest.importorskip("httpx")

from buildingmind.llm.openai_provider import OpenAIProvider  # noqa: E402
from buildingmind.rag.models import AnswerDraft, Context  # noqa: E402
from buildingmind.rag.prompt_builder import PromptBuilder  # noqa: E402
from buildingmind.rag.query_understanding import understand_query  # noqa: E402


def test_real_sdk_serializes_schema_and_parses_response_without_network():
    draft = AnswerDraft(observations=(), historical_evidence=(), hypotheses=())
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_offline_contract",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "offline-contract-model",
                "output": [
                    {
                        "id": "msg_offline",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": draft.model_dump_json(),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = openai.OpenAI(
            api_key="offline-placeholder", http_client=http_client, max_retries=0
        )
        provider = OpenAIProvider(model="offline-contract-model", client=client)
        prompt = PromptBuilder().build(Context(query=understand_query("Floor 7 AHU-07")))
        assert provider.generate(prompt) == draft
    assert len(requests) == 1
    format_spec = requests[0]["text"]["format"]
    assert format_spec["type"] == "json_schema"
    assert format_spec["strict"] is True
    schema = format_spec["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"observations", "historical_evidence", "hypotheses"}
    assert requests[0]["store"] is False
