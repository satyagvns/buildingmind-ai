import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from buildingmind.llm.base import ProviderError
from buildingmind.llm.factory import create_llm_provider
from buildingmind.llm.openai_provider import OpenAIProvider
from buildingmind.rag.models import AnswerDraft, Context
from buildingmind.rag.prompt_builder import PromptBuilder
from buildingmind.rag.query_understanding import understand_query


def prompt():
    return PromptBuilder().build(Context(query=understand_query("Floor 7 AHU-07")))


def test_mock_is_default_deterministic_and_optional_sdk_is_not_imported():
    provider = create_llm_provider()
    assert provider.generate(prompt()) == provider.generate(prompt())
    command = (
        "import sys; from buildingmind.llm.factory import create_llm_provider; "
        'create_llm_provider(); assert "openai" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_factory_requires_explicit_provider_and_model():
    with pytest.raises(ValueError):
        create_llm_provider("unknown")
    with pytest.raises(ValueError):
        create_llm_provider("mock", model="unexpected")
    with pytest.raises(TypeError):
        create_llm_provider("openai")
    with pytest.raises(ValueError):
        OpenAIProvider(model=" ", timeout=1)
    with pytest.raises(ValueError):
        OpenAIProvider(model="configured", timeout=float("inf"))


def test_openai_adapter_structured_output_with_injected_client():
    draft = AnswerDraft(observations=(), historical_evidence=(), hypotheses=())
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=draft)
    provider = OpenAIProvider(model="test-model", client=client)
    assert provider.generate(prompt()) == draft
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["text_format"] is AnswerDraft
    assert kwargs["store"] is False
    assert kwargs["timeout"] == 30
    assert kwargs["model"] == "test-model"
    assert [item["role"] for item in kwargs["input"]] == ["system", "user"]


@pytest.mark.parametrize("status,parsed", [("completed", None), ("incomplete", None)])
def test_refusal_and_incomplete_output_are_provider_errors(status, parsed):
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status=status, output_parsed=parsed)
    with pytest.raises(ProviderError):
        OpenAIProvider(model="test", client=client).generate(prompt())


def test_provider_exceptions_do_not_expose_payloads():
    client = Mock()
    client.responses.parse.side_effect = TimeoutError("secret-key-and-source-data")
    with pytest.raises(ProviderError) as error:
        OpenAIProvider(model="test", client=client).generate(prompt())
    assert "secret" not in str(error.value)


def test_prompt_has_separate_instructions_json_context_and_allowlisted_hypotheses():
    built = prompt()
    payload = json.loads(built.user)
    assert payload["context"]["query"]["floor"] == "7"
    assert payload["supported_hypotheses"] == []
    assert "untrusted data" in built.system
    assert "Read-only" in payload["context"]["constraints"][0]
