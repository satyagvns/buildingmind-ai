from buildingmind.llm.base import LLMProvider
from buildingmind.llm.mock_provider import MockLLMProvider


def create_llm_provider(name: str = "mock", **options) -> LLMProvider:
    if name == "mock":
        if options:
            raise ValueError("Mock provider takes no options")
        return MockLLMProvider()
    if name == "openai":
        from buildingmind.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(**options)
    raise ValueError(f"Unknown LLM provider: {name}")
