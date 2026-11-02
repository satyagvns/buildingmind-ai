"""Only application code registers implementations; model strings cannot load code."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from buildingmind.tools.authorization import Principal, ResourceCatalog
from buildingmind.tools.models import ToolArguments, ToolData, ToolDefinition


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    arguments: type[ToolArguments]
    output: type[ToolData]
    execute: Callable[[ToolArguments, Principal, Event], ToolData]
    catalog: ResourceCatalog


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        definition = ToolDefinition.model_validate(tool.definition.model_dump())
        if definition.name in self._tools:
            raise ValueError("Tool name already registered; versions cannot silently replace it")
        if (
            definition.input_schema != tool.arguments.model_json_schema()
            or definition.output_schema != tool.output.model_json_schema()
        ):
            raise ValueError("Tool schema mismatch")
        self._tools[definition.name] = tool

    def resolve(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def discover(self, principal: Principal | None) -> tuple[ToolDefinition, ...]:
        if not isinstance(principal, Principal):
            return ()
        principal = Principal.model_validate(principal.model_dump())
        return tuple(
            t.definition.model_copy(deep=True)
            for _, t in sorted(self._tools.items())
            if t.catalog.permitted_resources(principal, t.definition.required_scopes)
        )
