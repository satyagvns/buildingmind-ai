"""Selection is separate from grounded answer generation and contains no identity input."""

from typing import Literal, Protocol

from pydantic import Field

from buildingmind.models import Model
from buildingmind.tools.models import Resource, ToolCall, ToolDefinition, ToolResult


class ToolProposal(Model):
    status: Literal["completed", "refused", "incomplete"] = "completed"
    calls: tuple[ToolCall, ...] = Field(default=(), max_length=16)
    # Provider-owned replay data is never executed or used as evidence by orchestration.
    replay_items: tuple[dict, ...] = ()


class ToolTurn(Model):
    proposal: ToolProposal
    results: tuple[ToolResult, ...]


class ToolSelectionRequest(Model):
    question: str = Field(min_length=1, max_length=2000)
    resources: tuple[Resource, ...]
    tools: tuple[ToolDefinition, ...]
    history: tuple[ToolTurn, ...] = ()
    round_index: int = Field(ge=0, le=1)


class ToolCallingProvider(Protocol):
    def propose(self, request: ToolSelectionRequest) -> ToolProposal: ...
