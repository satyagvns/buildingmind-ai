"""Whole-evidence prompt budgeting, including provenance and instructions."""

from typing import Protocol

from buildingmind.rag.grounding import detect_conflicts
from buildingmind.rag.models import Context
from buildingmind.rag.prompt_builder import PromptBuilder


class TokenCounter(Protocol):
    exact: bool

    def count(self, text: str) -> int: ...


class UTF8BudgetCounter:
    # Conservative accounting proxy, not model-specific tokenization.
    exact = False

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


class EncodingTokenCounter:
    """Use a caller-supplied local tokenizer encoding; never download tokenizer data here.

    Counts serialized text tokens; transport message framing remains a separate reserve.
    """

    exact = True

    def __init__(self, encoding):
        self.encoding = encoding

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))


class ContextOptimizer:
    def __init__(
        self,
        *,
        token_budget: int = 32000,
        counter: TokenCounter | None = None,
        reserved_tokens: int = 256,
    ):
        if token_budget <= 0 or reserved_tokens < 0:
            raise ValueError("Token budget must be positive")
        self.token_budget = token_budget
        self.reserved_tokens = reserved_tokens
        self.counter = counter or UTF8BudgetCounter()
        self.builder = PromptBuilder()

    def cost(self, context):
        prompt = self.builder.build(context)
        return (
            self.reserved_tokens
            + self.counter.count(prompt.system)
            + self.counter.count(prompt.user)
        )

    def optimize(self, context: Context) -> Context:
        gaps = list(context.gaps)
        if not self.counter.exact:
            gaps.append(
                "Token accounting uses an approximate UTF-8-byte budget, not exact model tokens."
            )
        # Pin complete conflict groups and telemetry; partial conflict selection is unsafe.
        conflicts = detect_conflicts(context)
        pinned = {c.evidence_id for conflict in conflicts for c in conflict.citations}
        pinned.update(e.id for e in context.evidence if e.kind == "telemetry")
        ranked = sorted(
            enumerate(context.evidence),
            key=lambda item: (
                item[1].id not in pinned,
                item[1].approval != "approved",
                -item[1].authority,
                item[0],
            ),
        )
        selected = [e for _, e in ranked if e.id in pinned]
        rest = [e for _, e in ranked if e.id not in pinned]
        omitted = False

        def assembled():
            extra = ("Evidence omitted to fit the prompt budget.",) if omitted else ()
            return context.model_copy(
                update={"evidence": tuple(selected), "gaps": tuple(dict.fromkeys((*gaps, *extra)))}
            )

        if self.cost(assembled()) > self.token_budget:
            raise ValueError("Critical telemetry/conflicting evidence exceeds prompt budget")
        for evidence in rest:
            # Only remove exact contained spans within the same source revision metadata.
            if any(
                e.source_id == evidence.source_id
                and e.version == evidence.version
                and e.approval == evidence.approval
                and e.authority == evidence.authority
                and e.start_char <= evidence.start_char
                and e.end_char >= evidence.end_char
                and e.text[evidence.start_char - e.start_char : evidence.end_char - e.start_char]
                == evidence.text
                and set(evidence.facts).issubset(e.facts)
                for e in selected
            ):
                continue
            selected.append(evidence)
            if self.cost(assembled()) > self.token_budget:
                selected.pop()
                omitted = True
        # Adding the omission message itself can consume budget.
        while self.cost(assembled()) > self.token_budget:
            removable = next((e for e in reversed(selected) if e.id not in pinned), None)
            if removable is None:
                raise ValueError("Critical context and omission notice exceed prompt budget")
            selected.remove(removable)
        return assembled()
