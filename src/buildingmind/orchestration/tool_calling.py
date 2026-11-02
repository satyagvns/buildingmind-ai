"""Controlled proposal rounds; never autonomous planning or direct model execution."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from pydantic import Field

from buildingmind.context.mock_provider import MockHistoricalContextProvider
from buildingmind.llm.base import ProviderError
from buildingmind.llm.tool_calling_models import ToolProposal, ToolSelectionRequest, ToolTurn
from buildingmind.models import Model
from buildingmind.orchestration.tool_context import ToolEvidenceMapper
from buildingmind.query.understanding import BuildingContext
from buildingmind.rag.models import DiagnosticResponse
from buildingmind.retrieval.filters import AccessScope, RetrievalFilter
from buildingmind.telemetry.mock_provider import MockTelemetryProvider
from buildingmind.tools.authorization import Principal, authorize
from buildingmind.tools.models import Provenance, ToolArguments, ToolResult


class ToolCallingResponse(Model):
    answer: DiagnosticResponse
    tool_results: tuple[ToolResult, ...] = ()
    evidence_provenance: dict[str, Provenance] = Field(default_factory=dict)
    rounds: int = 0
    executions: int = 0
    outcome: str = "completed"
    elapsed_seconds: float = Field(default=0, ge=0)


class ToolCallingOrchestrator:
    def __init__(
        self,
        *,
        gateway,
        selector,
        catalog,
        rag_factory: Callable,
        building_id: str,
        max_rounds=2,
        max_executions=5,
        clock=lambda: datetime.now(UTC),
    ):
        if not 1 <= max_rounds <= 2 or not 1 <= max_executions <= 5:
            raise ValueError("Tool limits must be within two rounds/five executions")
        self.gateway, self.selector, self.catalog = gateway, selector, catalog
        self.rag_factory, self.building_id, self.clock = rag_factory, building_id, clock
        self.max_rounds, self.max_executions = max_rounds, max_executions

    def answer(self, question: str, principal: Principal | None):
        started = monotonic()
        request_id = uuid4().hex
        results, turns, seen, call_ids = [], [], set(), set()
        executions, rounds = 0, 0

        def finish(answer, outcome="completed", links=None):
            cited = {c.evidence_id for c in answer.citations}
            return ToolCallingResponse(
                answer=answer,
                tool_results=tuple(results),
                rounds=rounds,
                executions=executions,
                outcome=outcome,
                evidence_provenance={k: v for k, v in (links or {}).items() if k in cited},
                elapsed_seconds=monotonic() - started,
            )

        def unavailable(message, outcome="denied"):
            return finish(
                DiagnosticResponse(
                    question=question, status="insufficient_evidence", evidence_gaps=(message,)
                ),
                outcome,
            )

        try:
            if not isinstance(principal, Principal):
                return unavailable("Trusted authenticated principal is required.")
            principal = Principal.model_validate(principal.model_dump())
            if not principal.grants:
                return unavailable("No authorized building resources.")
            # Validate question using the existing parser, retaining its input/ambiguity limits.
            from buildingmind.rag.query_understanding import understand_query

            basic = understand_query(question)
            # Any operational-control request is denied before selector or backend access.
            if re.search(
                r"\b(set|change|adjust|override|turn|start|stop|open|close|operate|disable|enable)\b",
                question,
                re.I,
            ):
                return unavailable(
                    "Operational control is unavailable; only read-only tools are permitted.",
                    "control_refused",
                )
            # Building mentions are request constraints, never authority. Reject disagreement
            # with the application-selected building before any proposal/backend access.
            mentioned = re.findall(r"\btower[- ]([a-z])\b", question, re.I)
            if any("tower-" + letter.lower() != self.building_id for letter in mentioned):
                return unavailable("Requested building is outside the trusted request scope.")
            resources = []
            for equipment in basic.equipment:
                resource = self.catalog.resolve(
                    ToolArguments(
                        building_id=self.building_id,
                        equipment_id=equipment,
                        floor=int(basic.floor) if basic.floor else None,
                    )
                )
                authorize(principal, resource, ())
                resources.append(resource)
            # Floor-only questions may select explicitly granted assets on that floor.
            if not resources and basic.floor:
                for grant in principal.grants:
                    if grant.building_id == self.building_id and grant.floor == int(basic.floor):
                        for equipment in grant.equipment_ids:
                            resource = self.catalog.resolve(
                                ToolArguments(building_id=self.building_id, equipment_id=equipment)
                            )
                            authorize(principal, resource, ())
                            resources.append(resource)
            if not resources:
                return unavailable("Specify an authorized floor or equipment identifier.")
            # One request operates within one trusted building/floor/zone scope.
            if len({(r.building_id, r.floor, r.zone) for r in resources}) != 1:
                return unavailable("Ask about one authorized building/floor/zone scope.")
            definitions = self.gateway.registry.discover(principal)
            for round_index in range(self.max_rounds):
                request = ToolSelectionRequest(
                    question=question,
                    resources=tuple(resources),
                    tools=definitions,
                    history=tuple(turns),
                    round_index=round_index,
                )
                raw = self.selector.propose(request.model_copy(deep=True))
                proposal = ToolProposal.model_validate(raw.model_dump())
                rounds += 1
                if proposal.status != "completed":
                    return unavailable(
                        "Tool selection was refused or incomplete.", "selection_" + proposal.status
                    )
                if len({c.call_id for c in proposal.calls}) != len(proposal.calls) or any(
                    c.call_id in call_ids for c in proposal.calls
                ):
                    return unavailable(
                        "Repeated tool-call identifiers rejected.", "invalid_proposal"
                    )
                batch = []
                if not proposal.calls:
                    break
                for call in proposal.calls:
                    call_ids.add(call.call_id)
                    registered = self.gateway.registry.resolve(call.name)
                    try:
                        normalized = (
                            registered.arguments.model_validate(call.arguments).model_dump(
                                mode="json"
                            )
                            if registered
                            else call.arguments
                        )
                    except ValueError:
                        normalized = call.arguments
                    key = json.dumps([call.name, normalized], sort_keys=True, separators=(",", ":"))
                    code = None
                    if key in seen:
                        code = "duplicate_call"
                    elif executions >= self.max_executions:
                        code = "call_limit"
                    if code:
                        result = self.gateway.record_rejection(
                            call, principal, request_id=request_id, code=code
                        )
                    else:
                        seen.add(key)
                        executions += 1
                        result = self.gateway.execute(
                            call, principal, request_id=request_id, request_resource=resources[0]
                        )
                    results.append(result)
                    batch.append(result)
                turns.append(ToolTurn(proposal=proposal, results=tuple(batch)))
            telemetry, history, links, gaps = ToolEvidenceMapper(self.catalog).convert(
                results, principal, now=self.clock()
            )
            # Request data cannot broaden retrieval scope. Restrict each candidate using the
            # same exact principal grants, including floor/zone/equipment and source allowlist.
            active = resources[0]
            scope_equipment = tuple(dict.fromkeys(r.equipment_id for r in resources))
            # Only schema-validated registry results can add discovered equipment.
            for result in results:
                if result.error is None and result.tool_name == "get_equipment_details":
                    for equipment in result.data["related_equipment"]:
                        r = self.catalog.resolve(
                            ToolArguments(building_id=self.building_id, equipment_id=equipment)
                        )
                        authorize(principal, r, ())
                        if (r.floor, r.zone) == (active.floor, active.zone):
                            scope_equipment = tuple(dict.fromkeys((*scope_equipment, equipment)))
            context = BuildingContext(
                building=self.catalog.building_name(active.building_id),
                floor=str(active.floor),
                zone=active.zone,
                system="HVAC",
                equipment=scope_equipment,
            )
            pipeline = self.rag_factory(
                telemetry=MockTelemetryProvider(telemetry),
                historical=MockHistoricalContextProvider(history),
                building_context=context,
            )
            # Keep configured v0.5 source/tag restrictions and intersect principal scope.
            if context.building not in pipeline.access.buildings:
                return unavailable("Retrieval building is not authorized.")
            pipeline.access = AccessScope(
                buildings=(context.building,),
                source_ids=pipeline.access.source_ids,
                access_tags=pipeline.access.access_tags,
            )
            filters = RetrievalFilter(
                access=pipeline.access,
                building=context.building,
                floor=context.floor,
                zone=context.zone,
                system=context.system,
                equipment=scope_equipment,
            )
            original = pipeline.retriever
            outer = self

            class GrantedFilter(RetrievalFilter):
                allowed_equipment: tuple[str, ...]

                def matches(self, evidence):
                    return super().matches(evidence) and all(
                        equipment in self.allowed_equipment for equipment in evidence.equipment
                    )

            allowed_equipment = tuple(
                equipment
                for grant in principal.grants
                if (grant.building_id, grant.floor, grant.zone)
                == (active.building_id, active.floor, active.zone)
                for equipment in grant.equipment_ids
            )

            class ScopedRetrieval:
                def retrieve(self, query, requested):
                    bounded = GrantedFilter(
                        **requested.model_dump(), allowed_equipment=allowed_equipment
                    )
                    report = original.retrieve(query, bounded)
                    if any(
                        not filters.matches(c.evidence)
                        or not all(
                            outer.catalog.related(principal, active, equipment)
                            for equipment in c.evidence.equipment
                            if equipment.startswith(("AHU-", "VAV-"))
                        )
                        for c in report.candidates
                    ):
                        raise PermissionError("Document evidence exceeds principal grants")
                    return report

            pipeline.retriever = ScopedRetrieval()
            # Surface tool gaps before optimization, so prompt cost includes those disclosures.
            from buildingmind.retrieval.context_optimizer import ContextOptimizer

            base_optimizer = pipeline.optimizer

            class ToolOptimizer(ContextOptimizer):
                def optimize(self, context):
                    return base_optimizer.optimize(
                        context.model_copy(update={"gaps": (*context.gaps, *gaps)})
                    )

            pipeline.optimizer = ToolOptimizer()
            answer, _ = pipeline.answer(question)
            outcome = "tool_errors" if any(r.error for r in results) else "completed"
            return finish(answer, outcome, links)
        except PermissionError:
            return unavailable("Requested resource is outside the trusted authorization scope.")
        except (ProviderError, ValueError, TypeError, KeyError):
            return unavailable(
                "Tool selection or operational evidence was invalid or unavailable.",
                "processing_error",
            )
        except Exception:
            return unavailable("Tool processing failed safely.", "processing_error")
