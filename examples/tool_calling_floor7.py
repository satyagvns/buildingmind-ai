"""Read-only Article 6 demo. All operational data is synthetic and timestamped."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from buildingmind.llm.mock_tools import MockToolCallingProvider
from buildingmind.orchestration.tool_calling import ToolCallingOrchestrator
from buildingmind.tools.audit import MemoryAuditSink
from buildingmind.tools.authorization import Principal, ResourceCatalog, ResourceGrant
from buildingmind.tools.gateway import ToolGateway
from buildingmind.tools.implementations.synthetic import SyntheticBackend, synthetic_registry
from buildingmind.tools.models import EquipmentData, MaintenanceData, Resource, TelemetryData
from examples.advanced_floor7 import create_pipeline as advanced_pipeline
from examples.floor7_hvac import QUESTION

FIXTURE = Path(__file__).parent / "data" / "tool_calling.json"
NOW = datetime(2026, 10, 21, 8, tzinfo=UTC)


def create_demo(*, selector=None, llm=None, max_rounds=2, max_executions=5, timeout_seconds=1):
    fixture = json.loads(FIXTURE.read_text())
    resources = tuple(Resource.model_validate(r) for r in fixture["resources"])
    catalog = ResourceCatalog(resources, {"tower-a": "Tower A", "tower-b": "Tower B"})
    backend = SyntheticBackend(
        catalog,
        telemetry=tuple(TelemetryData.model_validate(d) for d in fixture["telemetry"]),
        equipment=tuple(EquipmentData.model_validate(d) for d in fixture["equipment"]),
        maintenance=tuple(MaintenanceData.model_validate(d) for d in fixture["maintenance"]),
    )
    audit = MemoryAuditSink()
    gateway = ToolGateway(
        synthetic_registry(backend, timeout_seconds=timeout_seconds), audit, clock=lambda: NOW
    )
    principal = Principal(
        subject="synthetic-facility-manager",
        scopes=("telemetry:read", "equipment:read", "maintenance:read"),
        grants=(
            ResourceGrant(
                building_id="tower-a", floor=7, zone="East", equipment_ids=("AHU-07", "VAV-7E-14")
            ),
        ),
    )

    def rag_factory(**kwargs):
        pipeline = advanced_pipeline()
        if llm is not None:
            pipeline.llm = llm
        for name, value in kwargs.items():
            setattr(pipeline, name, value)
        return pipeline

    orchestrator = ToolCallingOrchestrator(
        gateway=gateway,
        selector=selector or MockToolCallingProvider(),
        catalog=catalog,
        rag_factory=rag_factory,
        building_id="tower-a",
        clock=lambda: NOW,
        max_rounds=max_rounds,
        max_executions=max_executions,
    )
    return orchestrator, principal, backend, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "openai"), default="mock")
    parser.add_argument(
        "--model", help="Explicit model for native tools and structured final output"
    )
    args = parser.parse_args()
    if args.provider == "openai" and not args.model:
        parser.error("--model is required for the optional OpenAI provider")
    if args.provider == "openai":
        from buildingmind.llm.openai_provider import OpenAIProvider
        from buildingmind.llm.openai_tools import OpenAIToolCallingProvider

        orchestrator, principal, backend, audit = create_demo(
            selector=OpenAIToolCallingProvider(model=args.model),
            llm=OpenAIProvider(model=args.model),
        )
    else:
        orchestrator, principal, backend, audit = create_demo()
    try:
        result = orchestrator.answer(QUESTION, principal)
        print(
            json.dumps(
                {
                    "result": result.model_dump(mode="json"),
                    "audit": [event.model_dump(mode="json") for event in audit.events],
                },
                indent=2,
            )
        )
    finally:
        orchestrator.gateway.close()
