"""Gateway denial must precede operational backend access."""

from dataclasses import replace
from threading import Event

import pytest
from pydantic import ValidationError

from buildingmind.tools.authorization import Principal, ResourceGrant
from buildingmind.tools.gateway import ToolGateway
from buildingmind.tools.models import ToolCall, ToolDefinition
from buildingmind.tools.registry import ToolRegistry
from examples.tool_calling_floor7 import create_demo


@pytest.fixture
def demo():
    value = create_demo()
    yield value
    value[0].gateway.close()


def call(name="get_live_telemetry", **changes):
    args = {"building_id": "tower-a", "equipment_id": "VAV-7E-14", "metrics": ["airflow_deficit"]}
    args.update(changes)
    return ToolCall(call_id="test", name=name, arguments=args)


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"building_id": "tower-b", "equipment_id": "AHU-07"}, "unauthorized"),
        ({"floor": 3}, "unauthorized"),
        ({"zone": "West"}, "unauthorized"),
        ({"equipment_id": "AHU-03"}, "unauthorized"),
        ({"equipment_id": "AHU-99"}, "unauthorized"),
        ({"equipment_id": None}, "invalid_arguments"),
        ({"equipment_id": "__import__('os').system('id')"}, "invalid_arguments"),
        ({"building_id": "http://169.254.169.254/latest/meta-data"}, "invalid_arguments"),
        ({"url": "http://localhost"}, "invalid_arguments"),
        ({"user_id": "admin"}, "invalid_arguments"),
        ({"roles": ["admin"]}, "invalid_arguments"),
        ({"scopes": ["telemetry:read"]}, "invalid_arguments"),
        ({"metrics": ["set_temperature"]}, "invalid_arguments"),
        ({"floor": "7"}, "invalid_arguments"),
    ],
)
def test_invalid_or_denied_calls_never_access_backend(demo, changes, code):
    orchestrator, principal, backend, audit = demo
    result = orchestrator.gateway.execute(call(**changes), principal, request_id="request")
    assert result.error.code == code and result.data is None
    assert backend.accesses == ()
    assert len(audit.events) == 1 and audit.events[0].outcome == code


@pytest.mark.parametrize(
    "principal",
    [
        None,
        {},
        Principal(
            subject="no-scopes",
            scopes=(),
            grants=(
                ResourceGrant(
                    building_id="tower-a", floor=7, zone="East", equipment_ids=("AHU-07",)
                ),
            ),
        ),
    ],
)
def test_missing_identity_or_permissions_fail_closed(demo, principal):
    o, _, backend, _ = demo
    assert o.gateway.execute(call(), principal, request_id="r").error.code == "unauthorized"
    assert not backend.accesses
    assert o.gateway.registry.discover(principal) == ()


@pytest.mark.parametrize("name", ["set_hvac_setpoint", "os.system", "__import__", "http_get"])
def test_unknown_tool_is_never_executed_or_logged_as_code(demo, name):
    o, p, b, a = demo
    result = o.gateway.execute(call(name=name), p, request_id="r")
    assert result.error.code == "unknown_tool" and not b.accesses
    assert a.events[0].tool_name == "unregistered"


def test_registry_versions_unique_read_only_and_schema_bound(demo):
    o, p, _, _ = demo
    registry = o.gateway.registry
    tool = registry.resolve("get_live_telemetry")
    assert len(registry.discover(p)) == 3
    with pytest.raises(ValueError):
        registry.register(
            replace(tool, definition=tool.definition.model_copy(update={"version": "2.0.0"}))
        )
    with pytest.raises(ValidationError):
        ToolDefinition.model_validate({**tool.definition.model_dump(), "read_only": False})
    with pytest.raises(ValidationError):
        ToolRegistry().register(
            replace(tool, definition=tool.definition.model_copy(update={"read_only": False}))
        )
    with pytest.raises(ValueError):
        ToolRegistry().register(
            replace(tool, definition=tool.definition.model_copy(update={"input_schema": {}}))
        )


def registry_with(demo, execute, **definition):
    o, _, _, _ = demo
    original = o.gateway.registry.resolve("get_live_telemetry")
    registry = ToolRegistry()
    registry.register(
        replace(
            original, execute=execute, definition=original.definition.model_copy(update=definition)
        )
    )
    return registry


def test_all_three_tools_have_valid_provenance_and_no_fabricated_model(demo):
    o, p, b, a = demo
    for name, args in (
        ("get_live_telemetry", {"metrics": ["airflow_deficit"]}),
        ("get_equipment_details", {}),
        ("get_maintenance_history", {"limit": 1}),
    ):
        proposal = ToolCall(
            call_id=name,
            name=name,
            arguments={"building_id": "tower-a", "equipment_id": "VAV-7E-14", **args},
        )
        result = o.gateway.execute(proposal, p, request_id="r")
        assert result.error is None and result.data and result.provenance.simulated
        assert result.data["source_id"] == result.provenance.source_id
        if name == "get_equipment_details":
            assert result.data["manufacturer"] is None and result.data["model"] is None
    assert len(b.accesses) == 3 and len(a.events) == 6


def test_data_boundary_independently_denies_bypass(demo):
    _, p, b, _ = demo
    from buildingmind.tools.models import TelemetryArguments

    denied = TelemetryArguments(building_id="tower-b", equipment_id="AHU-07", metrics=("status",))
    with pytest.raises(PermissionError):
        b.telemetry(denied, p, Event())
    assert not b.accesses


def test_timeout_retains_capacity_and_discards_late_result(demo):
    o, p, b, a = demo
    entered, release, finished = Event(), Event(), Event()
    original = o.gateway.registry.resolve("get_live_telemetry").execute

    def uncooperative(args, principal, cancel):
        entered.set()
        release.wait(3)
        try:
            return original(args, principal, Event())
        finally:
            finished.set()

    gateway = ToolGateway(
        registry_with(demo, uncooperative, timeout_seconds=0.02), a, max_workers=1
    )
    try:
        result = gateway.execute(call(), p, request_id="first")
        assert entered.is_set() and result.error.code == "timeout" and result.data is None
        assert gateway.execute(call(), p, request_id="second").error.code == "busy"
        assert not b.accesses
        release.set()
        assert finished.wait(1)
        assert result.data is None and result.provenance is None
        assert not any(e.outcome == "success" and e.request_id == "first" for e in a.events)
    finally:
        release.set()
        gateway.close()


def test_cooperative_timeout_observes_cancellation(demo):
    o, p, b, a = demo
    cancelled = Event()

    def cooperative(args, principal, cancel):
        cancel.wait(2)
        if cancel.is_set():
            cancelled.set()
        raise TimeoutError()

    gateway = ToolGateway(registry_with(demo, cooperative, timeout_seconds=0.01), a, max_workers=1)
    try:
        assert gateway.execute(call(), p, request_id="r").error.code == "timeout"
        assert cancelled.wait(1) and not b.accesses
    finally:
        gateway.close()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("exception", "backend_failure"),
        ("malformed", "invalid_result"),
        ("oversized", "oversized_result"),
        ("wrong-scope", "unauthorized"),
    ],
)
def test_backend_failure_result_validation_and_sanitized_errors(demo, mode, expected):
    o, p, b, a = demo
    original = o.gateway.registry.resolve("get_live_telemetry").execute

    def backend(args, principal, cancel):
        if mode == "exception":
            raise RuntimeError("SECRET_API_KEY=do-not-log")
        if mode == "malformed":
            return {"secret": "do-not-log"}
        value = original(args, principal, cancel)
        if mode == "wrong-scope":
            return value.model_copy(
                update={"resource": value.resource.model_copy(update={"floor": 3})}
            )
        return value

    gateway = ToolGateway(
        registry_with(demo, backend, max_result_bytes=256 if mode == "oversized" else 16000), a
    )
    try:
        result = gateway.execute(call(), p, request_id="r")
        assert result.error.code == expected
        assert "do-not-log" not in result.model_dump_json() + str(a.events)
    finally:
        gateway.close()


def test_rate_limit_and_audit_failure_prevent_backend_access(demo):
    o, p, b, a = demo
    gateway = ToolGateway(o.gateway.registry, a, calls_per_minute=1)
    try:
        assert gateway.execute(call(), p, request_id="r").error is None
        assert gateway.execute(call(), p, request_id="r").error.code == "rate_limited"
        assert len(b.accesses) == 1
    finally:
        gateway.close()

    class BrokenAudit:
        def record(self, event):
            raise RuntimeError("secret")

    gateway = ToolGateway(o.gateway.registry, BrokenAudit())
    try:
        assert gateway.execute(call(), p, request_id="r").error.code == "audit_unavailable"
        assert len(b.accesses) == 1
    finally:
        gateway.close()


def test_two_running_workers_hold_capacity_until_completion(demo):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    _, p, _, audit = demo
    release, entered = Event(), Event()
    lock = Lock()
    running = 0
    peak = 0
    original = demo[0].gateway.registry.resolve("get_live_telemetry").execute

    def held(args, principal, cancel):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
            if running == 2:
                entered.set()
        try:
            release.wait(2)
            return original(args, principal, cancel)
        finally:
            with lock:
                running -= 1

    gateway = ToolGateway(registry_with(demo, held), audit, max_workers=2)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(gateway.execute, call(), p, request_id="one")
            second = callers.submit(gateway.execute, call(), p, request_id="two")
            assert entered.wait(1)
            assert gateway.execute(call(), p, request_id="third").error.code == "busy"
            release.set()
            assert first.result().error is None and second.result().error is None
            assert peak == 2
    finally:
        release.set()
        gateway.close()


def test_discovery_requires_a_real_granted_resource_and_closed_gateway_does_not_read(demo):
    o, p, b, _ = demo
    unknown = Principal(
        subject="unknown-resource",
        scopes=p.scopes,
        grants=(
            ResourceGrant(building_id="tower-a", floor=7, zone="East", equipment_ids=("AHU-99",)),
        ),
    )
    assert not o.gateway.registry.discover(unknown)
    o.gateway.close()
    assert o.gateway.execute(call(), p, request_id="closed").error.code == "cancelled"
    assert not b.accesses


def test_malformed_model_copy_and_oversized_arguments_fail_before_backend(demo):
    o, p, b, _ = demo
    bad = call().model_copy(update={"arguments": {"building_id": "x" * 9000}, "call_id": None})
    result = o.gateway.execute(bad, p, request_id="r")
    assert result.error.code == "invalid_arguments" and not b.accesses


def test_public_metric_aliases_normalize_to_existing_evidence_names(demo):
    o, p, _, _ = demo
    result = o.gateway.execute(call(metrics=["airflow", "damper_position"]), p, request_id="r")
    assert result.error is None
    assert {r["metric"]: r["value"] for r in result.data["readings"]} == {
        "airflow_deficit": "62",
        "damper_actual": "31",
    }
