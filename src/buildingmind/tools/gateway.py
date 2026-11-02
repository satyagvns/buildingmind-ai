"""Central execution boundary with no late-result delivery or queued excess work."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime
from threading import BoundedSemaphore, Event, Lock
from time import monotonic
from uuid import uuid4

from pydantic import ValidationError

from buildingmind.tools.authorization import Principal, authorize
from buildingmind.tools.models import AuditEvent, Provenance, ToolCall, ToolError, ToolResult


class ToolGateway:
    def __init__(
        self,
        registry,
        audit,
        *,
        max_workers=2,
        calls_per_minute=30,
        clock=lambda: datetime.now(UTC),
        timer=monotonic,
    ):
        if not 1 <= max_workers <= 8 or calls_per_minute <= 0:
            raise ValueError("Invalid gateway limits")
        self.registry, self.audit, self.clock, self.timer = registry, audit, clock, timer
        self._capacity = BoundedSemaphore(max_workers)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="buildingmind-tool"
        )
        self._lock = Lock()
        self._rates: dict[str, list[float]] = {}
        self.calls_per_minute = calls_per_minute
        self._closed = False

    def execute(
        self, call: ToolCall, principal: Principal | None, *, request_id: str, request_resource=None
    ) -> ToolResult:
        started = self.timer()
        execution_id = uuid4().hex
        tool = None
        safe_call_id = getattr(call, "call_id", "invalid")
        if not isinstance(safe_call_id, str):
            safe_call_id = "invalid"

        def audit(outcome):
            self.audit.record(
                AuditEvent(
                    request_id=request_id,
                    execution_id=execution_id,
                    principal_id=principal.subject if isinstance(principal, Principal) else None,
                    tool_name=tool.definition.name if tool else "unregistered",
                    outcome=outcome,
                    occurred_at=self.clock(),
                    elapsed_seconds=max(0, self.timer() - started),
                )
            )

        def result(code=None, data=None, provenance=None):
            try:
                audit(code or "success")
            except Exception:
                code, data, provenance = "audit_unavailable", None, None
            return ToolResult(
                call_id=safe_call_id,
                tool_name=tool.definition.name if tool else "unregistered",
                data=data,
                provenance=provenance,
                error=ToolError(code=code, message=f"Tool request {code}.") if code else None,
                elapsed_seconds=max(0, self.timer() - started),
            )

        # Reject malformed model_copy objects too; no raw validation payload is returned.
        try:
            call = ToolCall.model_validate(call.model_dump())
        except Exception:
            return result("invalid_arguments")
        tool = self.registry.resolve(call.name)
        if tool is None:
            return result("unknown_tool")
        try:
            args = tool.arguments.model_validate(call.arguments)
        except (ValidationError, TypeError, ValueError):
            return result("invalid_arguments")
        try:
            resource = tool.catalog.resolve(args)
            authorize(principal, resource, tool.definition.required_scopes)
            if request_resource is not None and (
                resource.building_id,
                resource.floor,
                resource.zone,
            ) != (request_resource.building_id, request_resource.floor, request_resource.zone):
                raise PermissionError("Outside trusted request boundary")
        except Exception:
            return result("unauthorized")
        if self._closed:
            return result("cancelled")
        now = self.timer()
        with self._lock:
            # Remove idle principals as well as expired timestamps; bound active key count.
            self._rates = {
                key: [t for t in stamps if now - t < 60]
                for key, stamps in self._rates.items()
                if stamps and now - stamps[-1] < 60
            }
            if len(self._rates) >= 10000 and principal.subject not in self._rates:
                return result("rate_limited")
            stamps = self._rates.setdefault(principal.subject, [])
            if len(stamps) >= self.calls_per_minute:
                return result("rate_limited")
            stamps.append(now)
        if not self._capacity.acquire(blocking=False):
            return result("busy")
        cancel = Event()
        try:
            # Persist a start event before any operational backend access.
            audit("started")
        except Exception:
            self._capacity.release()
            return result("audit_unavailable")

        def work():
            try:
                if cancel.is_set():
                    raise TimeoutError()
                raw = tool.execute(args, principal, cancel)
                if cancel.is_set():
                    raise TimeoutError()
                # Validate inside the capacity slot; also revalidate already-constructed models.
                value = tool.output.model_validate(
                    raw.model_dump() if hasattr(raw, "model_dump") else raw
                )
                if value.resource != resource:
                    raise PermissionError()
                authorize(principal, value.resource, tool.definition.required_scopes)
                if hasattr(value, "related_equipment") and any(
                    not tool.catalog.related(principal, resource, e)
                    for e in value.related_equipment
                ):
                    raise PermissionError()
                payload = value.model_dump_json()
                if len(payload.encode()) > tool.definition.max_result_bytes:
                    return None, "oversized_result"
                return value, None
            except PermissionError:
                return None, "unauthorized"
            except ValidationError:
                return None, "invalid_result"
            except Exception:
                return None, "backend_failure"

        try:
            future = self._pool.submit(work)
        except Exception:
            self._capacity.release()
            return result("cancelled")
        # Do not free capacity on caller timeout: running backend still owns its slot.
        future.add_done_callback(lambda finished: self._capacity.release())
        try:
            value, error = future.result(timeout=tool.definition.timeout_seconds)
            if self.timer() - started > tool.definition.timeout_seconds:
                cancel.set()
                return result("timeout")
        except TimeoutError:
            cancel.set()
            future.cancel()
            return result("timeout")
        if error:
            return result(error)
        provenance = Provenance(
            execution_id=execution_id,
            tool_name=tool.definition.name,
            tool_version=tool.definition.version,
            source_id=value.source_id,
            simulated=value.simulated,
            obtained_at=self.clock(),
        )
        return result(data=value.model_dump(mode="json"), provenance=provenance)

    def record_rejection(self, call, principal, *, request_id, code):
        tool = self.registry.resolve(call.name)
        name = tool.definition.name if tool else "unregistered"
        try:
            self.audit.record(
                AuditEvent(
                    request_id=request_id,
                    execution_id=uuid4().hex,
                    principal_id=principal.subject,
                    tool_name=name,
                    outcome=code,
                    occurred_at=self.clock(),
                    elapsed_seconds=0,
                )
            )
        except Exception:
            code = "audit_unavailable"
        return ToolResult(
            call_id=call.call_id,
            tool_name=name,
            error=ToolError(code=code, message="Tool rejected by request policy."),
        )

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
