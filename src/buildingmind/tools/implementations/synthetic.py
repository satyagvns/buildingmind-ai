"""Deterministic read-only adapters; authorization occurs before each store access."""

from threading import Event, Lock

from buildingmind.tools.authorization import Principal, authorize
from buildingmind.tools.models import (
    EquipmentData,
    MaintenanceArguments,
    MaintenanceData,
    TelemetryArguments,
    TelemetryData,
    ToolArguments,
    ToolDefinition,
)
from buildingmind.tools.registry import RegisteredTool, ToolRegistry

SCOPES = {
    "get_live_telemetry": "telemetry:read",
    "get_equipment_details": "equipment:read",
    "get_maintenance_history": "maintenance:read",
}


class SyntheticBackend:
    def __init__(
        self,
        catalog,
        *,
        telemetry: tuple[TelemetryData, ...],
        equipment: tuple[EquipmentData, ...],
        maintenance: tuple[MaintenanceData, ...],
    ):
        if any(not data.simulated for data in (*telemetry, *equipment, *maintenance)):
            raise ValueError("Synthetic adapters require explicit simulated data markers")
        self.catalog = catalog
        self._telemetry = {(d.resource.building_id, d.resource.equipment_id): d for d in telemetry}
        self._equipment = {(d.resource.building_id, d.resource.equipment_id): d for d in equipment}
        self._maintenance = {
            (d.resource.building_id, d.resource.equipment_id): d for d in maintenance
        }
        self._accesses: list[tuple[str, str, str]] = []
        self._lock = Lock()

    @property
    def accesses(self):
        with self._lock:
            return tuple(self._accesses)

    def _read(self, name, args, principal, cancel, store):
        resource = self.catalog.resolve(args)
        authorize(principal, resource, (SCOPES[name],))
        if cancel.is_set():
            raise TimeoutError()
        # Access-count evidence is recorded only after data-boundary authorization succeeds.
        with self._lock:
            self._accesses.append((name, resource.building_id, resource.equipment_id))
        return store[(resource.building_id, resource.equipment_id)]

    def telemetry(self, args: TelemetryArguments, principal: Principal, cancel: Event):
        data = self._read("get_live_telemetry", args, principal, cancel, self._telemetry)
        readings = tuple(r for r in data.readings if r.metric in args.metrics)
        return TelemetryData(
            **{
                **data.model_dump(),
                "readings": readings,
                "data_quality": "valid" if readings else "unavailable",
            }
        )

    def equipment(self, args: ToolArguments, principal: Principal, cancel: Event):
        data = self._read("get_equipment_details", args, principal, cancel, self._equipment)
        related = tuple(
            e for e in data.related_equipment if self.catalog.related(principal, data.resource, e)
        )
        return EquipmentData(**{**data.model_dump(), "related_equipment": related})

    def maintenance(self, args: MaintenanceArguments, principal: Principal, cancel: Event):
        data = self._read("get_maintenance_history", args, principal, cancel, self._maintenance)
        return MaintenanceData(
            **{
                **data.model_dump(),
                "events": tuple(
                    sorted(data.events, key=lambda e: (e.event_at, e.work_order_id), reverse=True)
                )[: args.limit],
            }
        )


def synthetic_registry(backend: SyntheticBackend, *, timeout_seconds=1, max_result_bytes=16000):
    registry = ToolRegistry()
    for name, description, args, output, execute in (
        (
            "get_live_telemetry",
            "Read simulated operational measurements; never physical live data.",
            TelemetryArguments,
            TelemetryData,
            backend.telemetry,
        ),
        (
            "get_equipment_details",
            "Read equipment metadata and authorized equipment relationships.",
            ToolArguments,
            EquipmentData,
            backend.equipment,
        ),
        (
            "get_maintenance_history",
            "Read historical maintenance; not proof of current faults.",
            MaintenanceArguments,
            MaintenanceData,
            backend.maintenance,
        ),
    ):
        registry.register(
            RegisteredTool(
                definition=ToolDefinition(
                    name=name,
                    description=description,
                    version="1.0.0",
                    required_scopes=(SCOPES[name],),
                    timeout_seconds=timeout_seconds,
                    max_result_bytes=max_result_bytes,
                    input_schema=args.model_json_schema(),
                    output_schema=output.model_json_schema(),
                ),
                arguments=args,
                output=output,
                execute=execute,
                catalog=backend.catalog,
            )
        )
    return registry
