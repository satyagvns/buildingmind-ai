"""Provider-neutral contracts. Identity never appears in tool arguments."""

from typing import Literal

from pydantic import AwareDatetime, Field, StrictInt, field_validator, model_validator

from buildingmind.models import Model


class Resource(Model):
    building_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)
    equipment_id: str = Field(pattern=r"^(?:AHU-\d+|VAV-\d+[A-Z]?-\d+)$", max_length=80)
    floor: StrictInt = Field(ge=0, le=200)
    zone: str = Field(min_length=1, max_length=80)


class ToolArguments(Model):
    building_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)
    equipment_id: str = Field(pattern=r"^(?:AHU-\d+|VAV-\d+[A-Z]?-\d+)$", max_length=80)
    floor: StrictInt | None = Field(default=None, ge=0, le=200)
    zone: str | None = Field(default=None, min_length=1, max_length=80)


Metric = Literal[
    "temperature",
    "setpoint",
    "status",
    "supply_air_temperature",
    "airflow_deficit",
    "damper_command",
    "damper_actual",
]


class TelemetryArguments(ToolArguments):
    metrics: tuple[Metric | Literal["airflow", "damper_position"], ...] = Field(
        min_length=1, max_length=9
    )

    @field_validator("metrics")
    @classmethod
    def normalize_metrics(cls, values):
        aliases = {"airflow": "airflow_deficit", "damper_position": "damper_actual"}
        return tuple(dict.fromkeys(aliases.get(value, value) for value in values))


class MaintenanceArguments(ToolArguments):
    limit: StrictInt = Field(default=5, ge=1, le=20)


class Reading(Model):
    metric: Metric
    value: str = Field(min_length=1, max_length=80)
    unit: str = Field(max_length=20)


class ToolData(Model):
    resource: Resource
    source_id: str = Field(min_length=1, max_length=200)
    simulated: bool


class TelemetryData(ToolData):
    observed_at: AwareDatetime
    data_quality: Literal["valid", "unavailable"]
    readings: tuple[Reading, ...] = Field(max_length=7)


class EquipmentData(ToolData):
    equipment_type: Literal["AHU", "VAV"]
    manufacturer: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=100)
    specifications: tuple[str, ...] = Field(default=(), max_length=20)
    related_equipment: tuple[str, ...] = Field(default=(), max_length=20)


class MaintenanceEvent(Model):
    work_order_id: str = Field(min_length=1, max_length=100)
    event_at: AwareDatetime
    fault: str = Field(min_length=1, max_length=2000)
    action: str = Field(min_length=1, max_length=2000)
    resolution: Literal["resolved", "open", "unknown"]
    source_id: str = Field(min_length=1, max_length=200)
    # Optional structured inspection comparison, not an annotation of current telemetry.
    inspection_metric: str | None = Field(default=None, max_length=100)
    inspection_value: str | None = Field(default=None, max_length=100)
    inspection_scope: str | None = Field(default=None, max_length=100)


class MaintenanceData(ToolData):
    events: tuple[MaintenanceEvent, ...] = Field(max_length=20)


class ToolDefinition(Model):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str = Field(min_length=1, max_length=1000)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    required_scopes: tuple[str, ...] = Field(min_length=1)
    read_only: Literal[True] = True
    timeout_seconds: float = Field(default=1, gt=0, le=30)
    max_result_bytes: int = Field(default=16000, ge=256, le=1000000)
    input_schema: dict
    output_schema: dict


class ToolCall(Model):
    call_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    name: str = Field(min_length=1, max_length=100)
    arguments: dict

    @model_validator(mode="after")
    def bound_arguments(self):
        import json

        if len(json.dumps(self.arguments, allow_nan=False).encode()) > 8192:
            raise ValueError("Tool arguments exceed size limit")
        return self


ErrorCode = Literal[
    "unknown_tool",
    "invalid_arguments",
    "unauthorized",
    "timeout",
    "busy",
    "rate_limited",
    "backend_failure",
    "invalid_result",
    "oversized_result",
    "call_limit",
    "duplicate_call",
    "cancelled",
    "audit_unavailable",
]


class ToolError(Model):
    code: ErrorCode
    message: str


class Provenance(Model):
    execution_id: str
    tool_name: str
    tool_version: str
    source_id: str
    simulated: bool
    obtained_at: AwareDatetime


class ToolResult(Model):
    call_id: str
    tool_name: str
    data: dict | None = None
    error: ToolError | None = None
    provenance: Provenance | None = None
    elapsed_seconds: float = Field(default=0, ge=0)


class AuditEvent(Model):
    request_id: str
    execution_id: str
    principal_id: str | None
    tool_name: str  # registered names only; arbitrary model text is never logged
    outcome: str
    occurred_at: AwareDatetime
    elapsed_seconds: float = Field(ge=0)
