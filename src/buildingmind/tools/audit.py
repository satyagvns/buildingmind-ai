"""Injected audit sink: no argument values, source content, or exception text."""

from threading import Lock
from typing import Protocol

from buildingmind.tools.models import AuditEvent


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class MemoryAuditSink:
    def __init__(self):
        self._events: list[AuditEvent] = []
        self._lock = Lock()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)
