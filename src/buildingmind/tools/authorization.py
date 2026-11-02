"""Trusted principal grants are independent of model-selected request filters."""

from pydantic import Field, StrictInt

from buildingmind.models import Model
from buildingmind.tools.models import Resource, ToolArguments


class ResourceGrant(Model):
    building_id: str
    floor: StrictInt = Field(ge=0, le=200)
    zone: str
    equipment_ids: tuple[str, ...] = Field(min_length=1)

    def permits(self, resource: Resource) -> bool:
        return (
            self.building_id == resource.building_id
            and self.floor == resource.floor
            and self.zone == resource.zone
            and resource.equipment_id in self.equipment_ids
        )


class Principal(Model):
    subject: str = Field(min_length=1, max_length=100)
    scopes: tuple[str, ...]
    grants: tuple[ResourceGrant, ...]

    def permits(self, resource: Resource, scopes: tuple[str, ...] = ()) -> bool:
        return set(scopes).issubset(self.scopes) and any(g.permits(resource) for g in self.grants)


def authorize(principal: Principal | None, resource: Resource, scopes: tuple[str, ...]) -> None:
    if not isinstance(principal, Principal):
        raise PermissionError("Missing trusted principal")
    principal = Principal.model_validate(principal.model_dump())
    if not principal.permits(resource, scopes):
        raise PermissionError("Resource or permission denied")


class ResourceCatalog:
    """Application-owned ownership metadata, not an operational backend lookup."""

    def __init__(self, resources: tuple[Resource, ...], building_names: dict[str, str]):
        resources = tuple(Resource.model_validate(r.model_dump()) for r in resources)
        self._resources = {(r.building_id, r.equipment_id): r for r in resources}
        if len(self._resources) != len(resources):
            raise ValueError("Duplicate resource ownership")
        self._names = dict(building_names)
        if any(r.building_id not in self._names for r in resources):
            raise ValueError("Missing explicit building name mapping")

    def resolve(self, args: ToolArguments) -> Resource:
        resource = self._resources.get((args.building_id, args.equipment_id))
        if (
            resource is None
            or (args.floor is not None and args.floor != resource.floor)
            or (args.zone is not None and args.zone != resource.zone)
        ):
            raise PermissionError("Unknown or mismatched resource")
        return resource

    def permitted_resources(self, principal, scopes=()):
        return tuple(r for r in self._resources.values() if principal.permits(r, scopes))

    def building_name(self, building_id: str) -> str:
        return self._names[building_id]

    def related(self, principal: Principal, resource: Resource, equipment_id: str) -> bool:
        related = self._resources.get((resource.building_id, equipment_id))
        return related is not None and principal.permits(related)
