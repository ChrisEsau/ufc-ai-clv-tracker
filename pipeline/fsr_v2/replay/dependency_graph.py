"""Dependency ordering and invalidation checks."""

from pipeline.fsr_v2.traits.registry import GROUPS, TraitGroup


def order_groups(groups: list[TraitGroup]) -> list[TraitGroup]:
    requested = {group.name: group for group in groups}
    ordered: list[TraitGroup] = []
    visiting: set[str] = set()

    def visit(group: TraitGroup) -> None:
        if group.name in {item.name for item in ordered}:
            return
        if group.name in visiting:
            raise ValueError(f"Cyclic FSR V2 dependency at {group.name}")
        visiting.add(group.name)
        for dependency in group.dependencies:
            if dependency in requested:
                visit(requested[dependency])
        visiting.remove(group.name)
        ordered.append(group)

    for item in groups:
        visit(item)
    return ordered


def dependency_versions(group: TraitGroup, fingerprints: dict[str, str]) -> dict[str, str]:
    return {name: fingerprints.get(name, "missing") for name in group.dependencies}
