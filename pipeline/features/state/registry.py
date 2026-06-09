"""Registry for modular fighter-state feature modules."""

from __future__ import annotations

from pipeline.features.state.contracts import FeatureStateModule
from pipeline.features.state.modules.legacy_v5 import LegacyV5Module


STATE_MODULES: list[FeatureStateModule] = [LegacyV5Module()]


def resolve_state_modules(
    modules: list[FeatureStateModule] | None = None,
) -> list[FeatureStateModule]:
    """Return modules sorted by dependency order."""

    modules = list(STATE_MODULES if modules is None else modules)
    module_by_name: dict[str, FeatureStateModule] = {}

    for module in modules:
        if module.name in module_by_name:
            raise ValueError(f"Duplicate fighter-state module: {module.name}")
        module_by_name[module.name] = module

    resolved: list[FeatureStateModule] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: FeatureStateModule) -> None:
        if module.name in visited:
            return
        if module.name in visiting:
            raise ValueError(f"Circular fighter-state dependency: {module.name}")

        visiting.add(module.name)
        for dependency in module.depends_on:
            if dependency not in module_by_name:
                raise ValueError(f"Missing fighter-state dependency: {dependency}")
            visit(module_by_name[dependency])
        visiting.remove(module.name)
        visited.add(module.name)
        resolved.append(module)

    for module in modules:
        visit(module)

    return resolved
