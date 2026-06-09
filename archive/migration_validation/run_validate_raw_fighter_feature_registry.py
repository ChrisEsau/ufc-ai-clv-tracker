"""Validate raw fighter feature registry and loader contracts.

This script does not rebuild fighter_state_history. It verifies that the raw
fighter feature registry parses, summarizes master-sourced vs calculated
feature groups, and validates any active plugin entries.

Run from repo root:

    python -m archive.migration_validation.run_validate_raw_fighter_feature_registry
"""

from __future__ import annotations

from pipeline.features.raw_fighter_feature_loader import (
    DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH,
    list_registered_feature_groups,
    list_registered_outputs,
    load_active_raw_fighter_feature_plugins,
    load_raw_fighter_feature_registry,
)


def main() -> None:
    """Run raw fighter feature registry validation."""

    print("=" * 80)
    print("RAW FIGHTER FEATURE REGISTRY VALIDATION")
    print("=" * 80)
    print(f"Registry path: {DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH}")

    registry = load_raw_fighter_feature_registry()
    groups = registry.get("feature_groups", {})
    outputs_by_group = list_registered_outputs()
    active_plugins = load_active_raw_fighter_feature_plugins()

    print(f"Registry name : {registry.get('registry_name')}")
    print(f"Version       : {registry.get('version')}")
    print(f"Status        : {registry.get('status')}")
    print(f"Groups        : {len(list_registered_feature_groups())}")
    print(f"Active plugins: {len(active_plugins)}")

    source_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    total_outputs = 0

    for group_name, group in groups.items():
        source_layer = str(group.get("source_layer", "unknown"))
        status = str(group.get("status", "unknown"))
        output_count = len(outputs_by_group.get(group_name, []))
        total_outputs += output_count
        source_counts[source_layer] = source_counts.get(source_layer, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Registered outputs: {total_outputs}")

    print("\nGroups by source layer:")
    for source_layer, count in sorted(source_counts.items()):
        print(f"  - {source_layer}: {count}")

    print("\nGroups by status:")
    for status, count in sorted(status_counts.items()):
        print(f"  - {status}: {count}")

    print("\nFeature groups:")
    for group_name, output_columns in outputs_by_group.items():
        group = groups[group_name]
        print(
            f"  - {group_name}: "
            f"source={group.get('source_layer')} "
            f"status={group.get('status')} "
            f"outputs={len(output_columns)}"
        )

    if active_plugins:
        print("\nLoaded active plugins:")
        for plugin in active_plugins:
            print(f"  - {plugin.feature_group}: outputs={len(plugin.output_columns)}")
    else:
        print("\nLoaded active plugins: none expected yet")

    print("DONE")


if __name__ == "__main__":
    main()
