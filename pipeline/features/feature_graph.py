"""Feature graph planner for UFC feature-view experiments.

The feature graph is a read-only planning layer. It resolves bundle and
transform registry selections into raw feature requirements and generated
feature names. It does not build feature values yet.

Run from repo root:

    python -m pipeline.features.feature_graph \
        --feature-view-config configs/feature_views/moneyline_base.yaml

Optional JSON export:

    python -m pipeline.features.feature_graph \
        --feature-view-config configs/feature_views/moneyline_base.yaml \
        --output data/features/feature_graph_moneyline_base.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_FEATURE_VIEW_CONFIG = "configs/feature_views/moneyline_base.yaml"
DEFAULT_BUNDLE_REGISTRY = "configs/features/feature_bundles.yaml"
DEFAULT_TRANSFORM_REGISTRY = "configs/features/transform_registry.yaml"


@dataclass(frozen=True)
class FeatureGraphPlan:
    """Resolved feature graph plan."""

    view_id: str
    view_family: str
    selected_bundles: list[str]
    selected_transforms: list[str]
    raw_feature_columns: list[str]
    generated_feature_columns: list[str]
    passthrough_feature_columns: list[str]


TRANSFORM_SUFFIXES = {
    "red_minus_blue": "diff",
    "blue_minus_red": "reverse_diff",
    "ratio": "ratio",
    "absolute_gap": "abs_gap",
    "interaction": "interaction",
    "percentile": "percentile",
    "zscore": "zscore",
    "style_matchup": "style_matchup",
    "market_gap": "market_gap",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Resolve a UFC feature graph plan.")
    parser.add_argument(
        "--feature-view-config",
        default=DEFAULT_FEATURE_VIEW_CONFIG,
        help="Path to feature-view YAML config.",
    )
    parser.add_argument(
        "--bundle-registry",
        default=DEFAULT_BUNDLE_REGISTRY,
        help="Path to feature bundle registry YAML.",
    )
    parser.add_argument(
        "--transform-registry",
        default=DEFAULT_TRANSFORM_REGISTRY,
        help="Path to transform registry YAML.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for the resolved graph plan.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must deserialize into a dictionary: {path}")

    return payload


def resolve_feature_graph_plan(
    feature_view_config_path: str | Path = DEFAULT_FEATURE_VIEW_CONFIG,
    bundle_registry_path: str | Path = DEFAULT_BUNDLE_REGISTRY,
    transform_registry_path: str | Path = DEFAULT_TRANSFORM_REGISTRY,
) -> FeatureGraphPlan:
    """Resolve a feature-view config into a feature graph plan.

    The current feature-view config is still adapter-based, so this resolver
    supports both future explicit ``bundles``/``transforms`` sections and the
    current ``include`` section used by ``moneyline_base.yaml``.
    """

    feature_view_config = load_yaml(feature_view_config_path)
    bundle_registry = load_yaml(bundle_registry_path)
    transform_registry = load_yaml(transform_registry_path)

    view_id = str(feature_view_config.get("view_id", ""))
    view_family = str(feature_view_config.get("view_family", ""))
    if not view_id or not view_family:
        raise ValueError("Feature-view config must define view_id and view_family")

    bundles = resolve_selected_bundles(feature_view_config, bundle_registry)
    transforms = resolve_selected_transforms(feature_view_config, bundle_registry, transform_registry, bundles)

    raw_columns: list[str] = []
    generated_columns: list[str] = []
    passthrough_columns: list[str] = []

    for bundle_id in bundles:
        bundle = bundle_registry["bundles"][bundle_id]
        candidate_columns = [str(column) for column in bundle.get("candidate_columns", [])]
        source_layer = str(bundle.get("source_layer", ""))
        source_prefix = str(bundle.get("source_prefix", ""))

        if source_layer == "engineered":
            passthrough_columns.extend(candidate_columns)
            continue

        for base_column in candidate_columns:
            canonical_column = f"{source_prefix}{base_column}" if source_prefix else base_column
            raw_columns.extend(resolve_raw_columns_for_source(source_layer, canonical_column))

            for transform_id in transforms:
                if transform_applies_to_bundle(transform_id, bundle):
                    generated_columns.append(build_generated_feature_name(canonical_column, transform_id))

    return FeatureGraphPlan(
        view_id=view_id,
        view_family=view_family,
        selected_bundles=dedupe_preserve_order(bundles),
        selected_transforms=dedupe_preserve_order(transforms),
        raw_feature_columns=dedupe_preserve_order(raw_columns),
        generated_feature_columns=dedupe_preserve_order(generated_columns),
        passthrough_feature_columns=dedupe_preserve_order(passthrough_columns),
    )


def resolve_selected_bundles(
    feature_view_config: dict[str, Any],
    bundle_registry: dict[str, Any],
) -> list[str]:
    """Resolve selected bundles from a feature-view config."""

    configured_bundles = feature_view_config.get("bundles")
    if isinstance(configured_bundles, list) and configured_bundles:
        bundle_ids = [str(bundle) for bundle in configured_bundles]
    else:
        bundle_ids = infer_bundles_from_legacy_include(feature_view_config)

    available = set(bundle_registry.get("bundles", {}).keys())
    missing = [bundle_id for bundle_id in bundle_ids if bundle_id not in available]
    if missing:
        raise ValueError(f"Feature-view config references unknown bundles: {missing}")

    return bundle_ids


def infer_bundles_from_legacy_include(feature_view_config: dict[str, Any]) -> list[str]:
    """Infer bundles from the current adapter-style feature-view config."""

    include = feature_view_config.get("include", {})
    bundles: list[str] = []

    if include.get("prefight_state", False):
        bundles.extend(["core_state", "striking", "grappling", "finish_profile", "recent_form"])
    if include.get("ewm_state", False):
        bundles.append("ewm_state")
    engineered = include.get("engineered_features", {})
    if isinstance(engineered, dict) and engineered.get("enabled", False):
        bundles.append("engineered_matchup")

    return bundles


def resolve_selected_transforms(
    feature_view_config: dict[str, Any],
    bundle_registry: dict[str, Any],
    transform_registry: dict[str, Any],
    bundles: list[str],
) -> list[str]:
    """Resolve selected transforms from config or bundle recommendations."""

    configured_transforms = feature_view_config.get("transforms")
    if isinstance(configured_transforms, list) and configured_transforms:
        transform_ids = [str(transform) for transform in configured_transforms]
    else:
        transform_ids = []
        for bundle_id in bundles:
            bundle = bundle_registry["bundles"][bundle_id]
            transform_ids.extend(str(item) for item in bundle.get("recommended_transforms", []))

    available = set(transform_registry.get("transforms", {}).keys())
    missing = [transform_id for transform_id in transform_ids if transform_id not in available]
    if missing:
        raise ValueError(f"Feature-view config references unknown transforms: {missing}")

    return dedupe_preserve_order(transform_ids)


def resolve_raw_columns_for_source(source_layer: str, column: str) -> list[str]:
    """Return raw columns needed for a source-layer feature."""

    if source_layer in {"fighter_state", "prepared_fights", "archetype"}:
        return [f"r_{column}", f"b_{column}"]
    if source_layer == "market":
        return [column]
    return [column]


def transform_applies_to_bundle(transform_id: str, bundle: dict[str, Any]) -> bool:
    """Return whether a transform is recommended for a bundle."""

    return transform_id in set(str(item) for item in bundle.get("recommended_transforms", []))


def build_generated_feature_name(column: str, transform_id: str) -> str:
    """Build a generated feature column name for a base column and transform."""

    suffix = TRANSFORM_SUFFIXES.get(transform_id, transform_id)

    if transform_id == "red_minus_blue":
        return f"{column}_diff"
    if transform_id == "blue_minus_red":
        return f"{column}_reverse_diff"
    if transform_id == "market_gap":
        return f"{column}_market_gap"
    if transform_id == "style_matchup":
        return f"{column}_style_matchup"

    return f"{column}_{suffix}"


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """De-duplicate values while preserving first-seen order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def feature_graph_plan_to_dict(plan: FeatureGraphPlan) -> dict[str, Any]:
    """Serialize a feature graph plan to a plain dictionary."""

    payload = asdict(plan)
    payload["counts"] = {
        "selected_bundles": len(plan.selected_bundles),
        "selected_transforms": len(plan.selected_transforms),
        "raw_feature_columns": len(plan.raw_feature_columns),
        "generated_feature_columns": len(plan.generated_feature_columns),
        "passthrough_feature_columns": len(plan.passthrough_feature_columns),
    }
    return payload


def write_feature_graph_plan(plan: FeatureGraphPlan, output_path: str | Path) -> Path:
    """Write a feature graph plan to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(feature_graph_plan_to_dict(plan), indent=2),
        encoding="utf-8",
    )
    return path


def print_feature_graph_plan(plan: FeatureGraphPlan) -> None:
    """Print a compact feature graph plan summary."""

    print("=" * 80)
    print("UFC FEATURE GRAPH PLAN")
    print("=" * 80)
    print(f"View ID                 : {plan.view_id}")
    print(f"View family             : {plan.view_family}")
    print(f"Bundles                 : {len(plan.selected_bundles)}")
    print(f"Transforms              : {len(plan.selected_transforms)}")
    print(f"Raw feature columns     : {len(plan.raw_feature_columns)}")
    print(f"Generated feature cols  : {len(plan.generated_feature_columns)}")
    print(f"Passthrough feature cols: {len(plan.passthrough_feature_columns)}")

    print("\nSelected bundles:")
    for item in plan.selected_bundles:
        print(f"  - {item}")

    print("\nSelected transforms:")
    for item in plan.selected_transforms:
        print(f"  - {item}")

    print("\nGenerated feature preview:")
    for item in plan.generated_feature_columns[:40]:
        print(f"  - {item}")
    if len(plan.generated_feature_columns) > 40:
        print(f"  ... {len(plan.generated_feature_columns) - 40} more")

    print("DONE")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    plan = resolve_feature_graph_plan(
        feature_view_config_path=args.feature_view_config,
        bundle_registry_path=args.bundle_registry,
        transform_registry_path=args.transform_registry,
    )
    print_feature_graph_plan(plan)

    if args.output:
        output_path = write_feature_graph_plan(plan, args.output)
        print(f"Wrote feature graph plan: {output_path}")


if __name__ == "__main__":
    main()
