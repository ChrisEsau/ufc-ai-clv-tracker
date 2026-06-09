"""Validate registry-loaded transform plugins against production feature columns.

This script proves that the generic transform engine is loading plugins from
``configs/features/transform_registry.yaml`` and still reproducing known
production moneyline columns.

Run from repo root:

    python -m archive.migration_validation.run_validate_transform_plugins
"""

from __future__ import annotations

import importlib

import pandas as pd
import yaml

from pipeline.features.transform_engine import apply_red_blue_transforms, load_transform_plugins


FEATURE_VIEW_PATH = "data/features/moneyline_feature_view.parquet"
TRANSFORM_REGISTRY_PATH = "configs/features/transform_registry.yaml"

BASE_COLUMNS = [
    "elo",
    "splm",
    "td_avg",
    "sub_avg",
    "avg_fight_time",
    "days_since_last_fight",
]

TRANSFORMS = [
    "red_minus_blue",
    "absolute_gap",
    "ratio",
]

PRODUCTION_COMPARISON_COLUMNS = [
    "elo_diff",
    "splm_diff",
    "td_avg_diff",
    "sub_avg_diff",
    "avg_fight_time_diff",
    "days_since_last_fight_diff",
]


def main() -> None:
    """Run plugin loading and parity validation."""

    print("=" * 80)
    print("TRANSFORM PLUGIN VALIDATION")
    print("=" * 80)
    print(f"Feature view path      : {FEATURE_VIEW_PATH}")
    print(f"Transform registry path: {TRANSFORM_REGISTRY_PATH}")

    validate_registry_imports()

    plugins = load_transform_plugins(tuple(TRANSFORMS), TRANSFORM_REGISTRY_PATH)
    print("\nLoaded plugins:")
    for plugin in plugins:
        print(f"  - {plugin.transform_id} -> suffix={plugin.output_suffix}")

    df = pd.read_parquet(FEATURE_VIEW_PATH)
    result = apply_red_blue_transforms(
        df=df,
        base_columns=BASE_COLUMNS,
        transforms=TRANSFORMS,
        red_prefix="r_pre_",
        blue_prefix="b_pre_",
        transform_registry_path=TRANSFORM_REGISTRY_PATH,
    )

    print(f"\nInput shape      : {df.shape}")
    print(f"Generated columns: {len(result.generated_columns)}")
    print(f"Missing pairs    : {len(result.missing_source_pairs)}")

    rows = []
    for column in PRODUCTION_COMPARISON_COLUMNS:
        old_values = pd.to_numeric(df[column], errors="coerce")
        new_values = pd.to_numeric(result.dataframe[column], errors="coerce")
        delta = (old_values - new_values).abs().dropna()
        nonzero_rows = int((delta > 1e-9).sum())
        rows.append(
            {
                "feature_name": column,
                "status": "PASS" if nonzero_rows == 0 else "FAIL",
                "max_abs_diff": float(delta.max()) if len(delta) else 0.0,
                "mean_abs_diff": float(delta.mean()) if len(delta) else 0.0,
                "nonzero_rows": nonzero_rows,
            }
        )

    audit_df = pd.DataFrame(rows)
    print("\nParity checks:")
    print(audit_df.to_string(index=False))

    failures = audit_df[audit_df["status"] != "PASS"]
    if not failures.empty:
        raise SystemExit("Transform plugin validation failed.")

    print("DONE")


def validate_registry_imports() -> None:
    """Validate active transform plugin entries are importable and callable."""

    with open(TRANSFORM_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    print("\nRegistry import checks:")
    for transform_id, entry in registry["transforms"].items():
        if str(entry.get("status", "")).lower() != "active":
            continue

        plugin_path = entry.get("plugin")
        function_name = entry.get("function")
        module = importlib.import_module(plugin_path)
        func = getattr(module, function_name)
        if not callable(func):
            raise TypeError(f"Plugin function not callable: {plugin_path}.{function_name}")
        print(f"  - {transform_id}: OK ({plugin_path}.{function_name})")


if __name__ == "__main__":
    main()
