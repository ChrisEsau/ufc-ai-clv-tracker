"""Export the full rolling feature inventory.

This runner reads the current rolling feature parquet artifact and writes a
machine-readable YAML inventory to:

    configs/features/full_rolling_feature_inventory.yaml

It is intentionally conservative: it fails if the observed rolling feature
column count does not match the expected protected schema count, or if the
current Moneyline V5 feature-selection rule does not produce 124 features.

Run from repo root:

    python -m pipeline.features.export_rolling_feature_inventory

Optional:

    python -m pipeline.features.export_rolling_feature_inventory \
        --input data/features/UFC_enhanced_rolling_features_EWM.parquet \
        --output configs/features/full_rolling_feature_inventory.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_INPUT_PATH = Path("data/features/UFC_enhanced_rolling_features_EWM.parquet")
DEFAULT_OUTPUT_PATH = Path("configs/features/full_rolling_feature_inventory.yaml")
EXPECTED_ROLLING_COLUMNS = 483
EXPECTED_MONEYLINE_V5_FEATURES = 124

REGISTERED_ENGINEERED_FEATURES = [
    "age_diff",
    "height_diff",
    "reach_diff",
    "weight_diff",
    "striking_edge",
    "grappling_edge",
    "finish_volatility",
    "wrestling_pressure_vs_defense",
    "reach_striking_combo",
    "chin_risk_diff",
    "experience_ratio_diff",
    "aggression_index_diff",
    "age_squared_diff",
    "pressure_striking_adv_diff",
    "wrestling_mismatch_diff",
    "submission_mismatch_diff",
]

UNSAFE_MONEYLINE_PREFIXES = (
    "r_pre_",
    "b_pre_",
    "R_",
    "B_",
    "r_",
    "b_",
)


def yaml_quote(value: object) -> str:
    """Return a simple, safe YAML scalar representation."""
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def classify_layer(column: str) -> str:
    """Classify the broad feature layer for a column name."""
    metadata_cols = {
        "event_id",
        "event_name",
        "date",
        "location",
        "fight_id",
        "division",
        "title_fight",
        "method",
        "finish_round",
        "match_time_sec",
        "total_rounds",
        "referee",
    }
    target_cols = {"target", "winner", "winner_id"}

    if column in metadata_cols:
        return "metadata"
    if column in target_cols:
        return "target_result"
    if column in REGISTERED_ENGINEERED_FEATURES:
        return "moneyline"
    if column.endswith("_diff"):
        return "moneyline"
    if column.startswith(("r_pre_", "b_pre_", "r_ewm_", "b_ewm_", "r_recent_form_", "b_recent_form_")):
        return "base"
    if column.startswith(("r_", "b_")):
        return "base"
    if column.startswith(("ewm_", "recent_", "recent_form_")):
        return "base"
    return "base"


def classify_family(column: str) -> str:
    """Classify a more specific feature family for a column name."""
    if column in {"target", "winner", "winner_id"}:
        return "target_result"
    if column in REGISTERED_ENGINEERED_FEATURES:
        return "engineered_matchup"
    if column.startswith(("r_pre_", "b_pre_")):
        return "prefight_state"
    if column.startswith(("r_ewm_", "b_ewm_")):
        return "corner_ewm_recent_form"
    if column.startswith("ewm_") and column.endswith("_diff"):
        return "ewm_diff"
    if column.startswith(("r_recent_form_", "b_recent_form_")):
        return "corner_recent_form"
    if column.startswith("recent_form_") and column.endswith("_diff"):
        return "recent_form_diff"
    if column.endswith("_diff"):
        return "career_or_matchup_diff"
    if column.startswith(("r_", "b_")):
        return "raw_corner_stat_or_identifier"
    return "metadata_or_base"


def is_current_moneyline_v5_feature(column: str, all_columns: Iterable[str]) -> bool:
    """Mirror the current training notebook safe_cols selection rule."""
    del all_columns  # kept for future expansion; rule is column-local today.
    return column.endswith("_diff") or column in REGISTERED_ENGINEERED_FEATURES


def build_moneyline_v5_features(columns: list[str]) -> list[str]:
    """Build the current V5 moneyline feature list exactly like the notebook."""
    safe_cols: list[str] = []

    for col in columns:
        if col.endswith("_diff"):
            safe_cols.append(col)
        elif col in REGISTERED_ENGINEERED_FEATURES:
            safe_cols.append(col)

    safe_cols = list(dict.fromkeys(safe_cols))

    unsafe_cols = [
        col
        for col in safe_cols
        if col.startswith(UNSAFE_MONEYLINE_PREFIXES)
    ]

    if unsafe_cols:
        raise ValueError(
            "Unsafe raw red/blue fighter columns detected in Moneyline V5 features: "
            + ", ".join(unsafe_cols)
        )

    return safe_cols


def write_inventory(df: pd.DataFrame, output_path: Path) -> None:
    """Write the full schema inventory YAML."""
    columns = list(df.columns)
    moneyline_v5_features = build_moneyline_v5_features(columns)

    if len(columns) != EXPECTED_ROLLING_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_ROLLING_COLUMNS} rolling columns, found {len(columns)}. "
            "Refusing to write inventory because the protected schema count changed."
        )

    if len(moneyline_v5_features) != EXPECTED_MONEYLINE_V5_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_MONEYLINE_V5_FEATURES} Moneyline V5 features, "
            f"found {len(moneyline_v5_features)}. Refusing to write inventory."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.extend(
        [
            "# Full Rolling Feature Inventory",
            "# Auto-generated by pipeline.features.export_rolling_feature_inventory.",
            "# Do not hand-edit column entries unless intentionally correcting metadata.",
            "",
            "inventory_name: full_rolling_feature_inventory",
            "source_artifact: data/features/UFC_enhanced_rolling_features_EWM.parquet",
            "source_notebook: UFC_rolling_dataset_V4_refactored.ipynb",
            "generated_by: pipeline.features.export_rolling_feature_inventory",
            "schema_status:",
            f"  expected_column_count: {EXPECTED_ROLLING_COLUMNS}",
            f"  observed_column_count: {len(columns)}",
            "  extraction_status: complete",
            "  full_column_list_present: true",
            "  refactor_blocked_until_full_column_list_present: false",
            "current_moneyline_v5_dependency:",
            "  contract_file: configs/features/current_moneyline_v5_features.yaml",
            f"  expected_training_feature_count: {EXPECTED_MONEYLINE_V5_FEATURES}",
            f"  observed_training_feature_count: {len(moneyline_v5_features)}",
            "  selection_rule: all_columns_ending_in_diff_plus_registered_engineered_features",
            "columns:",
        ]
    )

    for idx, col in enumerate(columns, start=1):
        dtype = str(df[col].dtype)
        layer = classify_layer(col)
        family = classify_family(col)
        used_by_v5 = col in moneyline_v5_features
        lines.extend(
            [
                f"  - ordinal: {idx}",
                f"    name: {yaml_quote(col)}",
                f"    dtype: {yaml_quote(dtype)}",
                f"    feature_layer: {yaml_quote(layer)}",
                f"    feature_family: {yaml_quote(family)}",
                "    preservation_status: protected",
                f"    used_by_current_moneyline_v5: {str(used_by_v5).lower()}",
            ]
        )

    lines.extend(
        [
            "validation_rules:",
            "  - full_column_inventory_must_contain_483_columns_before_refactor",
            "  - every_column_must_have_feature_layer_assignment",
            "  - every_column_must_have_preservation_status",
            "  - current_moneyline_v5_features_must_remain_available",
            "  - no_column_may_be_dropped_without_explicit_approval",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full rolling feature inventory YAML.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to rolling feature parquet artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write full rolling feature inventory YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    write_inventory(df=df, output_path=args.output)

    print("Wrote rolling feature inventory:", args.output)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Moneyline V5 features:", len(build_moneyline_v5_features(list(df.columns))))


if __name__ == "__main__":
    main()
