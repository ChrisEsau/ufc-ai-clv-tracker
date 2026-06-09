"""Validate generic transform engine against the current moneyline feature view.

This is a one-off migration validation script. It confirms that simple generic
red/blue transforms reproduce handcrafted current moneyline differentials.

Run from repo root:

    python archive/migration_validation/run_validate_transform_engine.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features.transform_engine import apply_red_blue_transforms


FEATURE_VIEW_PATH = "data/features/moneyline_feature_view.parquet"

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

COMPARISON_COLUMNS = [
    "elo_diff",
    "splm_diff",
    "td_avg_diff",
    "sub_avg_diff",
    "avg_fight_time_diff",
    "days_since_last_fight_diff",
]


def main() -> None:
    """Run transform-engine parity checks."""

    print("=" * 80)
    print("TRANSFORM ENGINE VALIDATION")
    print("=" * 80)
    print(f"Feature view path: {FEATURE_VIEW_PATH}")
    print(f"Base columns     : {BASE_COLUMNS}")
    print(f"Transforms       : {TRANSFORMS}")

    df = pd.read_parquet(FEATURE_VIEW_PATH)
    print(f"Input shape      : {df.shape}")

    result = apply_red_blue_transforms(
        df=df,
        base_columns=BASE_COLUMNS,
        transforms=TRANSFORMS,
        red_prefix="r_pre_",
        blue_prefix="b_pre_",
    )

    print(f"Generated columns: {len(result.generated_columns)}")
    print(f"Missing pairs    : {len(result.missing_source_pairs)}")
    if result.missing_source_pairs:
        print("First missing pairs:")
        for item in result.missing_source_pairs[:20]:
            print(f"  - {item}")

    print("\nParity checks:")
    status_counts: dict[str, int] = {}

    for column in COMPARISON_COLUMNS:
        if column not in df.columns:
            status = "MISSING_EXISTING_COLUMN"
            max_abs_diff = np.nan
            mean_abs_diff = np.nan
            nonzero_rows = np.nan
        elif column not in result.dataframe.columns:
            status = "MISSING_GENERATED_COLUMN"
            max_abs_diff = np.nan
            mean_abs_diff = np.nan
            nonzero_rows = np.nan
        else:
            old_values = pd.to_numeric(df[column], errors="coerce")
            new_values = pd.to_numeric(result.dataframe[column], errors="coerce")
            delta = (old_values - new_values).abs()
            comparable = delta.dropna()
            max_abs_diff = float(comparable.max()) if len(comparable) else 0.0
            mean_abs_diff = float(comparable.mean()) if len(comparable) else 0.0
            nonzero_rows = int((comparable > 1e-9).sum())
            status = "PASS" if nonzero_rows == 0 else "FAIL_VALUE_MISMATCH"

        status_counts[status] = status_counts.get(status, 0) + 1
        print(
            f"{column:30s} {status:24s} "
            f"max={max_abs_diff} mean={mean_abs_diff} nonzero={nonzero_rows}"
        )

    print("\nStatus counts:")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    if any(status != "PASS" for status in status_counts):
        raise SystemExit("Transform engine validation failed.")

    print("DONE")


if __name__ == "__main__":
    main()
