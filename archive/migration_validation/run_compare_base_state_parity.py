"""One-off base fighter-state parity audit for the feature refactor.

Compares legacy rolling r_pre_/b_pre_ columns against the new moneyline feature
view generated from fighter_state_history. This isolates whether parity problems
start in base fighter state before EWM/recent-form layers.

Run from repo root:

    python archive/migration_validation/run_compare_base_state_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LEGACY_ROLLING_PATH = Path("data/features/UFC_enhanced_rolling_features_EWM.parquet")
NEW_MONEYLINE_VIEW_PATH = Path("data/features/moneyline_feature_view.parquet")
AUDIT_OUTPUT_PATH = Path("archive/migration_validation/base_state_parity_audit.parquet")

TOLERANCE = 1e-9


def _status(max_abs_diff: float, old_nulls: int, new_nulls: int) -> str:
    """Return a simple parity status for one feature."""

    if old_nulls != new_nulls:
        return "FAIL_NULL_MISMATCH"
    if pd.isna(max_abs_diff):
        return "FAIL_NO_COMPARISON"
    if max_abs_diff <= TOLERANCE:
        return "PASS"
    if max_abs_diff <= 1e-6:
        return "WARN_TINY_DIFF"
    return "FAIL_VALUE_MISMATCH"


def _base_state_columns(df: pd.DataFrame) -> list[str]:
    """Return legacy-style r_pre_/b_pre_ columns excluding EWM/recent-form columns."""

    return sorted(
        column
        for column in df.columns
        if (column.startswith("r_pre_") or column.startswith("b_pre_"))
    )


def compare_feature(old_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict[str, object]:
    """Compare one base state feature between legacy and new feature artifacts."""

    old_series = old_df[feature]
    new_series = new_df[feature]

    comparable_mask = old_series.notna() & new_series.notna()
    diffs = (old_series[comparable_mask] - new_series[comparable_mask]).abs()

    max_abs_diff = float(diffs.max()) if len(diffs) else np.nan
    mean_abs_diff = float(diffs.mean()) if len(diffs) else np.nan
    exact_match_pct = float((diffs <= TOLERANCE).mean()) if len(diffs) else np.nan

    old_nulls = int(old_series.isna().sum())
    new_nulls = int(new_series.isna().sum())

    return {
        "feature_name": feature,
        "old_nulls": old_nulls,
        "new_nulls": new_nulls,
        "comparable_rows": int(comparable_mask.sum()),
        "exact_match_pct": exact_match_pct,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "old_mean": float(old_series.mean()),
        "new_mean": float(new_series.mean()),
        "status": _status(max_abs_diff, old_nulls, new_nulls),
    }


def main() -> None:
    """Run the base state migration parity audit."""

    print("=" * 80)
    print("BASE FIGHTER STATE MIGRATION PARITY AUDIT")
    print("=" * 80)
    print(f"Legacy path: {LEGACY_ROLLING_PATH}")
    print(f"New path   : {NEW_MONEYLINE_VIEW_PATH}")
    print(f"Audit path : {AUDIT_OUTPUT_PATH}")

    old_df = pd.read_parquet(LEGACY_ROLLING_PATH).sort_values(["date", "event_id", "fight_id"]).reset_index(drop=True)
    new_df = pd.read_parquet(NEW_MONEYLINE_VIEW_PATH).sort_values(["date", "event_id", "fight_id"]).reset_index(drop=True)

    print(f"Legacy shape: {old_df.shape}")
    print(f"New shape   : {new_df.shape}")

    if len(old_df) != len(new_df):
        raise ValueError(f"Row count mismatch: legacy={len(old_df)}, new={len(new_df)}")

    old_features = set(_base_state_columns(old_df))
    new_features = set(_base_state_columns(new_df))
    common_features = sorted(old_features & new_features)

    missing_from_new = sorted(old_features - new_features)
    missing_from_old = sorted(new_features - old_features)

    print(f"Legacy base features: {len(old_features)}")
    print(f"New base features   : {len(new_features)}")
    print(f"Common base features: {len(common_features)}")
    print(f"Missing from new    : {len(missing_from_new)}")
    print(f"Missing from legacy : {len(missing_from_old)}")

    audit_rows = [compare_feature(old_df, new_df, feature) for feature in common_features]
    audit_df = pd.DataFrame(audit_rows)

    AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_parquet(AUDIT_OUTPUT_PATH, index=False)

    print("Status counts:")
    print(audit_df["status"].value_counts(dropna=False).to_string())

    failures = audit_df[~audit_df["status"].isin(["PASS", "WARN_TINY_DIFF"])]
    if not failures.empty:
        print("\nTop base-state parity failures:")
        print(
            failures.sort_values("max_abs_diff", ascending=False)
            .head(30)[["feature_name", "status", "old_nulls", "new_nulls", "max_abs_diff", "mean_abs_diff"]]
            .to_string(index=False)
        )

    if missing_from_new:
        print("\nMissing from new:")
        print(missing_from_new[:30])

    print("DONE")


if __name__ == "__main__":
    main()
