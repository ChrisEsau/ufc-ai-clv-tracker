"""One-off migration parity audit for moneyline feature refactor.

Compares the legacy rolling feature artifact against the new moneyline feature
view built from fighter_state_history. This script is intentionally archived and
is not part of the active production pipeline.

Run from repo root:

    python archive/migration_validation/run_compare_moneyline_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LEGACY_ROLLING_PATH = Path("data/features/UFC_enhanced_rolling_features_EWM.parquet")
NEW_MONEYLINE_VIEW_PATH = Path("data/features/moneyline_feature_view.parquet")
AUDIT_OUTPUT_PATH = Path("archive/migration_validation/moneyline_parity_audit.parquet")

FEATURE_SUFFIX = "_diff"
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


def compare_feature(old_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict[str, object]:
    """Compare one numeric feature between legacy and new feature artifacts."""

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
    """Run the moneyline migration parity audit."""

    print("=" * 80)
    print("MONEYLINE FEATURE MIGRATION PARITY AUDIT")
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

    old_features = {column for column in old_df.columns if column.endswith(FEATURE_SUFFIX)}
    new_features = {column for column in new_df.columns if column.endswith(FEATURE_SUFFIX)}
    common_features = sorted(old_features & new_features)

    missing_from_new = sorted(old_features - new_features)
    missing_from_old = sorted(new_features - old_features)

    print(f"Legacy diff features : {len(old_features)}")
    print(f"New diff features    : {len(new_features)}")
    print(f"Common diff features : {len(common_features)}")
    print(f"Missing from new     : {len(missing_from_new)}")
    print(f"Missing from legacy  : {len(missing_from_old)}")

    audit_rows = [compare_feature(old_df, new_df, feature) for feature in common_features]
    audit_df = pd.DataFrame(audit_rows)

    AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_parquet(AUDIT_OUTPUT_PATH, index=False)

    print("Status counts:")
    print(audit_df["status"].value_counts(dropna=False).to_string())

    if missing_from_new:
        print("\nFirst missing-from-new features:")
        print(missing_from_new[:20])

    failures = audit_df[~audit_df["status"].isin(["PASS", "WARN_TINY_DIFF"])]
    if not failures.empty:
        print("\nTop parity failures:")
        print(
            failures.sort_values("max_abs_diff", ascending=False)
            .head(20)[["feature_name", "status", "old_nulls", "new_nulls", "max_abs_diff", "mean_abs_diff"]]
            .to_string(index=False)
        )

    print("DONE")


if __name__ == "__main__":
    main()
