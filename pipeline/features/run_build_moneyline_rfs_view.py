"""Build the experimental moneyline RFS feature view.

Run from repo root:

    python -m pipeline.features.run_build_moneyline_rfs_view

This runner writes experimental artifacts only:

    data/features/moneyline_rfs_feature_view.parquet
    data/audits/moneyline_rfs_feature_view_validation.parquet

It does not modify the production moneyline feature view or production model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    AUDITS_DIR,
    FEATURES_DIR,
    FIGHTER_STATE_HISTORY_PATH,
    MASTER_PATH,
    ensure_data_dirs,
)
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling
from pipeline.features.views.moneyline_round_fighter_state import (
    build_moneyline_feature_view_with_round_state,
    summarize_moneyline_round_state_view,
)
from ufc_feature_engineering import add_v5_engineered_features, get_engineered_feature_list


MONEYLINE_RFS_FEATURE_VIEW_PATH = FEATURES_DIR / "moneyline_rfs_feature_view.parquet"
MONEYLINE_RFS_FEATURE_VIEW_VALIDATION_PATH = AUDITS_DIR / "moneyline_rfs_feature_view_validation.parquet"


def _audit_row(
    check_name: str,
    status: str,
    observed: object,
    details: str = "",
    severity: str = "fatal",
) -> dict[str, object]:
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "observed": observed,
        "details": details,
    }


def build_validation_audit(
    *,
    prepared_df: pd.DataFrame,
    rfs_view_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build validation audit for the experimental RFS feature view."""

    rows: list[dict[str, object]] = []

    rows.append(
        _audit_row(
            "row_count_matches_prepared_fights",
            "PASS" if len(rfs_view_df) == len(prepared_df) else "FAIL",
            f"{len(rfs_view_df)} / {len(prepared_df)}",
        )
    )

    fight_unique = (
        int(rfs_view_df["fight_id"].nunique())
        if "fight_id" in rfs_view_df.columns
        else 0
    )
    rows.append(
        _audit_row(
            "fight_id_unique",
            "PASS" if fight_unique == len(rfs_view_df) else "FAIL",
            f"{fight_unique} unique / {len(rfs_view_df)} rows",
        )
    )

    target_nulls = (
        int(rfs_view_df["target"].isna().sum())
        if "target" in rfs_view_df.columns
        else len(rfs_view_df)
    )
    rows.append(
        _audit_row(
            "target_present_non_null",
            "PASS" if "target" in rfs_view_df.columns and target_nulls == 0 else "FAIL",
            target_nulls,
            "target column missing or contains nulls" if target_nulls else "",
        )
    )

    date_nulls = (
        int(pd.to_datetime(rfs_view_df["date"], errors="coerce").isna().sum())
        if "date" in rfs_view_df.columns
        else len(rfs_view_df)
    )
    rows.append(
        _audit_row(
            "date_present_parseable",
            "PASS" if "date" in rfs_view_df.columns and date_nulls == 0 else "FAIL",
            date_nulls,
            "date column missing or contains unparseable values" if date_nulls else "",
        )
    )

    current_fight_obs_cols = [
        column
        for column in rfs_view_df.columns
        if column.startswith("rfs_") and "_fight_" in column
    ]
    rows.append(
        _audit_row(
            "no_rfs_current_fight_observation_columns",
            "PASS" if not current_fight_obs_cols else "FAIL",
            len(current_fight_obs_cols),
            ", ".join(current_fight_obs_cols[:25]),
        )
    )

    side_specific_rfs_cols = [
        column
        for column in rfs_view_df.columns
        if column.startswith(("r_rfs_", "b_rfs_"))
    ]
    rows.append(
        _audit_row(
            "side_specific_rfs_columns_dropped",
            "PASS" if not side_specific_rfs_cols else "FAIL",
            len(side_specific_rfs_cols),
            ", ".join(side_specific_rfs_cols[:25]),
        )
    )

    diff_cols = [
        column
        for column in rfs_view_df.columns
        if column.startswith("rfs_") and column.endswith("_diff")
    ]
    rows.append(
        _audit_row(
            "rfs_diff_columns_exist",
            "PASS" if diff_cols else "FAIL",
            len(diff_cols),
        )
    )

    for family_key in ["traj", "suppress", "wrestle", "def"]:
        either_col = f"rfs_{family_key}_either_has_state"
        both_col = f"rfs_{family_key}_both_have_state"

        either_count = (
            int(pd.to_numeric(rfs_view_df[either_col], errors="coerce").fillna(0).sum())
            if either_col in rfs_view_df.columns
            else 0
        )
        both_count = (
            int(pd.to_numeric(rfs_view_df[both_col], errors="coerce").fillna(0).sum())
            if both_col in rfs_view_df.columns
            else 0
        )

        rows.append(
            _audit_row(
                f"rfs_{family_key}_availability",
                "PASS" if either_count > 0 else "WARN",
                f"either={either_count}, both={both_count}",
                severity="warn",
            )
        )

    numeric_rfs_cols = [
        column
        for column in rfs_view_df.columns
        if column.startswith("rfs_")
        and pd.api.types.is_numeric_dtype(rfs_view_df[column])
    ]

    inf_count = 0
    if numeric_rfs_cols:
        numeric = rfs_view_df[numeric_rfs_cols].apply(pd.to_numeric, errors="coerce")
        inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())

    rows.append(
        _audit_row(
            "no_infinite_rfs_values",
            "PASS" if inf_count == 0 else "FAIL",
            inf_count,
        )
    )

    return pd.DataFrame(rows)


def main() -> None:
    """Build and save the experimental moneyline RFS feature view."""

    ensure_data_dirs()

    print("=" * 80)
    print("BUILD EXPERIMENTAL MONEYLINE RFS FEATURE VIEW")
    print("=" * 80)
    print(f"Master path              : {MASTER_PATH}")
    print(f"Fighter state path       : {FIGHTER_STATE_HISTORY_PATH}")
    print(f"RFS feature view path    : {MONEYLINE_RFS_FEATURE_VIEW_PATH}")
    print(f"RFS validation path      : {MONEYLINE_RFS_FEATURE_VIEW_VALIDATION_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    print(f"Master shape             : {master_df.shape}")

    prepared_df = prepare_master_for_rolling(master_df)
    print(f"Prepared fight shape     : {prepared_df.shape}")

    fighter_state_history_df = pd.read_parquet(FIGHTER_STATE_HISTORY_PATH)
    print(f"Fighter state shape      : {fighter_state_history_df.shape}")

    rfs_view_df = build_moneyline_feature_view_with_round_state(
        prepared_fights_df=prepared_df,
        fighter_state_history_df=fighter_state_history_df,
        add_round_state_diffs=True,
        include_fight_observations=False,
        keep_side_features=False,
    )

    rfs_view_df = add_v5_engineered_features(rfs_view_df)

    duplicate_columns = rfs_view_df.columns[rfs_view_df.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Duplicate columns after engineering: {len(duplicate_columns)}")
        print(f"Deduplicating columns with keep='last': {duplicate_columns[:25]}")
        rfs_view_df = rfs_view_df.loc[:, ~rfs_view_df.columns.duplicated(keep="last")].copy()

    engineered_features = get_engineered_feature_list()
    missing_engineered_features = [
        column for column in engineered_features if column not in rfs_view_df.columns
    ]
    if missing_engineered_features:
        raise ValueError(
            "Experimental RFS feature view missing engineered features: "
            f"{missing_engineered_features}"
        )

    audit_df = build_validation_audit(
        prepared_df=prepared_df,
        rfs_view_df=rfs_view_df,
    )

    # Keep audit parquet schema stable. Mixed object columns can fail pyarrow
    # serialization when some rows contain ints and others contain strings.
    audit_df["observed"] = audit_df["observed"].astype(str)
    audit_df["details"] = audit_df["details"].astype(str)

    print(f"RFS view shape           : {rfs_view_df.shape}")
    print(f"Unique fights            : {rfs_view_df['fight_id'].nunique() if not rfs_view_df.empty else 0}")
    print(f"Engineered features      : {len(engineered_features)}")

    summary = summarize_moneyline_round_state_view(rfs_view_df)
    print()
    print("RFS SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("VALIDATION")
    print(audit_df.to_string(index=False))

    fatal_failures = audit_df[
        audit_df["severity"].eq("fatal") & audit_df["status"].eq("FAIL")
    ]
    if not fatal_failures.empty:
        raise ValueError(
            "Experimental RFS feature view failed validation:\n"
            f"{fatal_failures.to_string(index=False)}"
        )

    rfs_view_df.to_parquet(MONEYLINE_RFS_FEATURE_VIEW_PATH, index=False)
    audit_df.to_parquet(MONEYLINE_RFS_FEATURE_VIEW_VALIDATION_PATH, index=False)

    print()
    print("Saved experimental RFS feature view successfully.")
    print("DONE")


if __name__ == "__main__":
    main()
