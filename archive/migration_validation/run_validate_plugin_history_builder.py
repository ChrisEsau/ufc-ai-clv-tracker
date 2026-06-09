"""Validate plugin fighter-state builder against the legacy builder.

Run from repo root:

    python -m archive.migration_validation.run_validate_plugin_history_builder
"""

from __future__ import annotations

import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling
from pipeline.features.state.ewm_state import add_ewm_state_features
from pipeline.features.state.history_builder import build_fighter_state_history
from pipeline.features.state.plugin_history_builder import build_plugin_fighter_state_history

KEY_COLUMNS = ["fight_id", "fighter_id", "corner"]


def main() -> None:
    print("=" * 80)
    print("PLUGIN HISTORY BUILDER PARITY VALIDATION")
    print("=" * 80)
    print(f"Master path: {MASTER_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    prepared_df = prepare_master_for_rolling(master_df)

    legacy_df = add_ewm_state_features(build_fighter_state_history(prepared_df))
    plugin_df = build_plugin_fighter_state_history(prepared_df, add_ewm=True)

    print(f"Prepared fights: {len(prepared_df):,}")
    print(f"Legacy rows    : {len(legacy_df):,}")
    print(f"Plugin rows    : {len(plugin_df):,}")
    print(f"Legacy columns : {len(legacy_df.columns):,}")
    print(f"Plugin columns : {len(plugin_df.columns):,}")

    if len(legacy_df) != len(plugin_df):
        raise SystemExit("Row count mismatch between legacy and plugin builders.")

    shared_numeric = [
        column for column in legacy_df.columns
        if column in plugin_df.columns
        and column not in KEY_COLUMNS
        and pd.api.types.is_numeric_dtype(legacy_df[column])
        and pd.api.types.is_numeric_dtype(plugin_df[column])
    ]

    merged = legacy_df[KEY_COLUMNS + shared_numeric].merge(
        plugin_df[KEY_COLUMNS + shared_numeric],
        on=KEY_COLUMNS,
        suffixes=("_legacy", "_plugin"),
        how="inner",
    )

    print(f"Matched rows   : {len(merged):,}")
    print(f"Compared cols  : {len(shared_numeric):,}")

    if len(merged) != len(legacy_df):
        raise SystemExit(
            f"Matched rows mismatch: matched={len(merged):,}, legacy={len(legacy_df):,}"
        )

    audit = audit_columns(merged, shared_numeric)
    print("\nStatus counts:")
    print(audit["status"].value_counts().to_string())
    print("\nParity checks:")
    print(audit.to_string(index=False))

    if (audit["status"] != "PASS").any():
        raise SystemExit("Plugin history builder parity validation failed.")

    print("DONE")


def audit_columns(merged: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        legacy = pd.to_numeric(merged[f"{column}_legacy"], errors="coerce")
        plugin = pd.to_numeric(merged[f"{column}_plugin"], errors="coerce")
        delta = (legacy - plugin).abs().dropna()
        nonzero_rows = int((delta > 1e-9).sum())
        rows.append({
            "feature_name": column,
            "status": "PASS" if nonzero_rows == 0 else "FAIL",
            "max_abs_diff": float(delta.max()) if len(delta) else 0.0,
            "mean_abs_diff": float(delta.mean()) if len(delta) else 0.0,
            "nonzero_rows": nonzero_rows,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
