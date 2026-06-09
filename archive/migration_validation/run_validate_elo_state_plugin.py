"""Validate the Elo raw fighter feature plugin against production state history.

This script recomputes the Elo feature family from master fight rows using the
new raw_fighter_features.elo_state plugin and compares the results against the
existing fighter_state_history.parquet artifact.

Run from repo root:

    python -m archive.migration_validation.run_validate_elo_state_plugin
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from pipeline.common.paths import FIGHTER_STATE_HISTORY_PATH, MASTER_PATH
from pipeline.features.raw_fighter_features import elo_state
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling


COMPARE_COLUMNS = [
    "elo",
    "avg_opponent_elo",
    "best_win_elo",
    "worst_loss_elo",
]

KEY_COLUMNS = [
    "fight_id",
    "fighter_id",
    "corner",
]


def main() -> None:
    """Run Elo plugin shadow validation."""

    print("=" * 80)
    print("ELO STATE PLUGIN SHADOW VALIDATION")
    print("=" * 80)
    print(f"Master path        : {MASTER_PATH}")
    print(f"State history path : {FIGHTER_STATE_HISTORY_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    prepared_df = prepare_master_for_rolling(master_df)
    production_df = pd.read_parquet(FIGHTER_STATE_HISTORY_PATH)

    shadow_df = build_shadow_elo_history(prepared_df)

    print(f"Prepared fights    : {len(prepared_df):,}")
    print(f"Production rows    : {len(production_df):,}")
    print(f"Shadow rows        : {len(shadow_df):,}")

    expected_rows = len(prepared_df) * 2
    if len(shadow_df) != expected_rows:
        raise SystemExit(
            f"Shadow row count mismatch: expected {expected_rows:,}, observed {len(shadow_df):,}"
        )

    compare_df = production_df[KEY_COLUMNS + COMPARE_COLUMNS].merge(
        shadow_df[KEY_COLUMNS + COMPARE_COLUMNS],
        on=KEY_COLUMNS,
        suffixes=("_production", "_shadow"),
        how="inner",
    )

    print(f"Matched rows       : {len(compare_df):,}")
    if len(compare_df) != len(production_df):
        raise SystemExit(
            "Matched row count does not equal production rows. "
            f"matched={len(compare_df):,}, production={len(production_df):,}"
        )

    rows = []
    for column in COMPARE_COLUMNS:
        production_values = pd.to_numeric(compare_df[f"{column}_production"], errors="coerce")
        shadow_values = pd.to_numeric(compare_df[f"{column}_shadow"], errors="coerce")
        delta = (production_values - shadow_values).abs().dropna()
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
        raise SystemExit("Elo state plugin validation failed.")

    print("DONE")


def build_shadow_elo_history(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute Elo-family prefight rows using the Elo plugin."""

    states: defaultdict[str, dict[str, Any]] = defaultdict(elo_state.initial_state)
    rows: list[dict[str, Any]] = []

    for source_row_index, row in prepared_df.reset_index(drop=True).iterrows():
        r_id = str(row["r_id"])
        b_id = str(row["b_id"])

        r_features = elo_state.calculate(
            fighter_history=pd.DataFrame(),
            fight_row=row,
            context={"state": states[r_id]},
        )
        b_features = elo_state.calculate(
            fighter_history=pd.DataFrame(),
            fight_row=row,
            context={"state": states[b_id]},
        )

        rows.append(
            make_shadow_row(
                row=row,
                source_row_index=source_row_index,
                fighter_id=r_id,
                fighter_name=row.get("r_name"),
                corner="red",
                opponent_id=b_id,
                opponent_name=row.get("b_name"),
                features=r_features,
            )
        )
        rows.append(
            make_shadow_row(
                row=row,
                source_row_index=source_row_index,
                fighter_id=b_id,
                fighter_name=row.get("b_name"),
                corner="blue",
                opponent_id=r_id,
                opponent_name=row.get("r_name"),
                features=b_features,
            )
        )

        elo_state.update_after_fight(
            red_state=states[r_id],
            blue_state=states[b_id],
            red_won=bool(row["target"] == 1),
        )

    return pd.DataFrame(rows)


def make_shadow_row(
    *,
    row: pd.Series,
    source_row_index: int,
    fighter_id: str,
    fighter_name: Any,
    corner: str,
    opponent_id: str,
    opponent_name: Any,
    features: dict[str, float],
) -> dict[str, Any]:
    """Build one shadow fighter-state row for comparison."""

    output = {
        "fight_id": row["fight_id"],
        "date": row["date"],
        "fight_date": row["date"],
        "source_row_index": source_row_index,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "corner": corner,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
    }
    output.update(features)
    return output


if __name__ == "__main__":
    main()
