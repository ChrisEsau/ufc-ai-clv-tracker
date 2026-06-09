"""Validate raw fighter feature plugins against production fighter-state history.

Run from repo root:

    python -m archive.migration_validation.run_validate_raw_fighter_plugins
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from pipeline.common.paths import FIGHTER_STATE_HISTORY_PATH, MASTER_PATH
from pipeline.features.raw_fighter_features import (
    elo_state,
    ewm_state,
    finish_profile,
    grappling_rates,
    recent_form,
    record_state,
    striking_rates,
)
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling

KEY_COLUMNS = ["fight_id", "fighter_id", "corner"]
BASE_PLUGINS = [
    record_state,
    elo_state,
    striking_rates,
    grappling_rates,
    finish_profile,
    recent_form,
]
COMPARE_COLUMNS = [
    *record_state.OUTPUT_COLUMNS,
    *elo_state.OUTPUT_COLUMNS,
    *striking_rates.OUTPUT_COLUMNS,
    *grappling_rates.OUTPUT_COLUMNS,
    *finish_profile.OUTPUT_COLUMNS,
    *recent_form.OUTPUT_COLUMNS,
    *ewm_state.OUTPUT_COLUMNS,
]


def main() -> None:
    print("=" * 80)
    print("RAW FIGHTER PLUGIN FULL SHADOW VALIDATION")
    print("=" * 80)
    print(f"Master path       : {MASTER_PATH}")
    print(f"State history path: {FIGHTER_STATE_HISTORY_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    prepared_df = prepare_master_for_rolling(master_df)
    production_df = pd.read_parquet(FIGHTER_STATE_HISTORY_PATH)

    shadow_df = build_shadow_history(prepared_df)
    shadow_df = ewm_state.enrich_history(shadow_df)

    compare_columns = [
        c for c in dedupe(COMPARE_COLUMNS)
        if c in production_df.columns and c in shadow_df.columns
    ]
    missing = sorted(set(dedupe(COMPARE_COLUMNS)) - set(compare_columns))

    print(f"Prepared fights   : {len(prepared_df):,}")
    print(f"Production rows   : {len(production_df):,}")
    print(f"Shadow rows       : {len(shadow_df):,}")
    print(f"Compared columns  : {len(compare_columns):,}")
    if missing:
        print(f"Missing compare columns: {missing}")

    merged = production_df[KEY_COLUMNS + compare_columns].merge(
        shadow_df[KEY_COLUMNS + compare_columns],
        on=KEY_COLUMNS,
        suffixes=("_production", "_shadow"),
        how="inner",
    )
    print(f"Matched rows      : {len(merged):,}")

    if len(merged) != len(production_df):
        raise SystemExit(
            f"Matched rows mismatch: matched={len(merged):,}, production={len(production_df):,}"
        )

    audit = audit_columns(merged, compare_columns)
    print("\nStatus counts:")
    print(audit["status"].value_counts().to_string())
    print("\nParity checks:")
    print(audit.to_string(index=False))

    if (audit["status"] != "PASS").any():
        raise SystemExit("Raw fighter plugin validation failed.")

    print("DONE")


def build_shadow_history(prepared_df: pd.DataFrame) -> pd.DataFrame:
    states = {
        plugin_name(plugin): defaultdict(plugin.initial_state)
        for plugin in BASE_PLUGINS
    }
    rows: list[dict[str, Any]] = []

    for source_row_index, row in prepared_df.reset_index(drop=True).iterrows():
        r_id = str(row["r_id"])
        b_id = str(row["b_id"])
        fight_time_sec = row["match_time_sec"]
        fight_date = row["date"]
        red_won = bool(row["target"] == 1)
        r_stats = corner_stats(row, "r")
        b_stats = corner_stats(row, "b")

        rows.append(make_row(row, source_row_index, r_id, row.get("r_name"), "red", b_id, row.get("b_name"), calc_all(row, r_id, states)))
        rows.append(make_row(row, source_row_index, b_id, row.get("b_name"), "blue", r_id, row.get("r_name"), calc_all(row, b_id, states)))

        record_state.update_after_fight(state=states["record_state"][r_id], fight_date=fight_date, won=red_won)
        record_state.update_after_fight(state=states["record_state"][b_id], fight_date=fight_date, won=not red_won)
        elo_state.update_after_fight(red_state=states["elo_state"][r_id], blue_state=states["elo_state"][b_id], red_won=red_won)
        update_pair(striking_rates, states, r_id, b_id, r_stats, b_stats, fight_time_sec)
        update_pair(grappling_rates, states, r_id, b_id, r_stats, b_stats, fight_time_sec)
        finish_profile.update_after_fight(state=states["finish_profile"][r_id], method=row["method"], won=red_won, fight_time_sec=fight_time_sec)
        finish_profile.update_after_fight(state=states["finish_profile"][b_id], method=row["method"], won=not red_won, fight_time_sec=fight_time_sec)
        recent_form.update_after_fight(state=states["recent_form"][r_id], method=row["method"], won=red_won, own=r_stats, opp=b_stats, fight_time_sec=fight_time_sec)
        recent_form.update_after_fight(state=states["recent_form"][b_id], method=row["method"], won=not red_won, own=b_stats, opp=r_stats, fight_time_sec=fight_time_sec)

    return pd.DataFrame(rows)


def calc_all(row: pd.Series, fighter_id: str, states: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for plugin in BASE_PLUGINS:
        name = plugin_name(plugin)
        features.update(plugin.calculate(pd.DataFrame(), row, {"state": states[name][fighter_id]}))
    return features


def update_pair(plugin, states, r_id, b_id, r_stats, b_stats, fight_time_sec) -> None:
    name = plugin_name(plugin)
    plugin.update_after_fight(state=states[name][r_id], own=r_stats, opp=b_stats, fight_time_sec=fight_time_sec)
    plugin.update_after_fight(state=states[name][b_id], own=b_stats, opp=r_stats, fight_time_sec=fight_time_sec)


def make_row(row, source_row_index, fighter_id, fighter_name, corner, opponent_id, opponent_name, features):
    out = {
        "event_id": row.get("event_id"),
        "event_name": row.get("event_name"),
        "fight_id": row["fight_id"],
        "date": row["date"],
        "fight_date": row["date"],
        "division": row.get("division"),
        "title_fight": row.get("title_fight"),
        "total_rounds": row.get("total_rounds"),
        "source_row_index": source_row_index,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "corner": corner,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
    }
    out.update(features)
    return out


def corner_stats(row: pd.Series, prefix: str) -> dict[str, float]:
    return {
        "kd": row[f"{prefix}_kd"],
        "sig_str_landed": row[f"{prefix}_sig_str_landed"],
        "sig_str_attempted": row[f"{prefix}_sig_str_atmpted"],
        "td_landed": row[f"{prefix}_td_landed"],
        "td_attempted": row[f"{prefix}_td_atmpted"],
        "sub_att": row[f"{prefix}_sub_att"],
        "ctrl": row[f"{prefix}_ctrl"],
    }


def audit_columns(merged: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        prod = pd.to_numeric(merged[f"{column}_production"], errors="coerce")
        shadow = pd.to_numeric(merged[f"{column}_shadow"], errors="coerce")
        delta = (prod - shadow).abs().dropna()
        nonzero_rows = int((delta > 1e-9).sum())
        rows.append({
            "feature_name": column,
            "status": "PASS" if nonzero_rows == 0 else "FAIL",
            "max_abs_diff": float(delta.max()) if len(delta) else 0.0,
            "mean_abs_diff": float(delta.mean()) if len(delta) else 0.0,
            "nonzero_rows": nonzero_rows,
        })
    return pd.DataFrame(rows)


def plugin_name(plugin) -> str:
    return plugin.__name__.split(".")[-1]


def dedupe(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


if __name__ == "__main__":
    main()
