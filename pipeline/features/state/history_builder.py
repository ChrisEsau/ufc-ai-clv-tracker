"""Build reusable fighter-state artifacts from completed UFC fights.

This module creates fighter-level point-in-time snapshots without changing the
existing fight-level rolling feature artifact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from pipeline.features.base.elo import K_FACTOR, expected_score
from pipeline.features.state.registry import resolve_state_modules


SNAPSHOT_CONTEXT_COLUMNS = [
    "event_id",
    "event_name",
    "fight_id",
    "date",
    "division",
    "title_fight",
    "total_rounds",
]


def _corner_stats(row: pd.Series, prefix: str) -> dict[str, float]:
    """Return the corner stat dictionary expected by the V5 state updater."""

    return {
        "kd": row[f"{prefix}_kd"],
        "sig_str_landed": row[f"{prefix}_sig_str_landed"],
        "sig_str_attempted": row[f"{prefix}_sig_str_atmpted"],
        "td_landed": row[f"{prefix}_td_landed"],
        "td_attempted": row[f"{prefix}_td_atmpted"],
        "sub_att": row[f"{prefix}_sub_att"],
        "ctrl": row[f"{prefix}_ctrl"],
    }


def _context_payload(row: pd.Series, source_row_index: int) -> dict[str, Any]:
    """Return stable fight context columns for a fighter-state snapshot."""

    payload = {}
    for column in SNAPSHOT_CONTEXT_COLUMNS:
        if column in row.index:
            payload[column] = row[column]
    payload["fight_date"] = row["date"]
    payload["source_row_index"] = source_row_index
    return payload


def _snapshot_row(
    row: pd.Series,
    source_row_index: int,
    fighter_id: str,
    fighter_name: Any,
    corner: str,
    prefight_features: dict[str, Any],
) -> dict[str, Any]:
    """Build one fighter-level prefight snapshot row."""

    snapshot = _context_payload(row, source_row_index=source_row_index)
    snapshot.update(
        {
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "corner": corner,
            "opponent_id": row["b_id"] if corner == "red" else row["r_id"],
            "opponent_name": row["b_name"] if corner == "red" else row["r_name"],
        }
    )
    snapshot.update(prefight_features)
    return snapshot


def build_fighter_state_history(df: pd.DataFrame) -> pd.DataFrame:
    """Build fighter-level prefight state history from chronological fight rows.

    The input dataframe must already be sorted chronologically and contain a
    canonical ``target`` column where 1 means the red fighter won.
    """

    modules = resolve_state_modules()
    if len(modules) != 1:
        raise ValueError(
            "fighter_state_history V1 expects exactly one legacy module until split modules are implemented"
        )

    module = modules[0]
    fighter_state: defaultdict[str, dict[str, Any]] = defaultdict(module.initial_state)
    snapshot_rows: list[dict[str, Any]] = []

    for source_row_index, row in df.reset_index(drop=True).iterrows():
        fight_date = row["date"]
        fight_time_sec = row["match_time_sec"]

        r_id = row["r_id"]
        b_id = row["b_id"]

        r_pre = module.prefight_features(fighter_state, r_id, fight_date)
        b_pre = module.prefight_features(fighter_state, b_id, fight_date)

        snapshot_rows.append(
            _snapshot_row(
                row=row,
                source_row_index=source_row_index,
                fighter_id=r_id,
                fighter_name=row.get("r_name"),
                corner="red",
                prefight_features=r_pre,
            )
        )
        snapshot_rows.append(
            _snapshot_row(
                row=row,
                source_row_index=source_row_index,
                fighter_id=b_id,
                fighter_name=row.get("b_name"),
                corner="blue",
                prefight_features=b_pre,
            )
        )

        r_elo = fighter_state[r_id]["elo"]
        b_elo = fighter_state[b_id]["elo"]

        r_expected = expected_score(r_elo, b_elo)
        b_expected = expected_score(b_elo, r_elo)

        r_actual = row["target"]
        b_actual = 1 - row["target"]

        fighter_state[r_id]["elo"] = r_elo + K_FACTOR * (r_actual - r_expected)
        fighter_state[b_id]["elo"] = b_elo + K_FACTOR * (b_actual - b_expected)

        r_stats = _corner_stats(row, "r")
        b_stats = _corner_stats(row, "b")

        module.update_after_fight(
            fighter_state=fighter_state,
            fighter_id=r_id,
            fight_date=fight_date,
            won=(row["target"] == 1),
            method=row["method"],
            own=r_stats,
            opp=b_stats,
            fight_time_sec=fight_time_sec,
            opponent_elo=b_elo,
        )
        module.update_after_fight(
            fighter_state=fighter_state,
            fighter_id=b_id,
            fight_date=fight_date,
            won=(row["target"] == 0),
            method=row["method"],
            own=b_stats,
            opp=r_stats,
            fight_time_sec=fight_time_sec,
            opponent_elo=r_elo,
        )

    return pd.DataFrame(snapshot_rows)


def build_latest_fighter_state(history_df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent fighter-state snapshot for each fighter."""

    if history_df.empty:
        return history_df.copy()

    sort_columns = [column for column in ["fighter_id", "fight_date", "source_row_index", "fight_id"] if column in history_df.columns]
    latest = history_df.sort_values(sort_columns).groupby("fighter_id", as_index=False).tail(1)
    return latest.reset_index(drop=True)
