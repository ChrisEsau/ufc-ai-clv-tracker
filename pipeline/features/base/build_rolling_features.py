"""Orchestrates production rolling UFC base feature generation.

This module migrates the core rolling dataset loop from
``UFC_rolling_dataset_V4_refactored.ipynb``.

Current responsibility:
- Build chronological fighter-state rows
- Add r_pre_*, b_pre_*, and *_diff columns
- Update Elo and fighter state after each fight

EWM/recent-form merge-back and final artifact writing are migrated separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from pipeline.features.base.elo import K_FACTOR, expected_score
from pipeline.features.base.fighter_state import default_state, update_fighter
from pipeline.features.base.prefight_features import get_prefight_features


def _corner_stats(row: pd.Series, prefix: str) -> dict[str, float]:
    """Return the corner stat dictionary expected by update_fighter()."""
    return {
        "kd": row[f"{prefix}_kd"],
        "sig_str_landed": row[f"{prefix}_sig_str_landed"],
        "sig_str_attempted": row[f"{prefix}_sig_str_atmpted"],
        "td_landed": row[f"{prefix}_td_landed"],
        "td_attempted": row[f"{prefix}_td_atmpted"],
        "sub_att": row[f"{prefix}_sub_att"],
        "ctrl": row[f"{prefix}_ctrl"],
    }


def build_rolling_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling point-in-time base features from completed fight rows.

    The input dataframe is expected to already contain a chronological fight
    dataset with a ``target`` column where 1 means the red fighter won and 0
    means the blue fighter won.

    This function intentionally mirrors the notebook loop before the EWM step.
    It should produce the same intermediate 237-column rolling dataframe when
    given the same input data.
    """
    fighter_state: defaultdict[str, dict[str, Any]] = defaultdict(default_state)
    rolling_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        fight_date = row["date"]
        fight_time_sec = row["match_time_sec"]

        r_id = row["r_id"]
        b_id = row["b_id"]

        r_pre = get_prefight_features(fighter_state, r_id, fight_date)
        b_pre = get_prefight_features(fighter_state, b_id, fight_date)

        new_row = row.to_dict()

        for key in r_pre:
            new_row[f"r_pre_{key}"] = r_pre[key]
            new_row[f"b_pre_{key}"] = b_pre[key]
            new_row[f"{key}_diff"] = r_pre[key] - b_pre[key]

        rolling_rows.append(new_row)

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

        update_fighter(
            fighter_state,
            r_id,
            fight_date,
            won=(row["target"] == 1),
            method=row["method"],
            own=r_stats,
            opp=b_stats,
            fight_time_sec=fight_time_sec,
            opponent_elo=b_elo,
        )

        update_fighter(
            fighter_state,
            b_id,
            fight_date,
            won=(row["target"] == 0),
            method=row["method"],
            own=b_stats,
            opp=r_stats,
            fight_time_sec=fight_time_sec,
            opponent_elo=r_elo,
        )

    return pd.DataFrame(rolling_rows)
