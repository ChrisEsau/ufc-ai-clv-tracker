"""Plugin-driven fighter-state history builder.

This module mirrors the current legacy V5 fighter-state output using the raw
fighter feature plugins under ``pipeline.features.raw_fighter_features``.
It is introduced in shadow mode first; production runners should switch to this
builder only after parity validation passes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from pipeline.features.raw_fighter_features import (
    elo_state,
    ewm_state,
    finish_profile,
    grappling_rates,
    recent_form,
    record_state,
    striking_rates,
)
from pipeline.features.state.history_builder import SNAPSHOT_CONTEXT_COLUMNS

BASE_PLUGINS = [
    record_state,
    elo_state,
    striking_rates,
    grappling_rates,
    finish_profile,
    recent_form,
]


def build_plugin_fighter_state_history(df: pd.DataFrame, *, add_ewm: bool = True) -> pd.DataFrame:
    """Build fighter-state history using raw fighter feature plugins."""

    states = {
        plugin_name(plugin): defaultdict(plugin.initial_state)
        for plugin in BASE_PLUGINS
    }
    rows: list[dict[str, Any]] = []

    for source_row_index, row in df.reset_index(drop=True).iterrows():
        r_id = str(row["r_id"])
        b_id = str(row["b_id"])
        fight_time_sec = row["match_time_sec"]
        fight_date = row["date"]
        red_won = bool(row["target"] == 1)
        r_stats = corner_stats(row, "r")
        b_stats = corner_stats(row, "b")

        rows.append(
            snapshot_row(
                row=row,
                source_row_index=source_row_index,
                fighter_id=r_id,
                fighter_name=row.get("r_name"),
                corner="red",
                opponent_id=b_id,
                opponent_name=row.get("b_name"),
                features=calculate_all(row=row, fighter_id=r_id, states=states),
            )
        )
        rows.append(
            snapshot_row(
                row=row,
                source_row_index=source_row_index,
                fighter_id=b_id,
                fighter_name=row.get("b_name"),
                corner="blue",
                opponent_id=r_id,
                opponent_name=row.get("r_name"),
                features=calculate_all(row=row, fighter_id=b_id, states=states),
            )
        )

        update_after_fight(
            states=states,
            r_id=r_id,
            b_id=b_id,
            row=row,
            fight_date=fight_date,
            fight_time_sec=fight_time_sec,
            red_won=red_won,
            r_stats=r_stats,
            b_stats=b_stats,
        )

    history_df = pd.DataFrame(rows)
    if add_ewm:
        history_df = ewm_state.enrich_history(history_df)
    return history_df


def calculate_all(
    *,
    row: pd.Series,
    fighter_id: str,
    states: dict[str, defaultdict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Calculate all base plugin features for one fighter snapshot."""

    features: dict[str, Any] = {}
    for plugin in BASE_PLUGINS:
        name = plugin_name(plugin)
        features.update(
            plugin.calculate(
                fighter_history=pd.DataFrame(),
                fight_row=row,
                context={"state": states[name][fighter_id]},
            )
        )
    return features


def update_after_fight(
    *,
    states: dict[str, defaultdict[str, dict[str, Any]]],
    r_id: str,
    b_id: str,
    row: pd.Series,
    fight_date: Any,
    fight_time_sec: float,
    red_won: bool,
    r_stats: dict[str, float],
    b_stats: dict[str, float],
) -> None:
    """Update all plugin states after a completed fight."""

    record_state.update_after_fight(
        state=states["record_state"][r_id],
        fight_date=fight_date,
        won=red_won,
    )
    record_state.update_after_fight(
        state=states["record_state"][b_id],
        fight_date=fight_date,
        won=not red_won,
    )
    elo_state.update_after_fight(
        red_state=states["elo_state"][r_id],
        blue_state=states["elo_state"][b_id],
        red_won=red_won,
    )
    update_pair(striking_rates, states, r_id, b_id, r_stats, b_stats, fight_time_sec)
    update_pair(grappling_rates, states, r_id, b_id, r_stats, b_stats, fight_time_sec)
    finish_profile.update_after_fight(
        state=states["finish_profile"][r_id],
        method=row["method"],
        won=red_won,
        fight_time_sec=fight_time_sec,
    )
    finish_profile.update_after_fight(
        state=states["finish_profile"][b_id],
        method=row["method"],
        won=not red_won,
        fight_time_sec=fight_time_sec,
    )
    recent_form.update_after_fight(
        state=states["recent_form"][r_id],
        method=row["method"],
        won=red_won,
        own=r_stats,
        opp=b_stats,
        fight_time_sec=fight_time_sec,
    )
    recent_form.update_after_fight(
        state=states["recent_form"][b_id],
        method=row["method"],
        won=not red_won,
        own=b_stats,
        opp=r_stats,
        fight_time_sec=fight_time_sec,
    )


def update_pair(plugin, states, r_id, b_id, r_stats, b_stats, fight_time_sec) -> None:
    """Update paired red/blue plugin states for stat-rate plugins."""

    name = plugin_name(plugin)
    plugin.update_after_fight(
        state=states[name][r_id],
        own=r_stats,
        opp=b_stats,
        fight_time_sec=fight_time_sec,
    )
    plugin.update_after_fight(
        state=states[name][b_id],
        own=b_stats,
        opp=r_stats,
        fight_time_sec=fight_time_sec,
    )


def snapshot_row(
    *,
    row: pd.Series,
    source_row_index: int,
    fighter_id: str,
    fighter_name: Any,
    corner: str,
    opponent_id: str,
    opponent_name: Any,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Build one fighter-level prefight snapshot row."""

    output = {}
    for column in SNAPSHOT_CONTEXT_COLUMNS:
        if column in row.index:
            output[column] = row[column]
    output.update(
        {
            "fight_date": row["date"],
            "source_row_index": source_row_index,
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "corner": corner,
            "opponent_id": opponent_id,
            "opponent_name": opponent_name,
        }
    )
    output.update(features)
    return output


def corner_stats(row: pd.Series, prefix: str) -> dict[str, float]:
    """Return corner stats matching the legacy V5 updater contract."""

    return {
        "kd": row[f"{prefix}_kd"],
        "sig_str_landed": row[f"{prefix}_sig_str_landed"],
        "sig_str_attempted": row[f"{prefix}_sig_str_atmpted"],
        "td_landed": row[f"{prefix}_td_landed"],
        "td_attempted": row[f"{prefix}_td_atmpted"],
        "sub_att": row[f"{prefix}_sub_att"],
        "ctrl": row[f"{prefix}_ctrl"],
    }


def plugin_name(plugin) -> str:
    """Return short plugin module name."""

    return plugin.__name__.split(".")[-1]
