"""EWM enrichment for fighter-state history artifacts.

This module enriches fighter-level state snapshots directly. It does not change
the existing fight-level rolling EWM builder.
"""

from __future__ import annotations

import pandas as pd

from pipeline.features.base.ewm_features import EWM_SPAN

EWM_STATE_COLUMNS = [
    "elo","fights","wins","losses","win_pct","kd_avg","kd_absorbed_avg",
    "splm","sapm","str_acc","str_def","td_avg","td_acc","td_def",
    "sub_avg","ctrl_per_min","ctrl_against_per_min","finish_rate","ko_rate",
    "sub_win_rate","decision_win_rate","finish_loss_rate","decision_loss_rate",
    "avg_opponent_elo","best_win_elo","worst_loss_elo","avg_fight_time",
    "win_streak","loss_streak","days_since_last_fight","recent_win_pct",
    "recent_splm","recent_sapm","recent_td_avg","recent_finish_rate",
    "recent_avg_fight_time",
]


def get_ewm_state_columns(history_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in EWM_STATE_COLUMNS
        if column in history_df.columns and pd.api.types.is_numeric_dtype(history_df[column])
    ]


def add_ewm_state_features(history_df: pd.DataFrame, span: int = EWM_SPAN) -> pd.DataFrame:
    if history_df.empty:
        return history_df.copy()

    out = history_df.copy()

    sort_columns = [
        column
        for column in ["fighter_id", "fight_date", "source_row_index"]
        if column in out.columns
    ]
    out = out.sort_values(sort_columns).reset_index(drop=True)

    state_columns = get_ewm_state_columns(out)
    new_columns = {}

    for column in state_columns:
        ewm_column = f"ewm_{column}"
        form_delta_column = f"form_delta_{column}"

        ewm_values = out.groupby("fighter_id")[column].transform(
            lambda series: series.ewm(span=span, adjust=False).mean()
        )

        new_columns[ewm_column] = ewm_values
        new_columns[form_delta_column] = ewm_values - out[column]

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return out
