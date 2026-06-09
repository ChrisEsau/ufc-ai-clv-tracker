"""EWM enrichment for fighter-state history artifacts.

This module enriches fighter-level state snapshots directly. It does not change
the existing fight-level rolling EWM builder.
"""

from __future__ import annotations

import pandas as pd

from pipeline.features.base.ewm_features import EWM_SPAN


STATE_CONTEXT_COLUMNS = {
    "event_id",
    "event_name",
    "fight_id",
    "date",
    "division",
    "title_fight",
    "total_rounds",
    "fight_date",
    "fighter_id",
    "fighter_name",
    "corner",
    "opponent_id",
    "opponent_name",
}


def get_ewm_state_columns(history_df: pd.DataFrame) -> list[str]:
    """Return numeric fighter-state columns eligible for EWM enrichment."""

    return [
        column
        for column in history_df.columns
        if column not in STATE_CONTEXT_COLUMNS
        and pd.api.types.is_numeric_dtype(history_df[column])
        and not column.startswith("ewm_")
        and not column.startswith("form_delta_")
    ]


def add_ewm_state_features(
    history_df: pd.DataFrame,
    span: int = EWM_SPAN,
) -> pd.DataFrame:
    """Add fighter-level EWM and form-delta columns to state history.

    For every eligible numeric fighter-state column ``x``, this adds:

    - ``ewm_x``: fighter-level exponentially weighted mean of ``x``
    - ``form_delta_x``: ``ewm_x - x``
    """

    if history_df.empty:
        return history_df.copy()

    out = history_df.copy()
    sort_columns = [column for column in ["fighter_id", "fight_date", "fight_id"] if column in out.columns]
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
