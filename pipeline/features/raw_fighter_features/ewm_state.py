"""EWM raw fighter feature plugin adapter.

EWM state is a dataframe-level enrichment that depends on the full fighter-state
history after the base point-in-time features have been generated. This adapter
wraps the existing production EWM implementation so the raw fighter feature
plugin layer can reference EWM without duplicating formulas.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.features.state.ewm_state import (
    EWM_STATE_COLUMNS,
    add_ewm_state_features,
    get_ewm_state_columns,
)


OUTPUT_COLUMNS = [f"ewm_{column}" for column in EWM_STATE_COLUMNS]
FORM_DELTA_COLUMNS = [f"form_delta_{column}" for column in EWM_STATE_COLUMNS]


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return EWM values for one already-enriched fighter-history row.

    EWM is not calculated from a single row alone. The plugin contract is kept
    here for registry compatibility, while production use should call
    ``enrich_history`` after base fighter-state rows are built.
    """

    del fighter_history, fight_row
    context = context or {}
    row = context.get("row")
    if row is None:
        return {}

    return {
        column: row[column]
        for column in OUTPUT_COLUMNS
        if column in row.index
    }


def enrich_history(history_df: pd.DataFrame, span: int | None = None) -> pd.DataFrame:
    """Add EWM and form-delta columns to fighter-state history.

    This delegates to the existing validated implementation used by the current
    fighter-state runner.
    """

    if span is None:
        return add_ewm_state_features(history_df)
    return add_ewm_state_features(history_df, span=span)


def available_source_columns(history_df: pd.DataFrame) -> list[str]:
    """Return numeric base columns eligible for EWM enrichment."""

    return get_ewm_state_columns(history_df)


def initial_state() -> dict[str, Any]:
    """EWM has no per-fighter pre-enrichment mutable state."""

    return {}
