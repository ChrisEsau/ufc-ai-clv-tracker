from __future__ import annotations

import pandas as pd


MARKET_INTELLIGENCE_HISTORY_COLUMNS = [
    "refresh_id",
    "refresh_timestamp",
    "source_run_id",
    "bookmaker",
    "fight_id",
    "event_name",
    "fight_display",
    "market_key",
    "market_display",
    "outcome_key",
    "comparison_key",
    "outcome_display",
    "side",
    "fighter_name",
    "american_odds",
    "implied_probability",
    "decimal_odds",
    "line",
    "provider_event_id",
    "provider_market_id",
    "provider_selection_id",
    "provider_market_type_name",
    "snapshot_source_path",
]


MARKET_INTELLIGENCE_HISTORY_AUDIT_COLUMNS = [
    "refresh_id",
    "refresh_timestamp",
    "source_market_rows",
    "history_rows_appended",
    "total_history_rows",
    "passes_validation",
    "notes",
]


STRING_COLUMNS = [
    "refresh_id",
    "refresh_timestamp",
    "source_run_id",
    "bookmaker",
    "fight_id",
    "event_name",
    "fight_display",
    "market_key",
    "market_display",
    "outcome_key",
    "comparison_key",
    "outcome_display",
    "side",
    "fighter_name",
    "provider_event_id",
    "provider_market_id",
    "provider_selection_id",
    "provider_market_type_name",
    "snapshot_source_path",
]

NUMERIC_COLUMNS = [
    "american_odds",
    "implied_probability",
    "decimal_odds",
    "line",
]


def ensure_market_intelligence_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in MARKET_INTELLIGENCE_HISTORY_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    out = out[MARKET_INTELLIGENCE_HISTORY_COLUMNS]

    for column in STRING_COLUMNS:
        out[column] = out[column].astype("string")

    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def ensure_market_intelligence_history_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in MARKET_INTELLIGENCE_HISTORY_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[MARKET_INTELLIGENCE_HISTORY_AUDIT_COLUMNS]
