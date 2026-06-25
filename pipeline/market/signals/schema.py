# ============================================================
# pipeline/market/signals/schema.py
# ============================================================

"""Canonical Market Signal schema.

Market signals are sportsbook/market intelligence records generated from
market outcomes, model-market snapshots, line movement, and provider catalogs.

The dashboard should render this file instead of recomputing signal logic.
"""

from __future__ import annotations

import pandas as pd


MARKET_SIGNAL_COLUMNS = [
    "signal_id",
    "signal_run_id",
    "signal_timestamp",
    "signal_type",
    "signal_family",
    "severity",
    "confidence_score",
    "is_actionable",
    "action_label",
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
    "bookmaker",
    "bookmakers_involved",
    "best_bookmaker",
    "best_american_odds",
    "best_implied_probability",
    "worst_bookmaker",
    "worst_american_odds",
    "worst_implied_probability",
    "consensus_american_odds",
    "consensus_implied_probability",
    "book_american_odds",
    "book_implied_probability",
    "model_probability",
    "model_edge",
    "edge_pct",
    "ev_dollars_at_100",
    "spread_cents",
    "spread_probability",
    "line_move_cents",
    "line_move_probability",
    "age_minutes",
    "snapshot_count",
    "provider_count",
    "explanation",
    "suggested_action",
    "source_path",
]


MARKET_SIGNAL_AUDIT_COLUMNS = [
    "signal_run_id",
    "signal_timestamp",
    "source_market_rows",
    "output_signal_rows",
    "signal_type_counts",
    "passes_validation",
    "notes",
]


STRING_COLUMNS = [
    "signal_id",
    "signal_run_id",
    "signal_timestamp",
    "signal_type",
    "signal_family",
    "severity",
    "action_label",
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
    "bookmaker",
    "bookmakers_involved",
    "best_bookmaker",
    "worst_bookmaker",
    "explanation",
    "suggested_action",
    "source_path",
]

NUMERIC_COLUMNS = [
    "confidence_score",
    "best_american_odds",
    "best_implied_probability",
    "worst_american_odds",
    "worst_implied_probability",
    "consensus_american_odds",
    "consensus_implied_probability",
    "book_american_odds",
    "book_implied_probability",
    "model_probability",
    "model_edge",
    "edge_pct",
    "ev_dollars_at_100",
    "spread_cents",
    "spread_probability",
    "line_move_cents",
    "line_move_probability",
    "age_minutes",
    "snapshot_count",
    "provider_count",
]

BOOL_COLUMNS = [
    "is_actionable",
]


def ensure_market_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable market signal schema and parquet-safe dtypes."""

    out = df.copy()
    for column in MARKET_SIGNAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    out = out[MARKET_SIGNAL_COLUMNS]

    for column in STRING_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("string")

    for column in NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in BOOL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("boolean")

    return out


def ensure_market_signal_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable market signal audit schema."""

    out = df.copy()
    for column in MARKET_SIGNAL_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[MARKET_SIGNAL_AUDIT_COLUMNS]
