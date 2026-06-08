# ============================================================
# pipeline/market/market_validator.py
# ============================================================

"""Validation helpers for Market Pipeline V2 artifacts."""

from __future__ import annotations

import pandas as pd


REQUIRED_MARKET_OUTCOME_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "fight_id",
    "market_key",
    "outcome_label",
    "outcome_fighter_id",
    "american_odds",
    "decimal_odds",
    "implied_probability",
    "odds_match_type",
]

DUPLICATE_KEY_COLUMNS = [
    "snapshot_run_id",
    "bookmaker",
    "fight_id",
    "market_key",
    "outcome_fighter_id",
]

AUDIT_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "artifact_name",
    "rows",
    "required_columns_present",
    "missing_required_columns",
    "duplicate_key_rows",
    "missing_fight_id_rows",
    "missing_outcome_fighter_id_rows",
    "missing_american_odds_rows",
    "bad_decimal_odds_rows",
    "bad_implied_probability_rows",
    "bookmakers_found",
    "market_keys_found",
    "passes_validation",
]


def _count_blank(series: pd.Series) -> int:
    """Count null/blank values in a series."""

    return int(series.isna().sum() + series.fillna("").astype(str).str.strip().eq("").sum())


def validate_market_outcomes(
    market_df: pd.DataFrame,
    *,
    snapshot_run_id: str,
    snapshot_timestamp: str,
    artifact_name: str = "market_outcomes",
) -> pd.DataFrame:
    """Return a one-row validation audit for Market V2 outcomes."""

    df = market_df.copy()
    missing_required = [
        column for column in REQUIRED_MARKET_OUTCOME_COLUMNS
        if column not in df.columns
    ]

    required_present = len(missing_required) == 0

    duplicate_key_rows = 0
    missing_fight_id_rows = 0
    missing_outcome_fighter_id_rows = 0
    missing_american_odds_rows = 0
    bad_decimal_odds_rows = 0
    bad_implied_probability_rows = 0
    bookmakers_found = []
    market_keys_found = []

    if required_present and not df.empty:
        duplicate_key_rows = int(
            df.duplicated(subset=DUPLICATE_KEY_COLUMNS, keep=False).sum()
        )
        missing_fight_id_rows = _count_blank(df["fight_id"])
        missing_outcome_fighter_id_rows = _count_blank(df["outcome_fighter_id"])
        missing_american_odds_rows = int(pd.to_numeric(df["american_odds"], errors="coerce").isna().sum())

        decimal_odds = pd.to_numeric(df["decimal_odds"], errors="coerce")
        bad_decimal_odds_rows = int(decimal_odds.isna().sum() + decimal_odds.le(1.0).sum())

        implied_probability = pd.to_numeric(df["implied_probability"], errors="coerce")
        bad_implied_probability_rows = int(
            implied_probability.isna().sum()
            + implied_probability.lt(0.0).sum()
            + implied_probability.gt(1.0).sum()
        )

        bookmakers_found = sorted(df["bookmaker"].dropna().astype(str).unique().tolist())
        market_keys_found = sorted(df["market_key"].dropna().astype(str).unique().tolist())

    passes_validation = bool(
        required_present
        and duplicate_key_rows == 0
        and missing_fight_id_rows == 0
        and missing_outcome_fighter_id_rows == 0
        and missing_american_odds_rows == 0
        and bad_decimal_odds_rows == 0
        and bad_implied_probability_rows == 0
    )

    audit_row = {
        "snapshot_run_id": snapshot_run_id,
        "snapshot_timestamp": snapshot_timestamp,
        "artifact_name": artifact_name,
        "rows": int(len(df)),
        "required_columns_present": required_present,
        "missing_required_columns": missing_required,
        "duplicate_key_rows": duplicate_key_rows,
        "missing_fight_id_rows": missing_fight_id_rows,
        "missing_outcome_fighter_id_rows": missing_outcome_fighter_id_rows,
        "missing_american_odds_rows": missing_american_odds_rows,
        "bad_decimal_odds_rows": bad_decimal_odds_rows,
        "bad_implied_probability_rows": bad_implied_probability_rows,
        "bookmakers_found": bookmakers_found,
        "market_keys_found": market_keys_found,
        "passes_validation": passes_validation,
    }

    return ensure_market_audit_columns(pd.DataFrame([audit_row]))


def ensure_market_audit_columns(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable validation audit schema."""

    out = audit_df.copy()
    for column in AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")

    return out[AUDIT_COLUMNS]
