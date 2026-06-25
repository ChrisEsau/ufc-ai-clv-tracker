# ============================================================
# pipeline/market/normalizers/canonical_market_schema.py
# ============================================================

"""Canonical sportsbook-agnostic market catalog schema.

This schema is intentionally separate from Market V2 production outcome files.
It is the provider-neutral contract for discovered sportsbook markets before
fight matching, model probability joining, EV, staking, CLV, or dashboard use.
"""

from __future__ import annotations

import pandas as pd


CANONICAL_MARKET_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "event_name",
    "event_start_timestamp",
    "provider_subcategory_id",
    "provider_subcategory_name",
    "provider_market_id",
    "provider_market_name",
    "provider_market_type_id",
    "provider_market_type_name",
    "provider_selection_id",
    "provider_selection_name",
    "market_family",
    "market_key",
    "outcome_type",
    "outcome_key",
    "side",
    "fighter_name",
    "fighter_provider_id",
    "line",
    "american_odds",
    "decimal_odds",
    "true_odds",
    "implied_probability",
    "is_conditional_no_action",
    "condition_key",
    "round_number",
    "method_key",
    "is_parlay",
    "is_boost",
    "is_promo",
    "raw_payload_path",
    "request_url",
]


CANONICAL_MARKET_AUDIT_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "input_rows",
    "output_rows",
    "unmapped_rows",
    "mapped_rate",
    "market_family_counts",
    "unmapped_market_names",
    "passes_validation",
]


CANONICAL_STRING_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "event_name",
    "event_start_timestamp",
    "provider_subcategory_id",
    "provider_subcategory_name",
    "provider_market_id",
    "provider_market_name",
    "provider_market_type_id",
    "provider_market_type_name",
    "provider_selection_id",
    "provider_selection_name",
    "market_family",
    "market_key",
    "outcome_type",
    "outcome_key",
    "side",
    "fighter_name",
    "fighter_provider_id",
    "condition_key",
    "method_key",
    "raw_payload_path",
    "request_url",
]

CANONICAL_NUMERIC_COLUMNS = [
    "line",
    "american_odds",
    "decimal_odds",
    "true_odds",
    "implied_probability",
    "round_number",
]

CANONICAL_BOOL_COLUMNS = [
    "is_conditional_no_action",
    "is_parlay",
    "is_boost",
    "is_promo",
]


def enforce_canonical_market_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce stable dtypes for canonical market catalog parquet writes."""

    out = df.copy()

    for column in CANONICAL_STRING_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("string")

    for column in CANONICAL_NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in CANONICAL_BOOL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("boolean")

    return out


def ensure_canonical_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable canonical market catalog schema and dtypes."""

    out = df.copy()
    for column in CANONICAL_MARKET_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    out = out[CANONICAL_MARKET_COLUMNS]
    return enforce_canonical_market_dtypes(out)


def ensure_canonical_market_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable canonical market audit schema."""

    out = df.copy()
    for column in CANONICAL_MARKET_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[CANONICAL_MARKET_AUDIT_COLUMNS]
