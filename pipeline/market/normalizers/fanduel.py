# ============================================================
# pipeline/market/normalizers/fanduel.py
# ============================================================

"""FanDuel diagnostic-to-canonical market normalizer.

V1 intentionally supports only moneyline and goes-distance markets.
Prop/exotic markets remain in diagnostics until the prop engine is ready.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.market.normalizers.canonical_market_schema import ensure_canonical_market_columns
from pipeline.market.normalizers.comparison_key import build_comparison_key


def _safe_lower(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _selection_side(selection_name: Any, outcome_type: Any) -> str | None:
    text = _safe_lower(selection_name)
    outcome = _safe_lower(outcome_type)

    if text == "yes" or outcome == "yes":
        return "yes"
    if text == "no" or outcome == "no":
        return "no"
    if outcome in {"home", "away"}:
        return "fighter"

    return text or outcome or None


def _market_mapping(row: pd.Series) -> dict[str, Any]:
    supported_family = row.get("supported_market_family")
    raw_market_name = row.get("raw_market_name")
    raw_market_text = _safe_lower(raw_market_name)

    market_family = supported_family
    market_key = None
    outcome_type = None
    outcome_key = None
    side = _selection_side(row.get("raw_selection_name"), row.get("selection_outcome_type"))

    if market_family == "moneyline" or "moneyline" in raw_market_text:
        market_family = "main_lines"
        market_key = "moneyline"
        outcome_type = "fighter"
        outcome_key = "fighter_win"
        side = "fighter"

    elif market_family == "goes_distance":
        market_key = "goes_distance"
        outcome_type = "fight"
        outcome_key = side

    return {
        "market_family": market_family,
        "market_key": market_key,
        "outcome_type": outcome_type,
        "outcome_key": outcome_key,
        "side": side,
        "line": row.get("line"),
        "is_conditional_no_action": False,
        "condition_key": None,
        "round_number": None,
        "method_key": None,
    }


def normalize_fanduel_diagnostic_rows(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    """Convert FanDuel diagnostic rows into canonical market catalog rows."""

    # V1 scope: only normalize moneyline + goes-distance.
    scoped_df = diagnostic_df[
        diagnostic_df["supported_market_family"].isin(["moneyline", "goes_distance"])
    ].copy()

    rows: list[dict[str, Any]] = []
    for _, row in scoped_df.iterrows():
        mapping = _market_mapping(row)

        fighter_name = None
        if mapping.get("outcome_type") == "fighter":
            fighter_name = row.get("selection_participant_name") or row.get("raw_selection_name")

        rows.append(
            {
                "snapshot_run_id": row.get("snapshot_run_id"),
                "snapshot_timestamp": row.get("snapshot_timestamp"),
                "source": row.get("source"),
                "bookmaker": row.get("bookmaker"),
                "provider_event_id": row.get("provider_event_id"),
                "event_name": row.get("event_name"),
                "event_start_timestamp": row.get("event_start_timestamp"),
                "provider_subcategory_id": None,
                "provider_subcategory_name": row.get("provider_competition_name"),
                "provider_market_id": row.get("provider_market_id"),
                "provider_market_name": row.get("raw_market_name"),
                "provider_market_type_id": None,
                "provider_market_type_name": row.get("provider_market_type_name"),
                "provider_selection_id": row.get("provider_selection_id"),
                "provider_selection_name": row.get("raw_selection_name"),
                "normalized_selection_name": fighter_name or row.get("raw_selection_name"),
                "fighter_name": fighter_name,
                "fighter_provider_id": None,
                "american_odds": row.get("price_american"),
                "decimal_odds": row.get("price_decimal"),
                "true_odds": row.get("true_odds"),
                "implied_probability": row.get("implied_probability"),
                "is_parlay": row.get("is_parlay"),
                "is_boost": row.get("is_boost"),
                "is_promo": row.get("is_promo"),
                "raw_payload_path": row.get("raw_payload_path"),
                "request_url": row.get("request_url"),
                **mapping,
            }
        )

    canonical_df = pd.DataFrame(rows)
    if not canonical_df.empty:
        canonical_df["comparison_key"] = canonical_df.apply(build_comparison_key, axis=1)
    return ensure_canonical_market_columns(canonical_df)
