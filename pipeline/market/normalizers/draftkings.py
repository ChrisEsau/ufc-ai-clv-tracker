# ============================================================
# pipeline/market/normalizers/draftkings.py
# ============================================================

"""DraftKings diagnostic-to-canonical market normalizer.

This module converts DraftKings discovery diagnostics into a sportsbook-neutral
market catalog. It does not perform fighter matching, probability joins, EV,
staking, CLV, or dashboard logic.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from pipeline.market.normalizers.canonical_market_schema import ensure_canonical_market_columns


def _safe_lower(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _parse_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def _parse_round(value: Any) -> int | None:
    text = _safe_lower(value)
    match = re.search(r"round\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _method_key(*values: Any) -> str | None:
    text = " ".join(_safe_lower(v) for v in values)
    if "ko/tko" in text or "tko" in text or "knockout" in text:
        return "ko_tko_dq"
    if "submission" in text:
        return "submission"
    if "decision" in text:
        return "decision"
    if "draw" in text:
        return "draw"
    if "distance" in text:
        return "distance"
    return None


def _selection_side(selection_name: Any, outcome_type: Any) -> str | None:
    text = _safe_lower(selection_name)
    outcome = _safe_lower(outcome_type)
    if text.startswith("over") or outcome == "over":
        return "over"
    if text.startswith("under") or outcome == "under":
        return "under"
    if text == "yes" or outcome == "yes":
        return "yes"
    if text == "no" or outcome == "no":
        return "no"
    if outcome in {"home", "away"}:
        return "fighter"
    return text or outcome or None


def _market_mapping(row: pd.Series) -> dict[str, Any]:
    registry_family = row.get("registry_family")
    supported_family = row.get("supported_market_family")
    raw_market_name = row.get("raw_market_name")
    raw_selection_name = row.get("raw_selection_name")
    raw_market_text = _safe_lower(raw_market_name)
    raw_selection_text = _safe_lower(raw_selection_name)

    market_family = registry_family or supported_family
    market_key = market_family
    outcome_type = row.get("registry_outcome_type")
    outcome_key = None
    condition_key = None
    is_conditional_no_action = False
    round_number = _parse_round(raw_market_name) or _parse_round(raw_selection_name)
    method_key = _method_key(raw_market_name, raw_selection_name)
    line = _parse_number(row.get("line"))

    if market_family == "main_lines":
        if "moneyline" in raw_market_text:
            market_key = "moneyline"
            outcome_type = "fighter"
            outcome_key = "fighter_win"
        elif "point spread" in raw_market_text:
            market_key = "point_spread"
            outcome_type = "fighter_line"
            outcome_key = _selection_side(raw_selection_name, row.get("selection_outcome_type"))
        elif "total rounds" in raw_market_text:
            market_key = "total_rounds"
            outcome_type = "fight_line"
            outcome_key = _selection_side(raw_selection_name, row.get("selection_outcome_type"))

    elif market_family == "fighter_method_props":
        market_key = f"win_by_{method_key}" if method_key else "fighter_method"
        outcome_type = "fighter"
        outcome_key = market_key

    elif market_family == "exact_method":
        market_key = "exact_method"
        outcome_type = "fight"
        outcome_key = method_key

    elif market_family == "goes_distance":
        market_key = "goes_distance"
        outcome_type = "fight"
        outcome_key = _selection_side(raw_selection_name, row.get("selection_outcome_type"))

    elif market_family in {"alternate_total_rounds", "over_under_rounds"}:
        market_key = "total_rounds"
        outcome_type = "fight_line"
        outcome_key = _selection_side(raw_selection_name, row.get("selection_outcome_type"))
        if line is None:
            line = _parse_number(raw_selection_name)

    elif market_family == "alternate_point_spread":
        market_key = "point_spread"
        outcome_type = "fighter_line"
        outcome_key = _selection_side(raw_selection_name, row.get("selection_outcome_type"))

    elif market_family == "fighter_sig_strikes_total":
        market_key = "fighter_sig_strikes_total"
        outcome_type = "fighter_line"
        outcome_key = _selection_side(raw_selection_name, row.get("selection_outcome_type"))
        if line is None:
            line = _parse_number(raw_selection_name)

    elif market_family == "round_method":
        market_key = "round_method"
        outcome_type = "mixed"
        outcome_key = method_key or raw_selection_text

    elif market_family in {
        "finish_only_moneyline",
        "decision_only_moneyline",
        "round_1_only_moneyline",
        "submission_only_moneyline",
        "ko_tko_only_moneyline",
    }:
        market_key = market_family
        outcome_type = "fighter"
        outcome_key = "fighter_win"
        is_conditional_no_action = True
        condition_key = market_family.replace("_moneyline", "")

    return {
        "market_family": market_family,
        "market_key": market_key,
        "outcome_type": outcome_type,
        "outcome_key": outcome_key,
        "side": _selection_side(raw_selection_name, row.get("selection_outcome_type")),
        "line": line,
        "is_conditional_no_action": is_conditional_no_action,
        "condition_key": condition_key,
        "round_number": round_number,
        "method_key": method_key,
    }


def normalize_draftkings_diagnostic_rows(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    """Convert DraftKings diagnostic rows into canonical market catalog rows."""

    rows: list[dict[str, Any]] = []
    for _, row in diagnostic_df.iterrows():
        mapping = _market_mapping(row)
        fighter_name = row.get("selection_participant_name")
        if not fighter_name and mapping.get("outcome_type") in {"fighter", "fighter_line"}:
            fighter_name = row.get("raw_selection_name")

        rows.append(
            {
                "snapshot_run_id": row.get("snapshot_run_id"),
                "snapshot_timestamp": row.get("snapshot_timestamp"),
                "source": row.get("source"),
                "bookmaker": row.get("bookmaker"),
                "provider_event_id": row.get("provider_event_id"),
                "event_name": row.get("event_name"),
                "event_start_timestamp": row.get("event_start_timestamp"),
                "provider_subcategory_id": row.get("provider_subcategory_id"),
                "provider_subcategory_name": row.get("provider_subcategory_name"),
                "provider_market_id": row.get("provider_market_id"),
                "provider_market_name": row.get("raw_market_name"),
                "provider_market_type_id": row.get("provider_market_type_id"),
                "provider_market_type_name": row.get("provider_market_type_name"),
                "provider_selection_id": row.get("provider_selection_id"),
                "provider_selection_name": row.get("raw_selection_name"),
                "fighter_name": fighter_name,
                "fighter_provider_id": row.get("selection_participant_sdid"),
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

    return ensure_canonical_market_columns(pd.DataFrame(rows))
