from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _clean_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _fmt_line(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except Exception:
        return _clean_token(value)

    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def build_comparison_key(row: pd.Series | dict[str, Any]) -> str | None:
    """Build the canonical cross-sportsbook comparison key for one market row."""

    get = row.get
    market_key = _clean_token(get("market_key"))
    side = _clean_token(get("side"))
    outcome_key = _clean_token(get("outcome_key"))
    fighter_name = _clean_token(get("fighter_name"))
    method_key = _clean_token(get("method_key"))
    line = _fmt_line(get("line"))
    round_number = _fmt_line(get("round_number"))
    condition_key = _clean_token(get("condition_key"))

    if not market_key:
        return None

    if market_key in {
        "moneyline",
        "finish_only_moneyline",
        "decision_only_moneyline",
        "round_1_only_moneyline",
        "submission_only_moneyline",
        "ko_tko_only_moneyline",
    }:
        base = f"fighter:{fighter_name}" if fighter_name else outcome_key or side
        return f"{condition_key}:{base}" if condition_key else base

    if market_key == "goes_distance":
        return side or outcome_key

    if market_key in {"total_rounds", "fighter_sig_strikes_total"}:
        if side and line:
            return f"{side}:{line}"
        return side or outcome_key

    if market_key == "point_spread":
        if fighter_name and line:
            return f"fighter:{fighter_name}:{line}"
        return fighter_name or side or outcome_key

    if market_key in {"win_by_ko_tko_dq", "win_by_submission", "win_by_decision"}:
        if fighter_name:
            return f"fighter:{fighter_name}:{market_key.replace('win_by_', '')}"
        return market_key

    if market_key == "exact_method":
        return method_key or outcome_key

    if market_key == "round_method":
        parts = []
        if fighter_name:
            parts.append(f"fighter:{fighter_name}")
        if method_key:
            parts.append(method_key)
        elif outcome_key:
            parts.append(outcome_key)
        if round_number:
            parts.append(f"r{round_number}")
        return ":".join(parts) if parts else outcome_key or side

    if fighter_name:
        return f"fighter:{fighter_name}:{outcome_key or side or market_key}"

    return outcome_key or side or market_key
