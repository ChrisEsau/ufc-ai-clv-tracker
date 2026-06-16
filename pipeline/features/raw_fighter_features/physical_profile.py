"""Physical profile raw fighter feature plugin.

Copies point-in-time fighter physical attributes from the master fight row into
fighter-state history. These values are sourced from profile enrichment columns
on the red/blue corners and are emitted before the feature-view diff generator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


OUTPUT_COLUMNS = ["age", "height", "reach", "weight"]


def initial_state() -> dict[str, Any]:
    """Physical profile has no mutable fight-history state."""

    return {}


def calculate(
    fighter_history: pd.DataFrame,
    fight_row: pd.Series,
    context: dict | None = None,
) -> dict[str, float]:
    """Return physical attributes for the fighter being snapshotted."""

    del fighter_history
    context = context or {}
    fighter_id = str(context.get("fighter_id", ""))

    side = _side_for_fighter(fight_row, fighter_id)
    if side is None:
        return {column: 0.0 for column in OUTPUT_COLUMNS}

    return {
        "age": _calculate_age(fight_row.get(f"{side}_dob"), fight_row.get("date")),
        "height": _parse_numeric_measure(fight_row.get(f"{side}_height")),
        "reach": _parse_numeric_measure(fight_row.get(f"{side}_reach")),
        "weight": _parse_numeric_measure(fight_row.get(f"{side}_weight")),
    }


def _side_for_fighter(row: pd.Series, fighter_id: str) -> str | None:
    if fighter_id and str(row.get("r_id", "")) == fighter_id:
        return "r"
    if fighter_id and str(row.get("b_id", "")) == fighter_id:
        return "b"
    return None


def _calculate_age(dob_value: Any, fight_date_value: Any) -> float:
    dob = _parse_date(dob_value)
    fight_date = _parse_date(fight_date_value)
    if dob is None or fight_date is None:
        return 0.0
    return max((fight_date - dob).days / 365.25, 0.0)


def _parse_date(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _parse_numeric_measure(value: Any) -> float:
    """Parse height/reach/weight values into numeric units when possible.

    Heights like 5'11" are converted to inches. Numeric strings like "72" or
    "170 lbs" are converted to floats. Missing/unparseable values return 0.0.
    """

    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if not text or text in {"--", "nan", "none"}:
        return 0.0

    if "'" in text:
        cleaned = text.replace("\"", "").replace("in", "")
        feet_text, _, inches_text = cleaned.partition("'")
        try:
            feet = float(feet_text.strip() or 0)
            inches = float(inches_text.strip() or 0)
            return feet * 12.0 + inches
        except ValueError:
            return 0.0

    numeric = "".join(ch if ch.isdigit() or ch in {".", "-"} else " " for ch in text)
    parts = [part for part in numeric.split() if part]
    if not parts:
        return 0.0
    try:
        return float(parts[0])
    except ValueError:
        return 0.0
