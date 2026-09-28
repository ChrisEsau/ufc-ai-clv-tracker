"""Canonical schemas and deterministic identifiers for FSR V3 cold start."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

EXTERNAL_BOUT_COLUMNS = (
    "fight_id",
    "event_date",
    "organization",
    "is_major_org",
    "fighter_name",
    "opponent_name",
    "result",
    "method_class",
    "round_num",
    "time_finish_seconds",
    "fighter_height_cm",
    "fighter_weight_kg",
    "opponent_height_cm",
    "opponent_weight_kg",
)

RESULT_VALUES = {"W", "L", "D", "NC"}
METHOD_VALUES = {"KO_TKO", "SUB", "DEC", "OTHER"}


def normalize_name(value: object) -> str:
    """Stable conservative name key; never performs fuzzy/LLM matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalize_method(value: object) -> str:
    text = str(value or "").strip().lower()
    if "decision" in text:
        return "DEC"
    if "submission" in text or text.startswith("sub"):
        return "SUB"
    if "tko" in text or text == "ko" or text.startswith("ko ") or text.startswith("ko("):
        return "KO_TKO"
    return "OTHER"


def validate_external_bouts(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(EXTERNAL_BOUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"external bout frame missing required columns: {missing}")
    x = frame.copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="raise").dt.normalize()
    x["fight_id"] = x["fight_id"].astype(str)
    x["organization"] = x["organization"].fillna("unknown").astype(str).str.lower().str.strip()
    x["fighter_name"] = x["fighter_name"].astype(str)
    x["opponent_name"] = x["opponent_name"].astype(str)
    x["fighter_key"] = x["fighter_name"].map(normalize_name)
    x["opponent_key"] = x["opponent_name"].map(normalize_name)
    x["result"] = x["result"].fillna("NC").astype(str).str.upper().str.strip()
    bad_result = sorted(set(x["result"]).difference(RESULT_VALUES))
    if bad_result:
        raise ValueError(f"unsupported result values: {bad_result}")
    x["method_class"] = x["method_class"].map(normalize_method)
    for column in (
        "round_num", "time_finish_seconds", "fighter_height_cm", "fighter_weight_kg",
        "opponent_height_cm", "opponent_weight_kg", "fighter_pre_elo", "opponent_pre_elo",
        "fighter_post_elo",
    ):
        if column in x.columns:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    if x.duplicated(["fight_id", "fighter_key"]).any():
        raise ValueError("external bout frame has duplicate fight/fighter rows")
    if (x["fighter_key"] == "").any() or (x["opponent_key"] == "").any():
        raise ValueError("external bout frame contains empty normalized fighter names")
    return x.sort_values(["event_date", "fight_id", "fighter_key"]).reset_index(drop=True)


def evidence_bucket(external_bouts: object) -> str:
    try:
        n = int(external_bouts)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return "none"
    if n <= 2:
        return "1_2"
    if n <= 5:
        return "3_5"
    return "6plus"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
