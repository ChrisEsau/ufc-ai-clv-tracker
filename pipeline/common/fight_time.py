from __future__ import annotations

import re
from typing import Any

import pandas as pd

ROUND_SECONDS = 300


def clock_time_to_seconds(value: Any) -> float:
    """Convert a final-round clock string like '4:18' to seconds within that round."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if not text or ":" not in text:
        return float("nan")
    parts = text.split(":")
    if len(parts) != 2:
        return float("nan")
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return float("nan")
    if minutes < 0 or seconds < 0 or seconds >= 60:
        return float("nan")
    return float(minutes * 60 + seconds)


def elapsed_fight_time_seconds(finish_round: Any, final_round_time_seconds: Any) -> float:
    """Return total elapsed fight time in seconds.

    UFCStats fight rows store the clock time inside the final round. For a fight
    ending in round 3 at 2:00, total elapsed time is 12:00, not 2:00:

        ((finish_round - 1) * 300) + final_round_clock_seconds

    This helper intentionally assumes standard 5-minute UFC rounds because the
    existing master schema stores only finish round and final-round clock time.
    """
    round_number = pd.to_numeric(pd.Series([finish_round]), errors="coerce").iloc[0]
    round_clock = pd.to_numeric(pd.Series([final_round_time_seconds]), errors="coerce").iloc[0]
    if pd.isna(round_number) or pd.isna(round_clock):
        return float("nan")
    round_number = int(round_number)
    round_clock = float(round_clock)
    if round_number <= 0 or round_clock < 0:
        return float("nan")
    return float(((round_number - 1) * ROUND_SECONDS) + round_clock)


def repair_elapsed_match_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with match_time_sec converted to total elapsed fight time.

    Historical artifacts may contain match_time_sec as the final-round clock only.
    This is detectable for rounds after round 1 when match_time_sec <= 300. Rows
    already storing elapsed time are left unchanged.
    """
    out = df.copy()
    if "match_time_sec" not in out.columns or "finish_round" not in out.columns:
        return out

    finish_round = pd.to_numeric(out["finish_round"], errors="coerce")
    match_time = pd.to_numeric(out["match_time_sec"], errors="coerce")
    needs_repair = finish_round.gt(1) & match_time.notna() & match_time.le(ROUND_SECONDS)

    repaired = ((finish_round - 1) * ROUND_SECONDS) + match_time
    out.loc[needs_repair, "match_time_sec"] = repaired.loc[needs_repair]
    return out


def needs_elapsed_match_time_repair(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for rows that look like final-round-only match_time_sec values."""
    if "match_time_sec" not in df.columns or "finish_round" not in df.columns:
        return pd.Series(False, index=df.index)
    finish_round = pd.to_numeric(df["finish_round"], errors="coerce")
    match_time = pd.to_numeric(df["match_time_sec"], errors="coerce")
    return finish_round.gt(1) & match_time.notna() & match_time.le(ROUND_SECONDS)
