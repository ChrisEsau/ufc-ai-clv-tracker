from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.common.paths import CLV_RESULTS_PATH, MODEL_CANDIDATE_CLV_PATH, MODEL_CANDIDATE_TRACKER_PATH


@dataclass
class CLVIntelligenceData:
    candidate_clv: pd.DataFrame
    candidate_tracker: pd.DataFrame
    official_clv: pd.DataFrame


def _read_parquet(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def load_clv_intelligence_data() -> CLVIntelligenceData:
    return CLVIntelligenceData(
        candidate_clv=_read_parquet(MODEL_CANDIDATE_CLV_PATH),
        candidate_tracker=_read_parquet(MODEL_CANDIDATE_TRACKER_PATH),
        official_clv=_read_parquet(CLV_RESULTS_PATH),
    )


def prepare_candidate_clv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for column in [
        "candidate_timestamp",
        "commence_time",
        "closing_timestamp",
    ]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    for column in [
        "candidate_edge",
        "candidate_edge_pct",
        "candidate_confidence_pct",
        "candidate_ev",
        "candidate_odds",
        "closing_odds",
        "clv_pct",
        "hours_before_fight",
        "hours_before_close",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "beat_closing_line" in out.columns:
        out["beat_closing_line"] = out["beat_closing_line"].astype("boolean")
    return out


def prepare_official_clv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for column in ["placed_timestamp", "closing_timestamp", "event_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    for column in ["odds_taken", "closing_odds", "clv_pct", "edge", "ev", "model_probability"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    model_filter = st_state("clvi_filter_model", "All Models")
    market_filter = st_state("clvi_filter_market", "All Markets")
    book_filter = st_state("clvi_filter_book", "All Books")
    status_filter = st_state("clvi_filter_status", "All")
    if model_filter != "All Models" and "model_id" in out.columns:
        out = out[out["model_id"].astype(str) == model_filter]
    if market_filter != "All Markets" and "market_key" in out.columns:
        out = out[out["market_key"].astype(str) == market_filter]
    if book_filter != "All Books" and "bookmaker" in out.columns:
        out = out[out["bookmaker"].astype(str) == book_filter]
    if status_filter == "Priced" and "candidate_clv_status" in out.columns:
        out = out[out["candidate_clv_status"].astype(str).str.lower() == "priced"]
    elif status_filter == "Missing Close" and "candidate_clv_status" in out.columns:
        out = out[out["candidate_clv_status"].astype(str).str.lower() == "missing_close"]
    return out


def st_state(key: str, default):
    import streamlit as st

    return st.session_state.get(key, default)


def pct(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def signed_pct(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.{decimals}f}%"


def number(value, decimals: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def american(value) -> str:
    if pd.isna(value):
        return "—"
    value = int(round(float(value)))
    return f"+{value}" if value > 0 else str(value)
