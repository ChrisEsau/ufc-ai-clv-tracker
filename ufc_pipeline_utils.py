"""
ufc_pipeline_utils.py

Shared utility layer for the UFC betting pipeline.

Purpose:
- Keep repeated notebook logic consistent across training, backtest, live prediction,
  and future prop-bet notebooks.
- Avoid copy/paste drift between V4/V5 notebooks.
- Provide one source of truth for paths, odds math, EV, Kelly sizing, feature alignment,
  model artifact loading, and CLV snapshot logging.

Recommended Colab location:
    /content/drive/MyDrive/UFC_AI/ufc_pipeline_utils.py

Usage in notebooks:
    import sys
    sys.path.append("/content/drive/MyDrive/UFC_AI")

    from ufc_pipeline_utils import *
"""

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import (
    LIVE_ACTION_BOARD_PATH,
    LIVE_CARD_PATH,
    LIVE_ODDS_AUDIT_PATH,
    LIVE_WATCHLIST_PATH,
    MODEL_BEST_THRESHOLD_PATH,
    MODEL_CALIBRATED_PATH,
    MODEL_DIR,
    MODEL_FEATURE_COLUMNS_PATH,
    MODEL_PRODUCTION_CONFIG_PKL_PATH,
    MODEL_PREDICTIONS_PATH,
    MODEL_VERSION,
    ROLLING_FEATURES_PATH,
)
from pipeline.common.risk_settings import load_risk_settings, risk_settings_to_betting_filters


# ============================================================
# PATH / CONFIG HELPERS
# ============================================================

@dataclass
class UFCPipelinePaths:
    """
    Centralized path configuration.

    Change model_version here instead of hardcoding production paths
    independently inside every notebook.
    """
    base_path: str = "."
    model_version: str = MODEL_VERSION

    @property
    def production_dir(self) -> str:
        return str(MODEL_DIR)

    @property
    def rolling_features_path(self) -> str:
        return str(ROLLING_FEATURES_PATH)

    @property
    def feature_registry_path(self) -> str:
        return f"{self.base_path}/ufc_engineered_feature_registry.csv"

    @property
    def live_card_output(self) -> str:
        return str(LIVE_CARD_PATH)

    @property
    def live_predictions_output(self) -> str:
        return str(MODEL_PREDICTIONS_PATH)

    @property
    def live_betting_card_output(self) -> str:
        return str(LIVE_CARD_PATH)

    @property
    def watchlist_output(self) -> str:
        return str(LIVE_WATCHLIST_PATH)

    @property
    def action_board_output(self) -> str:
        return str(LIVE_ACTION_BOARD_PATH)

    @property
    def clv_log_path(self) -> str:
        return str(LIVE_ODDS_AUDIT_PATH)


def ensure_dir(path: str) -> None:
    """Create a folder if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================
# NAME NORMALIZATION / MATCHING
# ============================================================
def fuzzy_score(a, b):
    """
    Backward-compatible alias for older notebook code.
    """
    return token_set_score(a, b)


def decimal_kelly_fraction(model_prob, decimal_odds):
    """
    Full Kelly fraction using decimal odds.
    """
    try:
        decimal_odds = float(decimal_odds)
        model_prob = float(model_prob)

        b = decimal_odds - 1
        p = model_prob
        q = 1 - p

        if b <= 0:
            return 0.0

        return max(0.0, ((b * p) - q) / b)

    except Exception:
        return 0.0


def scaled_decimal_kelly_stake(
    bankroll,
    model_prob,
    decimal_odds,
    kelly_multiplier=0.50,
    max_stake_pct=0.03,
    min_stake=0.0,
):
    """
    Risk-controlled Kelly stake using decimal odds.
    """
    full_kelly = decimal_kelly_fraction(
        model_prob,
        decimal_odds
    )

    raw_stake = bankroll * full_kelly * kelly_multiplier

    capped_stake = min(
        raw_stake,
        bankroll * max_stake_pct
    )

    if capped_stake < min_stake:
        return 0.0

    return round(float(capped_stake), 2)
def decimal_ev(model_prob, decimal_odds, risk_amount=100):
    """
    Expected value using decimal odds.

    EV = p * profit_if_win - (1-p) * risk
    """

    try:
        if pd.isna(model_prob) or pd.isna(decimal_odds):
            return 0.0

        model_prob = float(model_prob)
        decimal_odds = float(decimal_odds)

        profit_if_win = risk_amount * (decimal_odds - 1)
        loss_if_lose = risk_amount

        return (
            model_prob * profit_if_win
            - (1 - model_prob) * loss_if_lose
        )

    except Exception:
        return 0.0
    
def to_decimal_rate(value):
    """
    Convert percentage-style stats into decimal form safely.

    Examples:
    55 -> 0.55
    0.55 -> 0.55
    NaN -> 0.0
    """
    try:
        if pd.isna(value):
            return 0.0

        value = float(value)

        if value > 1:
            return value / 100.0

        return value

    except Exception:
        return 0.0
def safe_value(row, column, default=0.0):
    """
    Safely retrieve a numeric value from a dataframe row.
    """
    try:
        value = row.get(column, default)

        if pd.isna(value):
            return default

        return value

    except Exception:
        return default
def normalize_name(name) -> str:
    """
    Normalize fighter names for matching across UFCStats, odds APIs, and CSV files.

    Handles:
    - accents
    - punctuation
    - capitalization
    - extra spaces
    """
    if pd.isna(name):
        return ""

    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def token_set_score(a, b) -> float:
    """
    Lightweight fuzzy score without requiring fuzzywuzzy/rapidfuzz.

    Returns a 0-100 score based on token overlap.
    This is intentionally simple and stable for Colab portability.
    """
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    if not a_norm or not b_norm:
        return 0.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())

    if not a_tokens or not b_tokens:
        return 0.0

    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    return 100.0 * overlap / union


def best_name_match(name, candidates: Iterable[str]) -> Tuple[Optional[str], float]:
    """
    Return the best matching candidate and score for a fighter name.
    """
    best_candidate = None
    best_score = -1.0

    for candidate in candidates:
        score = token_set_score(name, candidate)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate, best_score


# ============================================================
# ODDS HELPERS
# ============================================================

def american_to_decimal(odds) -> float:
    """
    Convert American odds to decimal odds.

    Example:
    -150 -> 1.6667
    +200 -> 3.0000
    """
    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds < 0:
        return 1 + (100 / abs(odds))

    return 1 + (odds / 100)


def american_to_implied_prob(odds) -> float:
    """
    Convert American odds to implied probability before removing vig.
    """
    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds < 0:
        return abs(odds) / (abs(odds) + 100)

    return 100 / (odds + 100)


def decimal_to_american(decimal_odds) -> float:
    """
    Convert decimal odds to American odds.
    """
    if pd.isna(decimal_odds):
        return np.nan

    decimal_odds = float(decimal_odds)

    if decimal_odds <= 1:
        return np.nan

    if decimal_odds >= 2:
        return (decimal_odds - 1) * 100

    return -100 / (decimal_odds - 1)


def remove_two_way_vig(prob_a, prob_b) -> Tuple[float, float]:
    """
    Remove vig from a two-way market by normalizing implied probabilities.
    """
    if pd.isna(prob_a) or pd.isna(prob_b):
        return np.nan, np.nan

    total = prob_a + prob_b

    if total <= 0:
        return np.nan, np.nan

    return prob_a / total, prob_b / total


# ============================================================
# EV / KELLY HELPERS
# ============================================================

def calculate_ev(model_prob, american_odds, risk_amount: float = 100.0) -> float:
    """
    Calculate expected value in dollars for a fixed risk amount.

    EV = probability * profit_if_win - probability_of_loss * risk
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan

    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = risk_amount * (decimal_odds - 1)
    loss_if_lose = risk_amount

    return (model_prob * profit_if_win) - ((1 - model_prob) * loss_if_lose)


def calculate_edge(model_prob, implied_prob) -> float:
    """
    Calculate model edge over market implied probability.
    """
    if pd.isna(model_prob) or pd.isna(implied_prob):
        return np.nan

    return model_prob - implied_prob


def kelly_fraction(model_prob, american_odds) -> float:
    """
    Full Kelly fraction.

    b = decimal_odds - 1
    p = model probability
    q = 1 - p
    f* = (bp - q) / b
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return 0.0

    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    p = float(model_prob)
    q = 1 - p

    if b <= 0:
        return 0.0

    fraction = ((b * p) - q) / b

    return max(0.0, fraction)


def scaled_kelly_stake(
    bankroll: float | None,
    model_prob,
    american_odds,
    kelly_multiplier: float | None = None,
    max_stake_pct: float | None = None,
    min_stake: float = 0.0,
) -> float:
    """
    Calculate a risk-controlled Kelly stake.

    Defaults come from pipeline.common.risk_settings.
    """
    settings = load_risk_settings()
    bankroll = settings.starting_bankroll if bankroll is None else bankroll
    kelly_multiplier = settings.kelly_fraction if kelly_multiplier is None else kelly_multiplier
    max_stake_pct = settings.max_stake_pct if max_stake_pct is None else max_stake_pct

    full_kelly = kelly_fraction(model_prob, american_odds)
    raw_stake = bankroll * full_kelly * kelly_multiplier
    capped_stake = min(raw_stake, bankroll * max_stake_pct)

    if capped_stake < min_stake:
        return 0.0

    return round(float(capped_stake), 2)


# ============================================================
# PROBABILITY / FEATURE HELPERS
# ============================================================

def clip_probability(prob, low: float = 0.03, high: float = 0.97) -> float:
    """
    Clip a single model probability to reduce overconfidence.
    """
    if pd.isna(prob):
        return np.nan

    return float(np.clip(prob, low, high))


def clip_probability_series(series: pd.Series, low: float = 0.03, high: float = 0.97) -> pd.Series:
    """
    Clip a probability column.
    """
    return series.clip(lower=low, upper=high)


def align_features(df: pd.DataFrame, feature_columns: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    """
    Force a dataframe to match the exact feature order used in training.

    Missing features are created.
    Extra features are ignored.
    """
    aligned = df.copy()

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = fill_value

    return aligned[feature_columns].copy()


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], label: str = "dataframe") -> None:
    """
    Raise a clear error if expected columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


# ============================================================
# MODEL ARTIFACT LOADING
# ============================================================

def load_production_artifacts(paths: UFCPipelinePaths) -> Dict:
    """
    Load frozen model artifacts from the production directory.
    """
    artifacts = {
        "model": joblib.load(MODEL_CALIBRATED_PATH),
        "feature_columns": joblib.load(MODEL_FEATURE_COLUMNS_PATH),
        "best_threshold": joblib.load(MODEL_BEST_THRESHOLD_PATH),
        "production_config": joblib.load(MODEL_PRODUCTION_CONFIG_PKL_PATH),
    }

    return artifacts


def get_betting_filters(production_config: Dict | None = None) -> Dict:
    """
    Return canonical betting filters from persisted risk settings.

    The production_config argument is accepted for backwards compatibility, but
    pipeline risk settings are the single source of truth.
    """
    return risk_settings_to_betting_filters(load_risk_settings())


# ============================================================
# CLV SNAPSHOT HELPERS
# ============================================================

def append_clv_snapshot(
    df: pd.DataFrame,
    output_path: str,
    snapshot_columns: List[str],
    timestamp_col: str = "snapshot_time",
) -> pd.DataFrame:
    """
    Append a CLV snapshot to an existing CSV log.

    If the file does not exist, it creates it.
    """
    snapshot = df[snapshot_columns].copy()
    snapshot[timestamp_col] = datetime.now(timezone.utc).isoformat()

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    combined.to_csv(output_path, index=False)

    return combined


# ============================================================
# BET FILTER HELPERS
# ============================================================

def apply_standard_bet_filters(
    df: pd.DataFrame,
    edge_col: str = "best_edge",
    confidence_col: str = "best_confidence",
    odds_col: str = "best_american_odds",
    ev_col: str = "best_ev",
    min_edge: float | None = None,
    min_confidence: float | None = None,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> pd.DataFrame:
    """
    Apply standard production betting filters.
    """
    settings = load_risk_settings()
    min_edge = settings.min_edge if min_edge is None else min_edge
    min_confidence = settings.min_confidence if min_confidence is None else min_confidence
    min_odds = settings.min_odds if min_odds is None else min_odds
    max_odds = settings.max_odds if max_odds is None else max_odds

    out = df.copy()

    out["passes_edge_filter"] = out[edge_col] >= min_edge
    out["passes_confidence_filter"] = out[confidence_col] >= min_confidence
    out["passes_odds_filter"] = (
        (out[odds_col] >= min_odds)
        &
        (out[odds_col] <= max_odds)
    )
    out["passes_positive_ev_filter"] = out[ev_col] > 0

    filter_cols = [
        "passes_edge_filter",
        "passes_confidence_filter",
        "passes_odds_filter",
        "passes_positive_ev_filter",
    ]

    out["is_official_bet"] = out[filter_cols].all(axis=1)

    return out
