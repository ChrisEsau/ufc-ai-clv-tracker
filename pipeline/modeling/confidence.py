# ============================================================
# pipeline/modeling/confidence.py
# ============================================================

"""Prediction confidence scoring utilities.

Confidence is intentionally separate from model probability. Probability is the
model's forecast for an outcome. Confidence measures how much the platform
trusts that forecast based on feature coverage, feature-family availability,
fighter history depth, data-quality flags, and historical bucket reliability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_BUCKETS_PATH = Path("models/moneyline/xgboost_v5/confidence_buckets.parquet")


@dataclass(frozen=True)
class PredictionConfidence:
    confidence_score: float
    confidence_pct: float
    confidence_tier: str
    confidence_data_quality: float
    confidence_feature_coverage: float
    confidence_family_coverage: float
    confidence_history_depth: float
    confidence_calibration_reliability: float
    confidence_bucket: str | None
    confidence_bucket_accuracy: float | None
    confidence_bucket_fight_count: int | None
    confidence_bucket_calibration_error: float | None
    confidence_penalty_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(value, default: float | None = None) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def _as_bool(value, default: bool = True) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(max(value, low), high))


@lru_cache(maxsize=8)
def load_confidence_buckets(path: str = str(DEFAULT_CONFIDENCE_BUCKETS_PATH)) -> pd.DataFrame:
    """Load persisted training confidence buckets when available."""

    bucket_path = Path(path)
    if not bucket_path.exists():
        return pd.DataFrame()

    try:
        buckets = pd.read_parquet(bucket_path)
    except Exception:
        return pd.DataFrame()

    required = {"bucket_min_prob", "bucket_max_prob", "accuracy"}
    if not required.issubset(buckets.columns):
        return pd.DataFrame()

    return buckets.copy()


def _numeric_nonzero_ratio(row: pd.Series, columns: list[str]) -> float | None:
    values = []
    for column in columns:
        value = _as_float(row.get(column), default=None)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(values) > 1e-12))


def _feature_completeness(row: pd.Series) -> float:
    expected = _as_float(row.get("feature_count_expected"), default=None)
    actual = _as_float(row.get("feature_count_actual"), default=None)
    nonzero = _as_float(row.get("nonzero_feature_count"), default=None)
    zero_pct = _as_float(row.get("zero_feature_pct"), default=None)

    if expected and nonzero is not None:
        return _bounded(nonzero / expected)

    if actual and nonzero is not None:
        return _bounded(nonzero / actual)

    if zero_pct is not None:
        return _bounded(1.0 - zero_pct)

    return 0.50


def _family_coverage(row: pd.Series) -> float:
    """Estimate coverage of important feature families from live feature columns."""

    family_prefixes = {
        "ewm": ["ewm_", "r_state_ewm_", "b_state_ewm_", "r_ewm_", "b_ewm_"],
        "recent_form": ["recent_form_", "r_recent_form_", "b_recent_form_"],
        "elo": ["elo_", "r_elo", "b_elo", "r_state_elo", "b_state_elo"],
        "striking": ["splm", "sapm", "str_acc", "str_def"],
        "grappling": ["td_avg", "td_acc", "td_def", "sub_avg", "ctrl"],
    }

    scores = []
    column_names = [str(column) for column in row.index]

    for prefixes in family_prefixes.values():
        cols = [
            column
            for column in column_names
            if any(str(column).startswith(prefix) or prefix in str(column) for prefix in prefixes)
        ]
        ratio = _numeric_nonzero_ratio(row, cols)
        if ratio is not None:
            scores.append(ratio)

    if not scores:
        return _feature_completeness(row)

    return _bounded(float(np.mean(scores)))


def _history_depth(row: pd.Series) -> float:
    """Estimate fighter history depth using available fight-count columns."""

    candidates = [
        ("r_state_ewm_fights", "b_state_ewm_fights"),
        ("r_ewm_fights", "b_ewm_fights"),
        ("r_fights", "b_fights"),
        ("red_fights", "blue_fights"),
    ]

    red_fights = None
    blue_fights = None
    for red_col, blue_col in candidates:
        red_fights = _as_float(row.get(red_col), default=None)
        blue_fights = _as_float(row.get(blue_col), default=None)
        if red_fights is not None or blue_fights is not None:
            break

    values = [value for value in [red_fights, blue_fights] if value is not None]
    if not values:
        return 0.50

    # Full history credit around 8 prior fights per fighter, partial credit below.
    scores = [min(max(value, 0.0) / 8.0, 1.0) for value in values]
    return _bounded(float(np.mean(scores)))


def _data_quality(row: pd.Series) -> float:
    passes_model_data_quality = _as_bool(row.get("passes_model_data_quality"), default=True)
    passes_feature_validation = _as_bool(row.get("passes_feature_validation"), default=True)
    feature_match_type = str(row.get("feature_match_type", "")).strip().lower()

    score = 1.0
    if not passes_model_data_quality:
        score *= 0.45
    if not passes_feature_validation:
        score *= 0.55
    if feature_match_type and feature_match_type != "both_matched":
        score *= 0.70

    return _bounded(score)


def _select_confidence_bucket(model_pick_probability: float | None, buckets: pd.DataFrame) -> pd.Series | None:
    """Select the training confidence bucket matching the pick probability."""

    if model_pick_probability is None or buckets.empty:
        return None

    pick_probability = _bounded(float(model_pick_probability))

    matches = buckets[
        (pd.to_numeric(buckets["bucket_min_prob"], errors="coerce") <= pick_probability)
        & (pick_probability < pd.to_numeric(buckets["bucket_max_prob"], errors="coerce"))
    ]

    if matches.empty and pick_probability >= 1.0:
        matches = buckets[pd.to_numeric(buckets["bucket_max_prob"], errors="coerce") >= 1.0]

    if matches.empty:
        return None

    return matches.iloc[0]


def _calibration_reliability(
    *,
    model_pick_probability: float | None,
    confidence_buckets_path: str | Path = DEFAULT_CONFIDENCE_BUCKETS_PATH,
) -> tuple[float, dict[str, Any]]:
    """Return historical bucket accuracy for the current pick probability.

    Probability is used only to choose the historical bucket. The reliability
    signal itself is the bucket's historical accuracy.
    """

    buckets = load_confidence_buckets(str(confidence_buckets_path))
    bucket = _select_confidence_bucket(model_pick_probability, buckets)

    if bucket is None:
        return 0.65, {
            "confidence_bucket": None,
            "confidence_bucket_accuracy": None,
            "confidence_bucket_fight_count": None,
            "confidence_bucket_calibration_error": None,
        }

    accuracy = _bounded(_as_float(bucket.get("accuracy"), default=0.65) or 0.65)
    fight_count = _as_float(bucket.get("fight_count"), default=None)
    calibration_error = _as_float(bucket.get("calibration_error"), default=None)

    return accuracy, {
        "confidence_bucket": bucket.get("bucket"),
        "confidence_bucket_accuracy": accuracy,
        "confidence_bucket_fight_count": None if fight_count is None else int(fight_count),
        "confidence_bucket_calibration_error": calibration_error,
    }


def _tier(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Medium"
    if score >= 0.35:
        return "Low"
    return "Very Low"


def score_prediction_confidence(
    row: pd.Series,
    *,
    model_pick_probability: float | None = None,
    confidence_buckets_path: str | Path = DEFAULT_CONFIDENCE_BUCKETS_PATH,
) -> PredictionConfidence:
    """Score prediction trust from feature quality rather than probability."""

    data_quality = _data_quality(row)
    feature_coverage = _feature_completeness(row)
    family_coverage = _family_coverage(row)
    history_depth = _history_depth(row)
    calibration_reliability, bucket_metadata = _calibration_reliability(
        model_pick_probability=model_pick_probability,
        confidence_buckets_path=confidence_buckets_path,
    )

    raw_score = (
        0.35 * feature_coverage
        + 0.25 * family_coverage
        + 0.20 * history_depth
        + 0.20 * calibration_reliability
    )

    penalty_reasons = []
    capped_score = raw_score * data_quality

    if history_depth <= 0.10:
        capped_score = min(capped_score, 0.35)
        penalty_reasons.append("debut_or_near_debut_history")
    elif history_depth <= 0.35:
        capped_score = min(capped_score, 0.50)
        penalty_reasons.append("sparse_fighter_history")

    if family_coverage <= 0.40:
        capped_score = min(capped_score, 0.60)
        penalty_reasons.append("low_feature_family_coverage")

    if not _as_bool(row.get("passes_model_data_quality"), default=True):
        capped_score = min(capped_score, 0.40)
        penalty_reasons.append("model_data_quality_failed")

    if not _as_bool(row.get("passes_feature_validation"), default=True):
        capped_score = min(capped_score, 0.45)
        penalty_reasons.append("feature_validation_failed")

    score = _bounded(capped_score)

    return PredictionConfidence(
        confidence_score=score,
        confidence_pct=score * 100.0,
        confidence_tier=_tier(score),
        confidence_data_quality=data_quality,
        confidence_feature_coverage=feature_coverage,
        confidence_family_coverage=family_coverage,
        confidence_history_depth=history_depth,
        confidence_calibration_reliability=calibration_reliability,
        confidence_bucket=bucket_metadata["confidence_bucket"],
        confidence_bucket_accuracy=bucket_metadata["confidence_bucket_accuracy"],
        confidence_bucket_fight_count=bucket_metadata["confidence_bucket_fight_count"],
        confidence_bucket_calibration_error=bucket_metadata["confidence_bucket_calibration_error"],
        confidence_penalty_reason=";".join(penalty_reasons) if penalty_reasons else "none",
    )
