"""Model evaluation metrics for UFC model training.

This module is model-agnostic. It evaluates any binary classifier probability
series against a binary target series and produces reusable artifacts for:
- model governance
- threshold selection
- calibration monitoring
- future betting-board confidence bucket displays
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class EvaluationResult:
    """Container for model evaluation outputs."""

    metrics: dict[str, Any]
    threshold_sweep: pd.DataFrame
    confidence_buckets: pd.DataFrame
    best_threshold: float


def evaluate_binary_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold_min: float = 0.40,
    threshold_max: float = 0.60,
    threshold_step: float = 0.01,
    bucket_edges: list[float] | None = None,
    probability_label: str = "model_probability",
) -> EvaluationResult:
    """Evaluate binary classifier probabilities.

    Returns core metrics, a threshold sweep, and calibration/confidence buckets.
    """
    y_array = _to_1d_numeric_array(y_true, name="y_true").astype(int)
    prob_array = _clip_probabilities(
        _to_1d_numeric_array(probabilities, name="probabilities")
    )

    if len(y_array) != len(prob_array):
        raise ValueError(
            f"y_true and probabilities length mismatch: {len(y_array)} != {len(prob_array)}"
        )

    threshold_sweep = calculate_threshold_sweep(
        y_true=y_array,
        probabilities=prob_array,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )
    best_threshold = float(threshold_sweep.iloc[0]["threshold"])

    predictions = (prob_array >= best_threshold).astype(int)
    metrics = calculate_core_metrics(
        y_true=y_array,
        probabilities=prob_array,
        predictions=predictions,
        threshold=best_threshold,
    )

    confidence_buckets = calculate_confidence_buckets(
        y_true=y_array,
        probabilities=prob_array,
        bucket_edges=bucket_edges,
        probability_label=probability_label,
    )

    return EvaluationResult(
        metrics=metrics,
        threshold_sweep=threshold_sweep,
        confidence_buckets=confidence_buckets,
        best_threshold=best_threshold,
    )


def calculate_core_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate primary binary probability and classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "positive_pick_rate_percent": float(predictions.mean() * 100),
        "negative_pick_rate_percent": float((1 - predictions.mean()) * 100),
        "actual_positive_rate_percent": float(y_true.mean() * 100),
        "row_count": int(len(y_true)),
    }


def calculate_threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold_min: float = 0.40,
    threshold_max: float = 0.60,
    threshold_step: float = 0.01,
) -> pd.DataFrame:
    """Return threshold performance table sorted by accuracy descending."""
    thresholds = np.arange(
        threshold_min,
        threshold_max + (threshold_step / 2),
        threshold_step,
    )

    rows = []
    for threshold in thresholds:
        rounded_threshold = round(float(threshold), 2)
        preds = (probabilities >= rounded_threshold).astype(int)
        rows.append(
            {
                "threshold": rounded_threshold,
                "accuracy": float(accuracy_score(y_true, preds)),
                "positive_pick_rate": float(preds.mean() * 100),
                "negative_pick_rate": float((1 - preds.mean()) * 100),
                "roc_auc": float(roc_auc_score(y_true, probabilities)),
                "log_loss": float(log_loss(y_true, probabilities)),
                "brier_score": float(brier_score_loss(y_true, probabilities)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["accuracy", "threshold"],
        ascending=[False, True],
    ).reset_index(drop=True)


def calculate_confidence_buckets(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bucket_edges: list[float] | None = None,
    probability_label: str = "model_probability",
) -> pd.DataFrame:
    """Create calibration/confidence buckets for probability quality monitoring.

    Default buckets cover the model-decision confidence range from 50% to 100%.
    This is intended for matchup models where confidence is interpreted as the
    probability assigned to the predicted side.
    """
    edges = bucket_edges or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    confidence = np.maximum(probabilities, 1 - probabilities)
    predicted_class = (probabilities >= 0.50).astype(int)
    actual_correct = (predicted_class == y_true).astype(int)

    bucket_df = pd.DataFrame(
        {
            "y_true": y_true,
            probability_label: probabilities,
            "model_confidence": confidence,
            "predicted_class": predicted_class,
            "correct_prediction": actual_correct,
        }
    )

    bucket_df["bucket"] = pd.cut(
        bucket_df["model_confidence"],
        bins=edges,
        include_lowest=True,
        right=False,
    )

    rows = []
    for bucket, group in bucket_df.groupby("bucket", observed=True):
        if group.empty:
            continue

        bucket_min = float(bucket.left)
        bucket_max = float(bucket.right)
        avg_confidence = float(group["model_confidence"].mean())
        actual_accuracy = float(group["correct_prediction"].mean())

        rows.append(
            {
                "bucket": f"{bucket_min:.2f}-{bucket_max:.2f}",
                "bucket_min_prob": bucket_min,
                "bucket_max_prob": bucket_max,
                "fight_count": int(len(group)),
                "avg_predicted_confidence": avg_confidence,
                "actual_win_rate": actual_accuracy,
                "calibration_error": float(actual_accuracy - avg_confidence),
                "accuracy": actual_accuracy,
                "avg_raw_probability": float(group[probability_label].mean()),
                "positive_pick_rate": float(group["predicted_class"].mean() * 100),
            }
        )

    return pd.DataFrame(rows)


def compare_raw_vs_calibrated(
    y_true: pd.Series | np.ndarray,
    raw_probabilities: pd.Series | np.ndarray,
    calibrated_probabilities: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return side-by-side core metrics for raw and calibrated probabilities."""
    raw_eval = evaluate_binary_probabilities(y_true, raw_probabilities)
    calibrated_eval = evaluate_binary_probabilities(y_true, calibrated_probabilities)

    rows = []
    for label, result in (("raw", raw_eval), ("calibrated", calibrated_eval)):
        for metric, value in result.metrics.items():
            rows.append({"probability_type": label, "metric": metric, "value": value})

    return pd.DataFrame(rows)


def _to_1d_numeric_array(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    """Convert values into a 1D numeric numpy array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN values")
    return array


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip probabilities to avoid log-loss edge cases at exactly 0 or 1."""
    return np.clip(probabilities, 1e-6, 1 - 1e-6)
