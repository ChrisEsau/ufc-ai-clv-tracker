"""Leakage-safe calibration for significant-strike pace predictions.

Point models trained on log-transformed pace systematically underpredict the
arithmetic mean. This module applies a sequential multiplicative correction and
estimates gamma-Poisson overdispersion using only completed walk-forward years
before the year being evaluated.

The calibrated outputs remain shadow-only. They are intended to define the
statistical contract that a future round-parameter provider will expose to the
simulator; they do not alter production predictions or betting artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error


class SigAttemptCalibrationError(RuntimeError):
    """Raised when strike-pace calibration inputs or outputs are invalid."""


@dataclass(frozen=True)
class SigAttemptCalibrationResult:
    schedule: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    final_parameters: pd.DataFrame


REQUIRED_COLUMNS = (
    "model_name",
    "test_year",
    "fight_id",
    "target_sig_attempted",
    "round_exposure_seconds",
    "predicted_rate_per_min",
    "predicted_count_at_actual_exposure",
)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SigAttemptCalibrationError(
            f"Walk-forward predictions are missing required columns: {missing}"
        )


def _coerce_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(predictions, REQUIRED_COLUMNS)
    out = predictions.copy()
    numeric_columns = (
        "test_year",
        "target_sig_attempted",
        "round_exposure_seconds",
        "predicted_rate_per_min",
        "predicted_count_at_actual_exposure",
    )
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[list(numeric_columns)].isna().any().any():
        raise SigAttemptCalibrationError(
            "Walk-forward predictions contain missing numeric calibration values"
        )
    if out["target_sig_attempted"].lt(0).any():
        raise SigAttemptCalibrationError("Observed strike attempts must be nonnegative")
    if out["round_exposure_seconds"].le(0).any():
        raise SigAttemptCalibrationError("Round exposure must be positive")
    if out["predicted_rate_per_min"].le(0).any():
        raise SigAttemptCalibrationError("Predicted strike pace must be positive")
    if out["predicted_count_at_actual_exposure"].le(0).any():
        raise SigAttemptCalibrationError("Predicted strike counts must be positive")
    out["test_year"] = out["test_year"].astype(int)
    return out


def multiplicative_mean_factor(
    actual_count: np.ndarray,
    predicted_count: np.ndarray,
    low: float = 0.50,
    high: float = 2.00,
) -> float:
    """Return a bounded factor aligning predicted and observed total counts."""
    actual = np.asarray(actual_count, dtype=float)
    predicted = np.asarray(predicted_count, dtype=float)
    predicted_total = float(predicted.sum())
    if predicted_total <= 0:
        raise SigAttemptCalibrationError("Predicted total must be positive")
    factor = float(actual.sum() / predicted_total)
    return float(np.clip(factor, low, high))


def gamma_poisson_overdispersion(
    actual_count: np.ndarray,
    predicted_mean: np.ndarray,
    floor: float = 0.001,
    ceiling: float = 5.0,
) -> float:
    """Estimate alpha in Var(Y|mu) = mu + alpha * mu^2.

    The estimator is a pooled method-of-moments ratio. It matches the simulator's
    gamma-Poisson sampler, where ``overdispersion`` is alpha.
    """
    actual = np.asarray(actual_count, dtype=float)
    mean = np.clip(np.asarray(predicted_mean, dtype=float), 0.001, None)
    numerator = float(np.sum(np.square(actual - mean) - actual))
    denominator = float(np.sum(np.square(mean)))
    if denominator <= 0:
        raise SigAttemptCalibrationError("Cannot estimate dispersion with zero mean")
    alpha = numerator / denominator
    return float(np.clip(alpha, floor, ceiling))


def _metric_row(
    model_name: str,
    calibration: str,
    frame: pd.DataFrame,
    predicted_count: np.ndarray,
) -> dict[str, object]:
    actual = frame["target_sig_attempted"].to_numpy(dtype=float)
    predicted = np.clip(np.asarray(predicted_count, dtype=float), 0.001, None)
    exposure_minutes = frame["round_exposure_seconds"].to_numpy(dtype=float) / 60.0
    actual_rate = actual / exposure_minutes
    predicted_rate = predicted / exposure_minutes
    weights = np.clip(exposure_minutes / 5.0, 0.02, 1.0)
    return {
        "model_name": model_name,
        "calibration": calibration,
        "rows": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()),
        "count_mae": float(mean_absolute_error(actual, predicted)),
        "count_rmse": float(sqrt(mean_squared_error(actual, predicted))),
        "count_poisson_deviance": float(mean_poisson_deviance(actual, predicted)),
        "weighted_rate_mae": float(
            np.average(np.abs(actual_rate - predicted_rate), weights=weights)
        ),
        "actual_mean_count": float(actual.mean()),
        "predicted_mean_count": float(predicted.mean()),
        "mean_count_bias": float(predicted.mean() - actual.mean()),
    }


def _schedule_for_model(
    frame: pd.DataFrame,
    minimum_prior_rows: int,
    default_overdispersion: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = str(frame["model_name"].iloc[0])
    years = sorted(frame["test_year"].unique().tolist())
    schedule_rows: list[dict[str, object]] = []
    calibrated_frames: list[pd.DataFrame] = []

    for year in years:
        current = frame.loc[frame["test_year"].eq(year)].copy()
        prior = frame.loc[frame["test_year"].lt(year)].copy()

        if len(prior) >= minimum_prior_rows:
            factor = multiplicative_mean_factor(
                prior["target_sig_attempted"].to_numpy(),
                prior["predicted_count_at_actual_exposure"].to_numpy(),
            )
            prior_calibrated_mean = (
                prior["predicted_count_at_actual_exposure"].to_numpy(dtype=float)
                * factor
            )
            overdispersion = gamma_poisson_overdispersion(
                prior["target_sig_attempted"].to_numpy(),
                prior_calibrated_mean,
            )
            source = "prior_walk_forward_years"
        else:
            factor = 1.0
            overdispersion = float(default_overdispersion)
            source = "cold_start_default"

        current["calibration_factor"] = factor
        current["gamma_poisson_overdispersion"] = overdispersion
        current["calibrated_rate_per_min"] = (
            current["predicted_rate_per_min"] * factor
        )
        current["calibrated_count_at_actual_exposure"] = (
            current["predicted_count_at_actual_exposure"] * factor
        )
        calibrated_frames.append(current)

        schedule_rows.append(
            {
                "model_name": model_name,
                "test_year": int(year),
                "prior_rows": int(len(prior)),
                "prior_fights": int(prior["fight_id"].nunique()),
                "calibration_factor": float(factor),
                "gamma_poisson_overdispersion": float(overdispersion),
                "calibration_source": source,
            }
        )

    return pd.DataFrame(schedule_rows), pd.concat(calibrated_frames, ignore_index=True)


def calibrate_walk_forward_predictions(
    predictions: pd.DataFrame,
    model_names: Iterable[str] = ("xgb_context", "xgb_context_rfs"),
    minimum_prior_rows: int = 1_000,
    default_overdispersion: float = 0.35,
) -> SigAttemptCalibrationResult:
    """Sequentially calibrate selected model predictions by test year."""
    if minimum_prior_rows < 1:
        raise SigAttemptCalibrationError("minimum_prior_rows must be positive")
    if default_overdispersion <= 0:
        raise SigAttemptCalibrationError("default_overdispersion must be positive")

    df = _coerce_predictions(predictions)
    requested = tuple(model_names)
    available = set(df["model_name"].astype(str).unique())
    missing = sorted(set(requested) - available)
    if missing:
        raise SigAttemptCalibrationError(
            f"Requested models are missing from walk-forward predictions: {missing}"
        )

    schedules: list[pd.DataFrame] = []
    calibrated: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for model_name in requested:
        model_frame = df.loc[df["model_name"].eq(model_name)].copy()
        schedule, calibrated_frame = _schedule_for_model(
            model_frame,
            minimum_prior_rows=minimum_prior_rows,
            default_overdispersion=default_overdispersion,
        )
        schedules.append(schedule)
        calibrated.append(calibrated_frame)

        metric_rows.append(
            _metric_row(
                model_name,
                "raw",
                model_frame,
                model_frame["predicted_count_at_actual_exposure"].to_numpy(),
            )
        )
        metric_rows.append(
            _metric_row(
                model_name,
                "sequential_mean_calibrated",
                calibrated_frame,
                calibrated_frame["calibrated_count_at_actual_exposure"].to_numpy(),
            )
        )

        final_factor = multiplicative_mean_factor(
            model_frame["target_sig_attempted"].to_numpy(),
            model_frame["predicted_count_at_actual_exposure"].to_numpy(),
        )
        final_mean = (
            model_frame["predicted_count_at_actual_exposure"].to_numpy(dtype=float)
            * final_factor
        )
        final_dispersion = gamma_poisson_overdispersion(
            model_frame["target_sig_attempted"].to_numpy(),
            final_mean,
        )
        final_rows.append(
            {
                "model_name": model_name,
                "calibration_rows": int(len(model_frame)),
                "calibration_fights": int(model_frame["fight_id"].nunique()),
                "mean_calibration_factor": float(final_factor),
                "gamma_poisson_overdispersion": float(final_dispersion),
                "variance_contract": "variance = mean + alpha * mean^2",
            }
        )

    prediction_output = pd.concat(calibrated, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["model_name", "calibration"]
    ).reset_index(drop=True)

    raw_lookup = metrics.loc[metrics["calibration"].eq("raw")].set_index("model_name")
    calibrated_lookup = metrics.loc[
        metrics["calibration"].eq("sequential_mean_calibrated")
    ].set_index("model_name")
    improvements: dict[str, float] = {}
    for model_name in requested:
        raw_deviance = float(raw_lookup.loc[model_name, "count_poisson_deviance"])
        calibrated_deviance = float(
            calibrated_lookup.loc[model_name, "count_poisson_deviance"]
        )
        improvements[model_name] = (raw_deviance - calibrated_deviance) / raw_deviance
    metrics["poisson_improvement_vs_raw"] = metrics.apply(
        lambda row: (
            improvements[str(row["model_name"])]
            if row["calibration"] == "sequential_mean_calibrated"
            else 0.0
        ),
        axis=1,
    )

    return SigAttemptCalibrationResult(
        schedule=pd.concat(schedules, ignore_index=True),
        predictions=prediction_output,
        metrics=metrics,
        final_parameters=pd.DataFrame(final_rows),
    )
