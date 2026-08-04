"""Historical replay diagnostics for calibrated strike-attempt distributions.

Replay uses out-of-fold fighter-round predictions and their sequential calibration.
It scores the distribution directly at the observed exposure. No heuristic
simulator pace, regime, fatigue, suppression, or confidence multipliers are
applied.

The first replay evaluates univariate count calibration and diagnoses paired
fighter/opponent residual correlation. It intentionally does not claim that the
independent gamma-Poisson component captures joint fight pace.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom, poisson
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error


class SigAttemptReplayError(RuntimeError):
    """Raised when calibrated replay inputs or diagnostics are invalid."""


@dataclass(frozen=True)
class SigAttemptReplayResult:
    aggregate_metrics: pd.DataFrame
    interval_coverage: pd.DataFrame
    metrics_by_year: pd.DataFrame
    metrics_by_round: pd.DataFrame
    calibration_deciles: pd.DataFrame
    pair_diagnostics: pd.DataFrame
    replay_rows: pd.DataFrame


REQUIRED_COLUMNS = (
    "fight_id",
    "fighter_id",
    "opponent_id",
    "round",
    "test_year",
    "model_name",
    "target_sig_attempted",
    "round_exposure_seconds",
    "predicted_count_at_actual_exposure",
    "calibrated_count_at_actual_exposure",
    "calibrated_rate_per_min",
    "gamma_poisson_overdispersion",
)

INTERVAL_LEVELS = (0.50, 0.80, 0.90)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SigAttemptReplayError(
            f"Calibrated replay predictions are missing columns: {missing}"
        )


def _coerce_replay_frame(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    _require_columns(predictions, REQUIRED_COLUMNS)
    frame = predictions.loc[
        predictions["model_name"].astype(str).eq(str(model_name))
    ].copy()
    if frame.empty:
        raise SigAttemptReplayError(
            f"Calibrated predictions contain no rows for model {model_name!r}"
        )

    numeric_columns = (
        "round",
        "test_year",
        "target_sig_attempted",
        "round_exposure_seconds",
        "predicted_count_at_actual_exposure",
        "calibrated_count_at_actual_exposure",
        "calibrated_rate_per_min",
        "gamma_poisson_overdispersion",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(numeric_columns)].isna().any().any():
        raise SigAttemptReplayError("Replay predictions contain missing numeric values")
    if frame["target_sig_attempted"].lt(0).any():
        raise SigAttemptReplayError("Observed strike counts must be nonnegative")
    if frame["round_exposure_seconds"].le(0).any():
        raise SigAttemptReplayError("Replay exposure must be positive")
    positive_columns = (
        "predicted_count_at_actual_exposure",
        "calibrated_count_at_actual_exposure",
        "calibrated_rate_per_min",
        "gamma_poisson_overdispersion",
    )
    if frame[list(positive_columns)].le(0).any().any():
        raise SigAttemptReplayError("Replay means, rates, and dispersion must be positive")

    frame["round"] = frame["round"].astype(int)
    frame["test_year"] = frame["test_year"].astype(int)
    duplicate_count = int(
        frame.duplicated(["fight_id", "fighter_id", "round"]).sum()
    )
    if duplicate_count:
        raise SigAttemptReplayError(
            f"Replay predictions contain duplicate fighter-round keys: {duplicate_count}"
        )
    return frame.reset_index(drop=True)


def gamma_poisson_logpmf(
    observed: np.ndarray,
    mean: np.ndarray,
    overdispersion: np.ndarray,
) -> np.ndarray:
    """Return log PMF under Var(Y)=mu+alpha*mu^2."""
    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    alpha = np.clip(np.asarray(overdispersion, dtype=float), 1e-9, None)
    if y.shape != mu.shape or y.shape != alpha.shape:
        raise SigAttemptReplayError("Observed, mean, and dispersion shapes must match")
    if np.any(y < 0) or np.any(np.floor(y) != y):
        raise SigAttemptReplayError("Gamma-Poisson observations must be integer counts")

    size = 1.0 / alpha
    probability = size / (size + mu)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * np.log(probability)
        + y * np.log1p(-probability)
    )


def _distribution_quantile(
    probability: float,
    mean: np.ndarray,
    overdispersion: np.ndarray | None,
) -> np.ndarray:
    mu = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    if overdispersion is None:
        return poisson.ppf(probability, mu).astype(float)
    alpha = np.clip(np.asarray(overdispersion, dtype=float), 1e-9, None)
    size = 1.0 / alpha
    success_probability = size / (size + mu)
    return nbinom.ppf(probability, size, success_probability).astype(float)


def _metric_row(
    frame: pd.DataFrame,
    distribution_name: str,
    mean_column: str,
    overdispersion_column: str | None,
) -> dict[str, object]:
    actual = frame["target_sig_attempted"].to_numpy(dtype=float)
    mean = np.clip(frame[mean_column].to_numpy(dtype=float), 1e-9, None)
    if overdispersion_column is None:
        logpmf = poisson.logpmf(actual, mean)
        variance = mean
    else:
        alpha = frame[overdispersion_column].to_numpy(dtype=float)
        logpmf = gamma_poisson_logpmf(actual, mean, alpha)
        variance = mean + alpha * np.square(mean)

    residual = actual - mean
    standardized = residual / np.sqrt(np.clip(variance, 1e-9, None))
    return {
        "distribution": distribution_name,
        "rows": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()),
        "mean_negative_log_likelihood": float(-np.mean(logpmf)),
        "count_poisson_deviance": float(mean_poisson_deviance(actual, mean)),
        "count_mae": float(mean_absolute_error(actual, mean)),
        "count_rmse": float(sqrt(mean_squared_error(actual, mean))),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(mean.mean()),
        "mean_bias": float(mean.mean() - actual.mean()),
        "actual_variance": float(np.var(actual, ddof=0)),
        "average_predicted_variance": float(np.mean(variance)),
        "variance_ratio_predicted_to_actual": float(
            np.mean(variance) / np.var(actual, ddof=0)
        ),
        "standardized_residual_mean": float(np.mean(standardized)),
        "standardized_residual_std": float(np.std(standardized, ddof=0)),
    }


def _interval_rows(
    frame: pd.DataFrame,
    distribution_name: str,
    mean_column: str,
    overdispersion_column: str | None,
) -> list[dict[str, object]]:
    actual = frame["target_sig_attempted"].to_numpy(dtype=float)
    mean = frame[mean_column].to_numpy(dtype=float)
    alpha = (
        None
        if overdispersion_column is None
        else frame[overdispersion_column].to_numpy(dtype=float)
    )
    rows: list[dict[str, object]] = []
    for level in INTERVAL_LEVELS:
        tail = (1.0 - level) / 2.0
        lower = _distribution_quantile(tail, mean, alpha)
        upper = _distribution_quantile(1.0 - tail, mean, alpha)
        covered = (actual >= lower) & (actual <= upper)
        rows.append(
            {
                "distribution": distribution_name,
                "nominal_coverage": float(level),
                "empirical_coverage": float(covered.mean()),
                "coverage_error": float(covered.mean() - level),
                "mean_interval_width": float(np.mean(upper - lower)),
                "median_interval_width": float(np.median(upper - lower)),
            }
        )
    return rows


def _group_metrics(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, group in frame.groupby(group_column, dropna=False):
        row = _metric_row(
            group,
            "calibrated_gamma_poisson",
            "calibrated_count_at_actual_exposure",
            "gamma_poisson_overdispersion",
        )
        row[group_column] = group_value
        rows.append(row)
    return pd.DataFrame(rows)


def _calibration_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rank = out["calibrated_count_at_actual_exposure"].rank(
        method="first",
        pct=True,
    )
    out["prediction_decile"] = np.ceil(rank * 10.0).clip(1, 10).astype(int)
    result = (
        out.groupby("prediction_decile")
        .agg(
            rows=("fight_id", "size"),
            fights=("fight_id", "nunique"),
            predicted_mean=("calibrated_count_at_actual_exposure", "mean"),
            actual_mean=("target_sig_attempted", "mean"),
            predicted_min=("calibrated_count_at_actual_exposure", "min"),
            predicted_max=("calibrated_count_at_actual_exposure", "max"),
        )
        .reset_index()
    )
    result["mean_error"] = result["predicted_mean"] - result["actual_mean"]
    return result


def _pair_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    pair_sizes = frame.groupby(["fight_id", "round"])["fighter_id"].nunique()
    valid_keys = pair_sizes.loc[pair_sizes.eq(2)].index
    paired = frame.set_index(["fight_id", "round"]).loc[valid_keys].reset_index()
    paired = paired.sort_values(["fight_id", "round", "fighter_id"])

    rows: list[dict[str, object]] = []
    first_actual: list[float] = []
    second_actual: list[float] = []
    first_mean: list[float] = []
    second_mean: list[float] = []
    first_residual: list[float] = []
    second_residual: list[float] = []

    for _, group in paired.groupby(["fight_id", "round"], sort=False):
        if len(group) != 2:
            continue
        first = group.iloc[0]
        second = group.iloc[1]
        first_actual.append(float(first["target_sig_attempted"]))
        second_actual.append(float(second["target_sig_attempted"]))
        first_mean.append(float(first["calibrated_count_at_actual_exposure"]))
        second_mean.append(float(second["calibrated_count_at_actual_exposure"]))
        first_residual.append(first_actual[-1] - first_mean[-1])
        second_residual.append(second_actual[-1] - second_mean[-1])

    def correlation(left: list[float], right: list[float]) -> float:
        if len(left) < 2:
            return float("nan")
        return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])

    actual_corr = correlation(first_actual, second_actual)
    mean_corr = correlation(first_mean, second_mean)
    residual_corr = correlation(first_residual, second_residual)
    rows.extend(
        [
            {
                "diagnostic": "paired_actual_count_correlation",
                "value": actual_corr,
                "pairs": len(first_actual),
                "interpretation": "Observed within-fight-round fighter/opponent count correlation.",
            },
            {
                "diagnostic": "paired_predicted_mean_correlation",
                "value": mean_corr,
                "pairs": len(first_actual),
                "interpretation": "Correlation already represented by paired predicted means.",
            },
            {
                "diagnostic": "paired_residual_correlation",
                "value": residual_corr,
                "pairs": len(first_actual),
                "interpretation": "Remaining dependence not represented by independent gamma-Poisson draws.",
            },
        ]
    )
    return pd.DataFrame(rows)


def replay_calibrated_strike_distribution(
    calibrated_predictions: pd.DataFrame,
    model_name: str = "xgb_context_rfs",
) -> SigAttemptReplayResult:
    """Score calibrated out-of-fold strike distributions and joint diagnostics."""
    frame = _coerce_replay_frame(calibrated_predictions, model_name=model_name)
    frame["raw_residual"] = (
        frame["target_sig_attempted"]
        - frame["predicted_count_at_actual_exposure"]
    )
    frame["calibrated_residual"] = (
        frame["target_sig_attempted"]
        - frame["calibrated_count_at_actual_exposure"]
    )

    distribution_specs = (
        (
            "raw_poisson",
            "predicted_count_at_actual_exposure",
            None,
        ),
        (
            "calibrated_poisson",
            "calibrated_count_at_actual_exposure",
            None,
        ),
        (
            "calibrated_gamma_poisson",
            "calibrated_count_at_actual_exposure",
            "gamma_poisson_overdispersion",
        ),
    )

    aggregate_rows = [
        _metric_row(frame, distribution, mean_column, alpha_column)
        for distribution, mean_column, alpha_column in distribution_specs
    ]
    interval_rows: list[dict[str, object]] = []
    for distribution, mean_column, alpha_column in distribution_specs:
        interval_rows.extend(
            _interval_rows(frame, distribution, mean_column, alpha_column)
        )

    return SigAttemptReplayResult(
        aggregate_metrics=pd.DataFrame(aggregate_rows).sort_values(
            "mean_negative_log_likelihood"
        ).reset_index(drop=True),
        interval_coverage=pd.DataFrame(interval_rows),
        metrics_by_year=_group_metrics(frame, "test_year"),
        metrics_by_round=_group_metrics(frame, "round"),
        calibration_deciles=_calibration_deciles(frame),
        pair_diagnostics=_pair_diagnostics(frame),
        replay_rows=frame,
    )
