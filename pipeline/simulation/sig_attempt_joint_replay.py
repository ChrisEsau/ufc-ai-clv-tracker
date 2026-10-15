"""Sequential joint replay for paired significant-strike attempt distributions.

The calibrated gamma-Poisson component has strong marginal performance, but
fighter and opponent residuals remain positively dependent within a fight-round.
This module adds a Gaussian copula around the existing negative-binomial
marginals. The copula preserves each fighter's calibrated marginal distribution
while introducing a shared fight-round pace shock.

Dependence is estimated sequentially: the copula correlation used for test year Y
is calculated only from probability-integral-transform residuals in completed
walk-forward years before Y. This remains a shadow-only replay diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm


class SigAttemptJointReplayError(RuntimeError):
    """Raised when paired replay inputs or dependence estimates are invalid."""


@dataclass(frozen=True)
class SigAttemptJointReplayResult:
    dependence_schedule: pd.DataFrame
    correlation_metrics: pd.DataFrame
    total_interval_coverage: pd.DataFrame
    final_dependence: pd.DataFrame


REQUIRED_COLUMNS = (
    "fight_id",
    "fighter_id",
    "opponent_id",
    "round",
    "test_year",
    "model_name",
    "target_sig_attempted",
    "calibrated_count_at_actual_exposure",
    "gamma_poisson_overdispersion",
)

INTERVAL_LEVELS = (0.50, 0.80, 0.90)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SigAttemptJointReplayError(
            f"Joint replay predictions are missing required columns: {missing}"
        )


def _coerce_frame(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    _require_columns(predictions, REQUIRED_COLUMNS)
    frame = predictions.loc[
        predictions["model_name"].astype(str).eq(str(model_name))
    ].copy()
    if frame.empty:
        raise SigAttemptJointReplayError(
            f"No calibrated rows found for model {model_name!r}"
        )

    numeric_columns = (
        "round",
        "test_year",
        "target_sig_attempted",
        "calibrated_count_at_actual_exposure",
        "gamma_poisson_overdispersion",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(numeric_columns)].isna().any().any():
        raise SigAttemptJointReplayError("Joint replay contains missing numeric values")
    if frame["target_sig_attempted"].lt(0).any():
        raise SigAttemptJointReplayError("Observed counts must be nonnegative")
    if frame["calibrated_count_at_actual_exposure"].le(0).any():
        raise SigAttemptJointReplayError("Calibrated means must be positive")
    if frame["gamma_poisson_overdispersion"].le(0).any():
        raise SigAttemptJointReplayError("Overdispersion must be positive")

    frame["round"] = frame["round"].astype(int)
    frame["test_year"] = frame["test_year"].astype(int)
    duplicate_count = int(
        frame.duplicated(["fight_id", "fighter_id", "round"]).sum()
    )
    if duplicate_count:
        raise SigAttemptJointReplayError(
            f"Joint replay contains duplicate fighter-round keys: {duplicate_count}"
        )
    return frame


def _mid_pit_z(observed: np.ndarray, mean: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return deterministic mid-PIT normal scores for discrete NB marginals."""
    y = np.asarray(observed, dtype=int)
    mu = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    dispersion = np.clip(np.asarray(alpha, dtype=float), 1e-9, None)
    size = 1.0 / dispersion
    probability = size / (size + mu)
    lower = nbinom.cdf(y - 1, size, probability)
    upper = nbinom.cdf(y, size, probability)
    uniform = np.clip((lower + upper) / 2.0, 1e-6, 1.0 - 1e-6)
    return norm.ppf(uniform)


def _pair_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per fight-round with two fighter marginals."""
    rows: list[dict[str, object]] = []
    for (fight_id, round_number), group in frame.groupby(
        ["fight_id", "round"],
        sort=False,
    ):
        if len(group) != 2 or group["fighter_id"].nunique() != 2:
            continue
        ordered = group.sort_values("fighter_id").reset_index(drop=True)
        first, second = ordered.iloc[0], ordered.iloc[1]
        if int(first["test_year"]) != int(second["test_year"]):
            raise SigAttemptJointReplayError(
                f"Paired rows disagree on test_year for fight {fight_id}, round {round_number}"
            )
        rows.append(
            {
                "fight_id": str(fight_id),
                "round": int(round_number),
                "test_year": int(first["test_year"]),
                "fighter_1_id": str(first["fighter_id"]),
                "fighter_2_id": str(second["fighter_id"]),
                "actual_1": int(first["target_sig_attempted"]),
                "actual_2": int(second["target_sig_attempted"]),
                "mean_1": float(first["calibrated_count_at_actual_exposure"]),
                "mean_2": float(second["calibrated_count_at_actual_exposure"]),
                "alpha_1": float(first["gamma_poisson_overdispersion"]),
                "alpha_2": float(second["gamma_poisson_overdispersion"]),
            }
        )
    paired = pd.DataFrame(rows)
    if paired.empty:
        raise SigAttemptJointReplayError("No complete fighter/opponent pairs were available")

    paired["pit_z_1"] = _mid_pit_z(
        paired["actual_1"].to_numpy(),
        paired["mean_1"].to_numpy(),
        paired["alpha_1"].to_numpy(),
    )
    paired["pit_z_2"] = _mid_pit_z(
        paired["actual_2"].to_numpy(),
        paired["mean_2"].to_numpy(),
        paired["alpha_2"].to_numpy(),
    )
    return paired


def estimate_gaussian_copula_rho(
    pairs: pd.DataFrame,
    lower: float = 0.0,
    upper: float = 0.95,
) -> float:
    """Estimate bounded Gaussian-copula dependence from paired PIT scores."""
    if len(pairs) < 2:
        raise SigAttemptJointReplayError("At least two pairs are required to estimate rho")
    correlation = float(
        np.corrcoef(
            pairs["pit_z_1"].to_numpy(dtype=float),
            pairs["pit_z_2"].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(correlation):
        raise SigAttemptJointReplayError("Estimated copula correlation is not finite")
    return float(np.clip(correlation, lower, upper))


def _sample_nb_from_uniform(
    uniform: np.ndarray,
    mean: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    mu = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    dispersion = np.clip(np.asarray(alpha, dtype=float), 1e-9, None)
    size = 1.0 / dispersion
    probability = size / (size + mu)
    return nbinom.ppf(
        np.clip(uniform, 1e-9, 1.0 - 1e-9),
        size,
        probability,
    ).astype(float)


def sample_gaussian_copula_pairs(
    rng: np.random.Generator,
    pairs: pd.DataFrame,
    rho: float,
    simulations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired NB marginals under a Gaussian copula."""
    correlation = float(rho)
    if not -0.99 < correlation < 0.99:
        raise SigAttemptJointReplayError("rho must be strictly between -0.99 and 0.99")
    if simulations <= 0:
        raise SigAttemptJointReplayError("simulations must be positive")

    pair_count = len(pairs)
    z_1 = rng.normal(size=(simulations, pair_count))
    z_independent = rng.normal(size=(simulations, pair_count))
    z_2 = correlation * z_1 + np.sqrt(1.0 - correlation**2) * z_independent
    u_1 = norm.cdf(z_1)
    u_2 = norm.cdf(z_2)

    draw_1 = _sample_nb_from_uniform(
        u_1,
        pairs["mean_1"].to_numpy(dtype=float)[None, :],
        pairs["alpha_1"].to_numpy(dtype=float)[None, :],
    )
    draw_2 = _sample_nb_from_uniform(
        u_2,
        pairs["mean_2"].to_numpy(dtype=float)[None, :],
        pairs["alpha_2"].to_numpy(dtype=float)[None, :],
    )
    return draw_1, draw_2


def _rowwise_correlations(draw_1: np.ndarray, draw_2: np.ndarray) -> np.ndarray:
    left = draw_1 - draw_1.mean(axis=1, keepdims=True)
    right = draw_2 - draw_2.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(np.square(left), axis=1) * np.sum(np.square(right), axis=1)
    )
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(len(draw_1), np.nan, dtype=float),
        where=denominator > 0,
    )


def _interval_metrics(
    pairs: pd.DataFrame,
    totals: np.ndarray,
    model_name: str,
    test_year: int,
) -> list[dict[str, object]]:
    actual_total = (
        pairs["actual_1"].to_numpy(dtype=float)
        + pairs["actual_2"].to_numpy(dtype=float)
    )
    rows: list[dict[str, object]] = []
    for level in INTERVAL_LEVELS:
        tail = (1.0 - level) / 2.0
        lower = np.quantile(totals, tail, axis=0)
        upper = np.quantile(totals, 1.0 - tail, axis=0)
        covered = (actual_total >= lower) & (actual_total <= upper)
        rows.append(
            {
                "joint_model": model_name,
                "test_year": int(test_year),
                "pairs": int(len(pairs)),
                "nominal_coverage": float(level),
                "empirical_coverage": float(covered.mean()),
                "coverage_error": float(covered.mean() - level),
                "mean_interval_width": float(np.mean(upper - lower)),
                "actual_total_mean": float(actual_total.mean()),
                "simulated_total_mean": float(totals.mean()),
                "actual_total_variance": float(np.var(actual_total, ddof=0)),
                "simulated_total_variance": float(
                    np.mean(np.var(totals, axis=1, ddof=0))
                ),
            }
        )
    return rows


def _joint_metrics_for_year(
    pairs: pd.DataFrame,
    rho: float,
    simulations: int,
    seed: int,
    test_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    actual_corr = float(
        np.corrcoef(
            pairs["actual_1"].to_numpy(dtype=float),
            pairs["actual_2"].to_numpy(dtype=float),
        )[0, 1]
    )
    correlation_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []

    for offset, (model_name, model_rho) in enumerate(
        (("independent", 0.0), ("gaussian_copula", rho))
    ):
        rng = np.random.default_rng(seed + offset)
        draw_1, draw_2 = sample_gaussian_copula_pairs(
            rng,
            pairs,
            rho=model_rho,
            simulations=simulations,
        )
        correlations = _rowwise_correlations(draw_1, draw_2)
        correlations = correlations[np.isfinite(correlations)]
        correlation_rows.append(
            {
                "joint_model": model_name,
                "test_year": int(test_year),
                "pairs": int(len(pairs)),
                "rho_used": float(model_rho),
                "actual_count_correlation": actual_corr,
                "simulated_correlation_mean": float(np.mean(correlations)),
                "simulated_correlation_p05": float(np.quantile(correlations, 0.05)),
                "simulated_correlation_p95": float(np.quantile(correlations, 0.95)),
                "absolute_correlation_error": float(
                    abs(np.mean(correlations) - actual_corr)
                ),
            }
        )
        interval_rows.extend(
            _interval_metrics(
                pairs,
                draw_1 + draw_2,
                model_name=model_name,
                test_year=test_year,
            )
        )
    return correlation_rows, interval_rows


def sequential_joint_strike_replay(
    calibrated_predictions: pd.DataFrame,
    model_name: str = "xgb_context_rfs",
    minimum_prior_pairs: int = 500,
    simulations: int = 750,
    seed: int = 71,
) -> SigAttemptJointReplayResult:
    """Replay independent and sequential copula pair distributions by year."""
    if minimum_prior_pairs < 2:
        raise SigAttemptJointReplayError("minimum_prior_pairs must be at least 2")
    frame = _coerce_frame(calibrated_predictions, model_name=model_name)
    pairs = _pair_rows(frame)

    schedule_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []

    years = sorted(pairs["test_year"].unique().tolist())
    for fold_index, test_year in enumerate(years):
        current = pairs.loc[pairs["test_year"].eq(test_year)].copy()
        prior = pairs.loc[pairs["test_year"].lt(test_year)].copy()
        if len(prior) >= minimum_prior_pairs:
            rho = estimate_gaussian_copula_rho(prior)
            source = "prior_walk_forward_years"
        else:
            rho = 0.0
            source = "cold_start_independent"

        schedule_rows.append(
            {
                "test_year": int(test_year),
                "prior_pairs": int(len(prior)),
                "current_pairs": int(len(current)),
                "gaussian_copula_rho": float(rho),
                "dependence_source": source,
            }
        )
        year_correlations, year_intervals = _joint_metrics_for_year(
            current,
            rho=rho,
            simulations=simulations,
            seed=seed + fold_index * 10,
            test_year=int(test_year),
        )
        correlation_rows.extend(year_correlations)
        interval_rows.extend(year_intervals)

    final_rho = estimate_gaussian_copula_rho(pairs)
    final_dependence = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "pairs": int(len(pairs)),
                "fights": int(pairs["fight_id"].nunique()),
                "gaussian_copula_rho": float(final_rho),
                "dependence_contract": (
                    "Gaussian copula over calibrated gamma-Poisson marginals"
                ),
            }
        ]
    )

    return SigAttemptJointReplayResult(
        dependence_schedule=pd.DataFrame(schedule_rows),
        correlation_metrics=pd.DataFrame(correlation_rows),
        total_interval_coverage=pd.DataFrame(interval_rows),
        final_dependence=final_dependence,
    )
