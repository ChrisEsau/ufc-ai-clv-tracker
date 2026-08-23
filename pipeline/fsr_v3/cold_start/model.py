"""Deterministic external-evidence models and held-out strength calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from pipeline.fsr_v3.replay.math import nb2_log_likelihood

from .features import BASE_FEATURE_COLUMNS
from .priors import combine_positive_rate_prior, gamma_logweights


MODEL_FEATURE_COLUMNS = [
    column
    for column in BASE_FEATURE_COLUMNS
    if column not in {
        "has_external_record", "has_opponent_quality", "has_pathway_stats", "has_pedigree"
    }
]
MODEL_FEATURE_COLUMNS += [
    "has_external_record", "has_opponent_quality", "has_pathway_stats", "has_pedigree"
]


def _nb2_log_likelihood_vector(y, mu, alpha):
    """NB2 log likelihood with row-varying dispersion for external-model fitting."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
    alpha = np.maximum(np.asarray(alpha, dtype=float), 1e-12)
    size = 1.0 / alpha
    p = size / (size + mu)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * np.log(p)
        + y * np.log1p(-p)
    )


@dataclass
class ColdStartNB2RateModel:
    """Ridge-regularized NB2 log-rate model relative to the UFC population rate.

    ``q_external = q_population * exp(intercept + z @ beta)``

    The likelihood is the trait-native UFC next-fight count/exposure NB2
    likelihood.  External data therefore earns influence only when it predicts
    the actual UFC-native target.
    """

    feature_columns: tuple[str, ...] = tuple(MODEL_FEATURE_COLUMNS)
    ridge_alpha: float = 20.0
    intercept_: float = 0.0
    coefficients_: np.ndarray | None = None
    medians_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    fitted_: bool = False

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        missing = sorted(set(self.feature_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"cold-start model frame missing features: {missing}")
        raw = frame.loc[:, self.feature_columns].copy()
        for column in self.feature_columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        arr = raw.to_numpy(dtype=float)

        if fit:
            medians = np.nanmedian(arr, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            filled = np.where(np.isfinite(arr), arr, medians)
            means = filled.mean(axis=0)
            scales = filled.std(axis=0)
            scales = np.where(scales > 1e-8, scales, 1.0)
            self.medians_ = medians
            self.means_ = means
            self.scales_ = scales
        elif self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("cold-start model is not fitted")

        filled = np.where(np.isfinite(arr), arr, self.medians_)
        return (filled - self.means_) / self.scales_

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        count_column: str = "numerator",
        exposure_column: str = "exposure_seconds",
        population_rate_column: str = "population_rate_15m",
        alpha_column: str = "observation_alpha",
    ) -> "ColdStartNB2RateModel":
        required = [count_column, exposure_column, population_rate_column, alpha_column]
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"cold-start fit frame missing target fields: {missing}")
        x = self._matrix(frame, fit=True)
        y = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(float)
        exposure = pd.to_numeric(frame[exposure_column], errors="coerce").to_numpy(float)
        q_pop = pd.to_numeric(frame[population_rate_column], errors="coerce").to_numpy(float)
        alpha = pd.to_numeric(frame[alpha_column], errors="coerce").to_numpy(float)
        keep = (
            np.isfinite(y) & np.isfinite(exposure) & np.isfinite(q_pop) & np.isfinite(alpha)
            & (exposure > 0.0) & (q_pop > 0.0) & (alpha > 0.0)
        )
        x, y, exposure, q_pop, alpha = x[keep], y[keep], exposure[keep], q_pop[keep], alpha[keep]
        if len(y) < 30:
            raise ValueError(f"cold-start fit requires at least 30 usable observations, got {len(y)}")

        p = x.shape[1]
        ridge = max(float(self.ridge_alpha), 0.0)

        def objective(theta: np.ndarray) -> float:
            intercept = float(theta[0])
            beta = theta[1:]
            eta = np.clip(intercept + x @ beta, -3.0, 3.0)
            q = q_pop * np.exp(eta)
            mu = exposure / 900.0 * q
            ll = float(np.sum(_nb2_log_likelihood_vector(y, mu, alpha)))
            penalty = 0.5 * ridge * float(np.dot(beta, beta))
            return -ll + penalty

        result = minimize(
            objective,
            np.zeros(p + 1, dtype=float),
            method="L-BFGS-B",
            bounds=[(-2.0, 2.0)] + [(-2.0, 2.0)] * p,
            options={"maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"cold-start NB2 fit failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.coefficients_ = np.asarray(result.x[1:], dtype=float)
        self.fitted_ = True
        return self

    def predict_rate(
        self,
        frame: pd.DataFrame,
        *,
        population_rate_column: str = "population_rate_15m",
    ) -> np.ndarray:
        if not self.fitted_ or self.coefficients_ is None:
            raise RuntimeError("cold-start model is not fitted")
        if population_rate_column not in frame:
            raise ValueError(f"missing {population_rate_column}")
        x = self._matrix(frame, fit=False)
        q_pop = pd.to_numeric(frame[population_rate_column], errors="coerce").to_numpy(float)
        eta = np.clip(self.intercept_ + x @ self.coefficients_, -3.0, 3.0)
        return np.maximum(q_pop, 1e-12) * np.exp(eta)

    def coefficient_frame(self) -> pd.DataFrame:
        if not self.fitted_ or self.coefficients_ is None:
            raise RuntimeError("cold-start model is not fitted")
        return pd.DataFrame(
            {
                "feature": ["<intercept>", *self.feature_columns],
                "coefficient": [self.intercept_, *self.coefficients_.tolist()],
            }
        ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _posterior_predictive_ll(
    *,
    y: float,
    exposure: float,
    q_pop: float,
    pop_seconds: float,
    q_ext: float,
    extra_seconds: float,
    alpha: float,
    grid: np.ndarray,
) -> float:
    prior = combine_positive_rate_prior(
        population_mean_rate_15m=q_pop,
        population_seconds=pop_seconds,
        external_mean_rate_15m=q_ext,
        extra_seconds=extra_seconds,
    )
    lp = gamma_logweights(grid, prior)
    mu = exposure / 900.0 * grid
    obs_ll = nb2_log_likelihood(y, mu, alpha)
    return float(logsumexp(lp + obs_ll) - logsumexp(lp))


def calibrate_extra_evidence_seconds(
    frame: pd.DataFrame,
    *,
    population_seconds: float,
    grid: np.ndarray,
    candidates: Iterable[float] = (0.0, 30.0, 60.0, 90.0, 180.0, 360.0, 720.0, 1440.0),
    bucket_column: str = "evidence_bucket",
    external_rate_column: str = "external_predicted_rate_15m",
    count_column: str = "numerator",
    exposure_column: str = "exposure_seconds",
    population_rate_column: str = "population_rate_15m",
    alpha_column: str = "observation_alpha",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose external equivalent evidence seconds by held-out native likelihood.

    Calibration is performed separately by evidence-amount bucket.  The ``none``
    bucket is forced to zero, preserving the production population prior exactly
    for fighters with no usable external data.
    """
    required = {
        bucket_column, external_rate_column, count_column, exposure_column,
        population_rate_column, alpha_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cold-start calibration frame missing columns: {missing}")
    values = sorted({max(float(v), 0.0) for v in candidates})
    if 0.0 not in values:
        values = [0.0, *values]

    records: list[dict[str, object]] = []
    chosen: dict[str, float] = {"none": 0.0}
    for bucket, part in frame.groupby(bucket_column, sort=True):
        bucket = str(bucket)
        if bucket == "none":
            chosen[bucket] = 0.0
            continue
        p = part.copy()
        p = p[
            pd.to_numeric(p[exposure_column], errors="coerce").gt(0)
            & pd.to_numeric(p[external_rate_column], errors="coerce").gt(0)
        ]
        if p.empty:
            chosen[bucket] = 0.0
            continue
        for k_ext in values:
            lls = []
            for row in p.itertuples(index=False):
                d = row._asdict()
                lls.append(
                    _posterior_predictive_ll(
                        y=float(d[count_column]),
                        exposure=float(d[exposure_column]),
                        q_pop=float(d[population_rate_column]),
                        pop_seconds=float(population_seconds),
                        q_ext=float(d[external_rate_column]),
                        extra_seconds=float(k_ext),
                        alpha=float(d[alpha_column]),
                        grid=grid,
                    )
                )
            records.append(
                {
                    "evidence_bucket": bucket,
                    "extra_seconds": float(k_ext),
                    "rows": int(len(lls)),
                    "total_predictive_ll": float(np.sum(lls)),
                    "mean_predictive_ll": float(np.mean(lls)),
                }
            )
        bucket_rows = [r for r in records if r["evidence_bucket"] == bucket]
        best = max(bucket_rows, key=lambda r: (r["total_predictive_ll"], -r["extra_seconds"]))
        chosen[bucket] = float(best["extra_seconds"])

    table = pd.DataFrame(records)
    if not table.empty:
        baseline = table[table["extra_seconds"] == 0.0][
            ["evidence_bucket", "total_predictive_ll"]
        ].rename(columns={"total_predictive_ll": "baseline_total_predictive_ll"})
        table = table.merge(baseline, on="evidence_bucket", how="left", validate="many_to_one")
        table["delta_predictive_ll_vs_zero"] = (
            table["total_predictive_ll"] - table["baseline_total_predictive_ll"]
        )
    return chosen, table
