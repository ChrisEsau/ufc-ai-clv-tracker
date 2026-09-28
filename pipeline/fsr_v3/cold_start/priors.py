"""Prior-combination math for FSR V3 cold start.

External evidence is never treated as UFC observation data.  Instead it defines
an additional prior component with a held-out-calibrated equivalent evidence
strength.  With ``extra_seconds == 0`` this module reproduces the existing
population Gamma prior exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, sqrt

import numpy as np


@dataclass(frozen=True)
class PositiveRatePrior:
    mean_rate_15m: float
    total_seconds: float
    shape: float
    rate: float
    sd_rate_15m: float
    population_seconds: float
    external_seconds: float
    population_mean_rate_15m: float
    external_mean_rate_15m: float | None


def combine_positive_rate_prior(
    *,
    population_mean_rate_15m: float,
    population_seconds: float,
    external_mean_rate_15m: float | None = None,
    extra_seconds: float = 0.0,
) -> PositiveRatePrior:
    """Combine population and external pseudo-evidence on the Gamma-rate scale.

    For the native FSR tendency model, a population prior with mean ``q`` and
    evidence ``K`` seconds has Gamma shape ``q*K/900`` and rate ``K/900``.
    Independent external pseudo-evidence is represented by another Gamma kernel
    with mean ``q_ext`` and equivalent evidence ``K_ext``.  Multiplying those
    kernels yields the formulas below.

    The external strength is not hand assigned in production; it must be chosen
    by held-out native-target validation.  ``extra_seconds=0`` is the exact
    existing population prior.
    """
    q_pop = max(float(population_mean_rate_15m), 1e-12)
    k_pop = max(float(population_seconds), 1e-12)
    k_ext = max(float(extra_seconds), 0.0)

    if external_mean_rate_15m is None or not np.isfinite(external_mean_rate_15m) or k_ext <= 0.0:
        q_ext: float | None = None
        k_ext = 0.0
    else:
        q_ext = max(float(external_mean_rate_15m), 1e-12)

    total_seconds = k_pop + k_ext
    weighted_events = q_pop * k_pop
    if q_ext is not None:
        weighted_events += q_ext * k_ext
    mean = weighted_events / total_seconds
    shape = max(weighted_events / 900.0, 1e-12)
    gamma_rate = max(total_seconds / 900.0, 1e-12)
    sd = sqrt(shape) / gamma_rate

    return PositiveRatePrior(
        mean_rate_15m=float(mean),
        total_seconds=float(total_seconds),
        shape=float(shape),
        rate=float(gamma_rate),
        sd_rate_15m=float(sd),
        population_seconds=float(k_pop),
        external_seconds=float(k_ext),
        population_mean_rate_15m=float(q_pop),
        external_mean_rate_15m=q_ext,
    )


def gamma_logweights(grid: np.ndarray, prior: PositiveRatePrior) -> np.ndarray:
    """Evaluate the combined Gamma prior on an existing FSR positive-rate grid."""
    x = np.asarray(grid, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("positive-rate prior grid must be strictly positive")
    a = float(prior.shape)
    b = float(prior.rate)
    return a * log(b) - lgamma(a) + (a - 1.0) * np.log(x) - b * x
