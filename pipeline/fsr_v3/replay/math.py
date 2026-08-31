"""Small numerical helpers shared by FSR V3 replay modules."""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln


def nb2_log_likelihood(y, mu, alpha):
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
    alpha = max(float(alpha), 1e-12)
    size = 1.0 / alpha
    p = size / (size + mu)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * np.log(p)
        + y * np.log1p(-p)
    )


def beta_binomial_log_likelihood(y, n, p, rho):
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    rho = max(float(rho), 1e-8)
    concentration = 1.0 / rho - 1.0
    a = p * concentration
    b = (1.0 - p) * concentration
    choose = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
    return choose + betaln(y + a, n - y + b) - betaln(a, b)


def normalize_log_weights(log_weights):
    log_weights = np.asarray(log_weights, dtype=float)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("posterior normalization failed")
    return weights / total


def weighted_mean_sd(grid, weights):
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float(np.sum(grid * weights))
    variance = float(np.sum(weights * (grid - mean) ** 2))
    return mean, float(np.sqrt(max(variance, 0.0)))
