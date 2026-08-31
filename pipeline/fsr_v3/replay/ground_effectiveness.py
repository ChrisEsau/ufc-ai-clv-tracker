"""Validated attacker-only FSR V3 ground-striking effectiveness replay.

Population landing probability is fit by Beta-Binomial likelihood, matching the
validated research study.  Fighter offense effects are posterior means under
O ~ Normal(0, .25^2); epistemic sampling is disabled (c=0).
"""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.math import (
    beta_binomial_log_likelihood,
    normalize_log_weights,
    weighted_mean_sd,
)

KEYS = [
    "event_date",
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
]


def _normal_prior(grid: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    return -0.5 * (grid / sigma) ** 2 - log(sigma)


def _fit_population_beta(
    landed_history: list[float],
    attempted_history: list[float],
    previous_beta: float | None,
    config: FSRV3Config,
) -> float:
    """Fit the population intercept using only strictly prior observations."""
    if not attempted_history:
        return 0.0  # neutral 50% before any UFC ground-strike evidence exists

    y = np.asarray(landed_history, dtype=float)
    n = np.asarray(attempted_history, dtype=float)

    def objective(beta):
        p = expit(float(beta))
        return -float(
            beta_binomial_log_likelihood(
                y,
                n,
                p,
                config.ground_effectiveness_rho,
            ).sum()
        )

    result = minimize_scalar(
        objective,
        bounds=(-5.0, 5.0),
        method="bounded",
    )
    if not result.success:
        if previous_beta is not None:
            return float(previous_beta)
        aggregate = float((y.sum() + 0.5) / (n.sum() + 1.0))
        aggregate = float(np.clip(aggregate, 1e-6, 1.0 - 1e-6))
        return log(aggregate / (1.0 - aggregate))
    return float(result.x)


def replay_ground_effectiveness(
    fights: pd.DataFrame,
    config: FSRV3Config | None = None,
) -> pd.DataFrame:
    config = config or FSRV3Config()
    offense_grid = np.linspace(
        config.ground_effectiveness_grid_min,
        config.ground_effectiveness_grid_max,
        config.ground_effectiveness_grid_points,
    )
    prior_lp = _normal_prior(offense_grid, config.ground_effectiveness_sigma)

    states: dict[str, np.ndarray] = {}
    landed_history: list[float] = []
    attempted_history: list[float] = []
    population_beta: float | None = None
    rows: list[dict] = []

    for event_date, batch in fights.groupby("event_date", sort=True):
        population_beta = _fit_population_beta(
            landed_history,
            attempted_history,
            population_beta,
            config,
        )
        baseline_probability = float(expit(population_beta))
        pending: list[tuple[str, float, float, np.ndarray | None]] = []

        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            landed = float(record["ground_landed"])
            attempted = float(record["ground_attempted"])

            lp = prior_lp.copy()
            if fighter in states:
                lp += states[fighter]
            weights = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(offense_grid, weights)

            observation_ll = None
            if attempted > 0.0:
                p = expit(population_beta + offense_grid)
                observation_ll = beta_binomial_log_likelihood(
                    landed,
                    attempted,
                    p,
                    config.ground_effectiveness_rho,
                )
                post_weights = normalize_log_weights(lp + observation_ll)
                post_mean, post_sd = weighted_mean_sd(offense_grid, post_weights)
                observed = landed / attempted
            else:
                post_mean, post_sd = pre_mean, pre_sd
                observed = np.nan

            rows.append(
                {
                    **{key: record[key] for key in KEYS},
                    "trait": "ground_striking_offense",
                    "pre_rating": pre_mean,
                    "pre_posterior_sd": pre_sd,
                    "post_rating": post_mean,
                    "post_posterior_sd": post_sd,
                    "observed_probability": observed,
                    "successes": landed,
                    "trials": attempted,
                    "population_logit": population_beta,
                    "population_baseline": baseline_probability,
                    "beta_binomial_rho": config.ground_effectiveness_rho,
                    "prior_sigma": config.ground_effectiveness_sigma,
                    "variance_multiplier": config.ground_effectiveness_variance_multiplier,
                    "sampling_enabled": False,
                }
            )
            pending.append((fighter, landed, attempted, observation_ll))

        # Same-event delayed update.
        for fighter, landed, attempted, observation_ll in pending:
            if attempted <= 0.0 or observation_ll is None:
                continue
            if fighter in states:
                states[fighter] = states[fighter] + observation_ll
                states[fighter] -= np.max(states[fighter])
            else:
                states[fighter] = observation_ll - np.max(observation_ll)
            landed_history.append(landed)
            attempted_history.append(attempted)

    history = pd.DataFrame(rows)
    return history.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
