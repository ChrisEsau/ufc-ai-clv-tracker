"""Validated FSR V3 ground tendency and suppression replay.

Ground tendency
    Y ~ NB2(b + own_CTRL/900 * q_fighter, alpha)
    Gamma shrinkage on q with K=90 own-control seconds.
    Posterior mean only; epistemic variance multiplier c=0.

Ground suppression
    Y ~ NB2(b + s_defender * attacker_slope, alpha)
    attacker_slope = own_CTRL/900 * q_attacker_prefight
    Gamma prior shape=2 around the rolling population multiplier.
    The global burst is NOT suppressed.
    Posterior mean only; c=0.

Ground effectiveness lives in ``ground_effectiveness.py`` because its validated
observation model is Beta-Binomial rather than NB2.

All event-date updates are delayed until every row on that date has been
scored, preserving leakage-safe same-event prefight state.
"""

from __future__ import annotations

from math import lgamma, log

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.math import (
    nb2_log_likelihood,
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


def build_ground_fighter_fights(
    paired_rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate raw paired round stats to one reciprocal row per fighter-fight."""
    paired = build_paired_rounds() if paired_rounds is None else paired_rounds.copy()
    required = set(KEYS + ["ground_landed", "ground_attempted", "ctrl_sec"])
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"ground replay source missing columns: {sorted(missing)}")

    frame = (
        paired.groupby(KEYS, as_index=False)
        .agg(
            ground_landed=("ground_landed", "sum"),
            ground_attempted=("ground_attempted", "sum"),
            own_control_seconds=("ctrl_sec", "sum"),
        )
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame["opponent_id"] = frame["opponent_id"].astype(str)
    for column in ("ground_landed", "ground_attempted", "own_control_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame["ground_landed"] = np.minimum(frame["ground_landed"], frame["ground_attempted"])

    duplicate = frame.duplicated(["event_date", "fight_id", "fighter_id"])
    if duplicate.any():
        sample = frame.loc[duplicate, KEYS].head().to_dict("records")
        raise ValueError(f"duplicate ground fighter-fight observations: {sample}")

    return frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _log_gamma_prior(grid: np.ndarray, mean: float, shape: float) -> np.ndarray:
    mean = max(float(mean), 1e-6)
    shape = max(float(shape), 1e-6)
    rate = shape / mean
    return (
        shape * log(rate)
        - lgamma(shape)
        + (shape - 1.0) * np.log(grid)
        - rate * grid
    )


def _fit_tendency_population(
    history_y: list[float],
    history_e: list[float],
    previous: tuple[float, float, float] | None,
    config: FSRV3Config,
) -> tuple[float, float, float]:
    if not history_y:
        return (
            config.ground_tendency_initial_burst,
            config.ground_tendency_initial_population_rate_15m,
            config.ground_tendency_initial_alpha,
        )

    y = np.asarray(history_y, dtype=float)
    e = np.asarray(history_e, dtype=float)

    if previous is None:
        burst0 = float(np.clip(np.median(y), 0.05, 5.0))
        residual = max(float(y.sum() - burst0 * len(y)), 0.1)
        q0 = max(residual / max(e.sum(), 1.0) * 900.0, 1.0)
        alpha0 = 1.0
    else:
        burst0, q0, alpha0 = previous

    def objective(theta):
        burst, q_pop, alpha = np.exp(theta)
        mu = burst + e / 900.0 * q_pop
        return -float(nb2_log_likelihood(y, mu, alpha).sum())

    result = minimize(
        objective,
        np.log([max(burst0, 1e-4), max(q0, 1e-4), max(alpha0, 1e-4)]),
        method="L-BFGS-B",
        bounds=[(-8.0, 5.0), (-5.0, 8.0), (-8.0, 6.0)],
    )
    if not result.success and previous is not None:
        return previous
    burst, q_pop, alpha = np.exp(result.x)
    return float(burst), float(q_pop), float(alpha)


def replay_ground_tendency(
    fights: pd.DataFrame,
    config: FSRV3Config | None = None,
) -> pd.DataFrame:
    config = config or FSRV3Config()
    q_grid = np.linspace(
        config.ground_tendency_q_grid_min,
        config.ground_tendency_q_grid_max,
        config.ground_tendency_q_grid_points,
    )

    states: dict[str, np.ndarray] = {}
    population_y: list[float] = []
    population_e: list[float] = []
    population_parameters: tuple[float, float, float] | None = None
    rows: list[dict] = []

    for _, batch in fights.groupby("event_date", sort=True):
        population_parameters = _fit_tendency_population(
            population_y,
            population_e,
            population_parameters,
            config,
        )
        burst, q_pop, alpha = population_parameters
        prior_shape = max(q_pop * config.ground_tendency_prior_seconds / 900.0, 1e-6)
        prior_lp = _log_gamma_prior(q_grid, q_pop, prior_shape)
        pending: list[tuple[str, float, float, np.ndarray | None]] = []

        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            y = float(record["ground_attempted"])
            exposure = float(record["own_control_seconds"])
            lp = prior_lp.copy()
            if fighter in states:
                lp += states[fighter]
            weights = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(q_grid, weights)

            observation_ll = None
            if exposure > 0.0:
                mu = burst + exposure / 900.0 * q_grid
                observation_ll = nb2_log_likelihood(y, mu, alpha)
                post_weights = normalize_log_weights(lp + observation_ll)
                post_mean, post_sd = weighted_mean_sd(q_grid, post_weights)
                observed = y / exposure * 900.0
            else:
                post_mean, post_sd = pre_mean, pre_sd
                observed = np.nan

            rows.append(
                {
                    **{key: record[key] for key in KEYS},
                    "trait": "ground_striking_tendency",
                    "pre_rating": pre_mean,
                    "pre_posterior_sd": pre_sd,
                    "post_rating": post_mean,
                    "post_posterior_sd": post_sd,
                    "observed_rate_15m_own_control": observed,
                    "numerator": y,
                    "denominator": exposure,
                    "population_burst": burst,
                    "population_rate_15m": q_pop,
                    "observation_alpha": alpha,
                    "prior_seconds": config.ground_tendency_prior_seconds,
                    "variance_multiplier": config.ground_tendency_variance_multiplier,
                    "sampling_enabled": False,
                }
            )
            pending.append((fighter, y, exposure, observation_ll))

        # Same-event delayed update.
        for fighter, y, exposure, observation_ll in pending:
            if exposure <= 0.0 or observation_ll is None:
                continue
            if fighter in states:
                states[fighter] = states[fighter] + observation_ll
                states[fighter] -= np.max(states[fighter])
            else:
                states[fighter] = observation_ll - np.max(observation_ll)
            population_y.append(y)
            population_e.append(exposure)

    history = pd.DataFrame(rows)
    return history.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _fit_suppression_population(
    history_y: list[float],
    history_burst: list[float],
    history_slope: list[float],
    previous: tuple[float, float] | None,
    config: FSRV3Config,
) -> tuple[float, float]:
    if not history_y:
        return (
            config.ground_suppression_initial_population,
            config.ground_suppression_initial_alpha,
        )

    y = np.asarray(history_y, dtype=float)
    burst = np.asarray(history_burst, dtype=float)
    slope = np.asarray(history_slope, dtype=float)
    if previous is None:
        s0, alpha0 = 1.0, 1.0
    else:
        s0, alpha0 = previous

    def objective(theta):
        s_pop, alpha = np.exp(theta)
        mu = burst + s_pop * slope
        return -float(nb2_log_likelihood(y, mu, alpha).sum())

    result = minimize(
        objective,
        np.log([max(s0, 1e-4), max(alpha0, 1e-4)]),
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (-8.0, 6.0)],
    )
    if not result.success and previous is not None:
        return previous
    s_pop, alpha = np.exp(result.x)
    return float(s_pop), float(alpha)


def replay_ground_suppression(
    tendency_history: pd.DataFrame,
    config: FSRV3Config | None = None,
) -> pd.DataFrame:
    """Replay defender suppression using leakage-safe attacker tendency baselines."""
    config = config or FSRV3Config()
    s_grid = np.linspace(
        config.ground_suppression_grid_min,
        config.ground_suppression_grid_max,
        config.ground_suppression_grid_points,
    )
    source = tendency_history.copy()
    source["attacker_slope"] = (
        source["denominator"].astype(float)
        / 900.0
        * source["pre_rating"].astype(float)
    )

    states: dict[str, np.ndarray] = {}
    population_y: list[float] = []
    population_burst: list[float] = []
    population_slope: list[float] = []
    population_parameters: tuple[float, float] | None = None
    rows: list[dict] = []

    for _, batch in source.groupby("event_date", sort=True):
        population_parameters = _fit_suppression_population(
            population_y,
            population_burst,
            population_slope,
            population_parameters,
            config,
        )
        s_pop, alpha = population_parameters
        prior_lp = _log_gamma_prior(
            s_grid,
            s_pop,
            config.ground_suppression_prior_shape,
        )
        pending: list[tuple[str, float, float, float, np.ndarray | None]] = []

        for record in batch.to_dict("records"):
            # The source row is the attacker. Suppression belongs to the opponent.
            defender = str(record["opponent_id"])
            y = float(record["numerator"])
            burst = float(record["population_burst"])
            slope = float(record["attacker_slope"])
            lp = prior_lp.copy()
            if defender in states:
                lp += states[defender]
            weights = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(s_grid, weights)

            observation_ll = None
            if slope > 0.0:
                mu = burst + s_grid * slope
                observation_ll = nb2_log_likelihood(y, mu, alpha)
                post_weights = normalize_log_weights(lp + observation_ll)
                post_mean, post_sd = weighted_mean_sd(s_grid, post_weights)
                observed = max((y - burst) / slope, 0.0)
            else:
                post_mean, post_sd = pre_mean, pre_sd
                observed = np.nan

            rows.append(
                {
                    "event_date": record["event_date"],
                    "fight_id": record["fight_id"],
                    "fighter_id": defender,
                    "fighter_name": record["opponent_name"],
                    "opponent_id": str(record["fighter_id"]),
                    "opponent_name": record["fighter_name"],
                    "trait": "ground_striking_suppression",
                    "pre_rating": pre_mean,
                    "pre_posterior_sd": pre_sd,
                    "post_rating": post_mean,
                    "post_posterior_sd": post_sd,
                    "observed_multiplier": observed,
                    "opponent_actual_attempts": y,
                    "opponent_expected_slope_attempts": slope,
                    "population_burst": burst,
                    "population_multiplier": s_pop,
                    "observation_alpha": alpha,
                    "prior_shape": config.ground_suppression_prior_shape,
                    "variance_multiplier": config.ground_suppression_variance_multiplier,
                    "sampling_enabled": False,
                }
            )
            pending.append((defender, y, burst, slope, observation_ll))

        # Same-event delayed update.
        for defender, y, burst, slope, observation_ll in pending:
            if slope <= 0.0 or observation_ll is None:
                continue
            if defender in states:
                states[defender] = states[defender] + observation_ll
                states[defender] -= np.max(states[defender])
            else:
                states[defender] = observation_ll - np.max(observation_ll)
            population_y.append(y)
            population_burst.append(burst)
            population_slope.append(slope)

    history = pd.DataFrame(rows)
    duplicate = history.duplicated(["event_date", "fight_id", "fighter_id"])
    if duplicate.any():
        sample = history.loc[duplicate, KEYS].head().to_dict("records")
        raise ValueError(f"duplicate ground suppression snapshots: {sample}")
    return history.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
