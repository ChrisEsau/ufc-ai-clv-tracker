"""Validated FSR V3 standing/takedown tendency and suppression replays.

The two rate families use the same statistical architecture but retain their
trait-native parameters.

Tendency
--------
    Y ~ NB2(E/900 * q_fighter, alpha)

    q_fighter has a Gamma population prior whose mean is the leakage-safe
    prior-date population rate and whose evidence strength is K seconds.

Suppression
-----------
    expected opponent attempts = E/900 * q_attacker_prefight
    Y ~ NB2(expected * s_defender, alpha)

    s < 1 suppresses opponent event generation.

Same-event updates are delayed.  The NB2 alpha is observation/fight noise and
is never propagated as epistemic FSR uncertainty.  Posterior SD of the latent
fighter trait is epistemic and is published for the families where c=1 was
validated.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class RateFamilySpec:
    name: str
    tendency_trait: str
    suppression_trait: str
    numerator_column: str
    exposure_column: str
    tendency_prior_seconds: float
    tendency_initial_population_rate_15m: float
    tendency_initial_alpha: float
    tendency_grid_min: float
    tendency_grid_max: float
    tendency_grid_points: int
    tendency_variance_multiplier: float
    suppression_prior_shape: float
    suppression_initial_population: float
    suppression_initial_alpha: float
    suppression_grid_min: float
    suppression_grid_max: float
    suppression_grid_points: int
    suppression_variance_multiplier: float


def standing_spec(config: FSRV3Config) -> RateFamilySpec:
    return RateFamilySpec(
        name="standing_striking",
        tendency_trait="standing_striking_tendency",
        suppression_trait="standing_striking_suppression",
        numerator_column="distance_attempted",
        exposure_column="standing_exposure_seconds",
        tendency_prior_seconds=config.standing_tendency_prior_seconds,
        tendency_initial_population_rate_15m=config.standing_tendency_initial_population_rate_15m,
        tendency_initial_alpha=config.standing_tendency_initial_alpha,
        tendency_grid_min=config.standing_tendency_q_grid_min,
        tendency_grid_max=config.standing_tendency_q_grid_max,
        tendency_grid_points=config.standing_tendency_q_grid_points,
        tendency_variance_multiplier=config.standing_tendency_variance_multiplier,
        suppression_prior_shape=config.standing_suppression_prior_shape,
        suppression_initial_population=config.standing_suppression_initial_population,
        suppression_initial_alpha=config.standing_suppression_initial_alpha,
        suppression_grid_min=config.standing_suppression_grid_min,
        suppression_grid_max=config.standing_suppression_grid_max,
        suppression_grid_points=config.standing_suppression_grid_points,
        suppression_variance_multiplier=config.standing_suppression_variance_multiplier,
    )


def takedown_spec(config: FSRV3Config) -> RateFamilySpec:
    return RateFamilySpec(
        name="takedown",
        tendency_trait="takedown_tendency",
        suppression_trait="takedown_suppression",
        numerator_column="td_attempted",
        exposure_column="td_tendency_exposure_seconds",
        tendency_prior_seconds=config.takedown_tendency_prior_seconds,
        tendency_initial_population_rate_15m=config.takedown_tendency_initial_population_rate_15m,
        tendency_initial_alpha=config.takedown_tendency_initial_alpha,
        tendency_grid_min=config.takedown_tendency_q_grid_min,
        tendency_grid_max=config.takedown_tendency_q_grid_max,
        tendency_grid_points=config.takedown_tendency_q_grid_points,
        tendency_variance_multiplier=config.takedown_tendency_variance_multiplier,
        suppression_prior_shape=config.takedown_suppression_prior_shape,
        suppression_initial_population=config.takedown_suppression_initial_population,
        suppression_initial_alpha=config.takedown_suppression_initial_alpha,
        suppression_grid_min=config.takedown_suppression_grid_min,
        suppression_grid_max=config.takedown_suppression_grid_max,
        suppression_grid_points=config.takedown_suppression_grid_points,
        suppression_variance_multiplier=config.takedown_suppression_variance_multiplier,
    )


def build_rate_fighter_fights(
    spec: RateFamilySpec,
    paired_rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate raw paired rounds to one leakage-replay row per fighter-fight."""
    paired = build_paired_rounds() if paired_rounds is None else paired_rounds.copy()
    required = set(KEYS + [spec.numerator_column, spec.exposure_column])
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"{spec.name} source missing columns: {sorted(missing)}")

    frame = (
        paired.groupby(KEYS, as_index=False)
        .agg(
            numerator=(spec.numerator_column, "sum"),
            exposure_seconds=(spec.exposure_column, "sum"),
        )
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame["opponent_id"] = frame["opponent_id"].astype(str)
    frame["numerator"] = pd.to_numeric(frame["numerator"], errors="raise").astype(float)
    frame["exposure_seconds"] = pd.to_numeric(frame["exposure_seconds"], errors="raise").astype(float)
    frame["exposure_seconds"] = frame["exposure_seconds"].clip(lower=0.0)

    duplicate = frame.duplicated(["event_date", "fight_id", "fighter_id"])
    if duplicate.any():
        raise ValueError(f"duplicate {spec.name} fighter-fight observations")
    return frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _log_gamma_prior(grid: np.ndarray, mean: float, shape: float) -> np.ndarray:
    mean = max(float(mean), 1e-9)
    shape = max(float(shape), 1e-9)
    rate = shape / mean
    return (
        shape * log(rate)
        - lgamma(shape)
        + (shape - 1.0) * np.log(grid)
        - rate * grid
    )


def _fit_tendency_population(
    y_values: list[float],
    e_values: list[float],
    previous: tuple[float, float] | None,
    spec: RateFamilySpec,
) -> tuple[float, float]:
    if not y_values:
        return spec.tendency_initial_population_rate_15m, spec.tendency_initial_alpha

    y = np.asarray(y_values, dtype=float)
    e = np.asarray(e_values, dtype=float)
    keep = e > 0.0
    y, e = y[keep], e[keep]
    if len(y) == 0:
        return spec.tendency_initial_population_rate_15m, spec.tendency_initial_alpha

    if previous is None:
        q0 = max(float(y.sum()) / max(float(e.sum()), 1.0) * 900.0, 1e-4)
        alpha0 = spec.tendency_initial_alpha
    else:
        q0, alpha0 = previous

    def objective(theta):
        q_pop, alpha = np.exp(theta)
        mu = e / 900.0 * q_pop
        return -float(nb2_log_likelihood(y, mu, alpha).sum())

    result = minimize(
        objective,
        np.log([max(q0, 1e-4), max(alpha0, 1e-4)]),
        method="L-BFGS-B",
        bounds=[(-8.0, 9.0), (-10.0, 6.0)],
    )
    if not result.success and previous is not None:
        return previous
    q_pop, alpha = np.exp(result.x)
    return float(q_pop), float(alpha)


def replay_tendency(
    fights: pd.DataFrame,
    spec: RateFamilySpec,
) -> pd.DataFrame:
    """Replay a validated positive event-rate tendency family."""
    q_grid = np.linspace(
        spec.tendency_grid_min,
        spec.tendency_grid_max,
        spec.tendency_grid_points,
    )
    states: dict[str, np.ndarray] = {}
    population_y: list[float] = []
    population_e: list[float] = []
    population_parameters: tuple[float, float] | None = None
    rows: list[dict] = []

    for event_date, batch in fights.groupby("event_date", sort=True):
        population_parameters = _fit_tendency_population(
            population_y,
            population_e,
            population_parameters,
            spec,
        )
        q_pop, alpha = population_parameters
        prior_shape = max(q_pop * spec.tendency_prior_seconds / 900.0, 1e-9)
        prior_lp = _log_gamma_prior(q_grid, q_pop, prior_shape)
        pending: list[tuple[str, float, float, np.ndarray | None]] = []

        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            y = float(record["numerator"])
            exposure = float(record["exposure_seconds"])
            lp = prior_lp.copy()
            if fighter in states:
                lp += states[fighter]
            weights = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(q_grid, weights)

            observation_ll = None
            if exposure > 0.0:
                mu = exposure / 900.0 * q_grid
                observation_ll = nb2_log_likelihood(y, mu, alpha)
                post_weights = normalize_log_weights(lp + observation_ll)
                post_mean, post_sd = weighted_mean_sd(q_grid, post_weights)
                observed = y / exposure * 900.0
            else:
                post_mean, post_sd = pre_mean, pre_sd
                observed = np.nan

            rows.append({
                **{key: record[key] for key in KEYS},
                "trait": spec.tendency_trait,
                "pre_rating": pre_mean,
                "pre_posterior_sd": pre_sd,
                "post_rating": post_mean,
                "post_posterior_sd": post_sd,
                "observed_rate_15m": observed,
                "numerator": y,
                "denominator": exposure,
                "population_rate_15m": q_pop,
                "observation_alpha": alpha,
                "prior_seconds": spec.tendency_prior_seconds,
                "variance_multiplier": spec.tendency_variance_multiplier,
                "sampling_enabled": bool(spec.tendency_variance_multiplier > 0.0),
                "posterior_family": "positive_grid",
            })
            pending.append((fighter, y, exposure, observation_ll))

        # Same-event delayed updates.
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

    return pd.DataFrame(rows).sort_values(
        ["event_date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)


def _fit_suppression_population(
    y_values: list[float],
    expected_values: list[float],
    previous: tuple[float, float] | None,
    spec: RateFamilySpec,
) -> tuple[float, float]:
    if not y_values:
        return spec.suppression_initial_population, spec.suppression_initial_alpha

    y = np.asarray(y_values, dtype=float)
    expected = np.asarray(expected_values, dtype=float)
    keep = expected > 0.0
    y, expected = y[keep], expected[keep]
    if len(y) == 0:
        return spec.suppression_initial_population, spec.suppression_initial_alpha

    if previous is None:
        s0, alpha0 = spec.suppression_initial_population, spec.suppression_initial_alpha
    else:
        s0, alpha0 = previous

    def objective(theta):
        s_pop, alpha = np.exp(theta)
        mu = expected * s_pop
        return -float(nb2_log_likelihood(y, mu, alpha).sum())

    result = minimize(
        objective,
        np.log([max(s0, 1e-4), max(alpha0, 1e-4)]),
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0), (-10.0, 6.0)],
    )
    if not result.success and previous is not None:
        return previous
    s_pop, alpha = np.exp(result.x)
    return float(s_pop), float(alpha)


def replay_suppression(
    tendency_history: pd.DataFrame,
    spec: RateFamilySpec,
) -> pd.DataFrame:
    """Replay multiplicative defender suppression from attacker prefight norms."""
    s_grid = np.linspace(
        spec.suppression_grid_min,
        spec.suppression_grid_max,
        spec.suppression_grid_points,
    )
    source = tendency_history.copy()
    source["expected_attempts"] = (
        source["denominator"].astype(float)
        / 900.0
        * source["pre_rating"].astype(float)
    )

    states: dict[str, np.ndarray] = {}
    population_y: list[float] = []
    population_expected: list[float] = []
    population_parameters: tuple[float, float] | None = None
    rows: list[dict] = []

    for event_date, batch in source.groupby("event_date", sort=True):
        population_parameters = _fit_suppression_population(
            population_y,
            population_expected,
            population_parameters,
            spec,
        )
        s_pop, alpha = population_parameters
        prior_lp = _log_gamma_prior(s_grid, s_pop, spec.suppression_prior_shape)
        pending: list[tuple[str, float, float, np.ndarray | None]] = []

        for record in batch.to_dict("records"):
            # Source row belongs to the attacker; suppression belongs to opponent.
            defender = str(record["opponent_id"])
            y = float(record["numerator"])
            expected = float(record["expected_attempts"])
            lp = prior_lp.copy()
            if defender in states:
                lp += states[defender]
            weights = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(s_grid, weights)

            observation_ll = None
            if expected > 0.0:
                mu = expected * s_grid
                observation_ll = nb2_log_likelihood(y, mu, alpha)
                post_weights = normalize_log_weights(lp + observation_ll)
                post_mean, post_sd = weighted_mean_sd(s_grid, post_weights)
                observed = y / expected
            else:
                post_mean, post_sd = pre_mean, pre_sd
                observed = np.nan

            rows.append({
                "event_date": record["event_date"],
                "fight_id": record["fight_id"],
                "fighter_id": defender,
                "fighter_name": record["opponent_name"],
                "opponent_id": str(record["fighter_id"]),
                "opponent_name": record["fighter_name"],
                "trait": spec.suppression_trait,
                "pre_rating": pre_mean,
                "pre_posterior_sd": pre_sd,
                "post_rating": post_mean,
                "post_posterior_sd": post_sd,
                "observed_multiplier": observed,
                "opponent_actual_attempts": y,
                "opponent_expected_attempts": expected,
                "population_multiplier": s_pop,
                "observation_alpha": alpha,
                "prior_shape": spec.suppression_prior_shape,
                "variance_multiplier": spec.suppression_variance_multiplier,
                "sampling_enabled": bool(spec.suppression_variance_multiplier > 0.0),
                "posterior_family": "positive_grid",
            })
            pending.append((defender, y, expected, observation_ll))

        # Same-event delayed updates.
        for defender, y, expected, observation_ll in pending:
            if expected <= 0.0 or observation_ll is None:
                continue
            if defender in states:
                states[defender] = states[defender] + observation_ll
                states[defender] -= np.max(states[defender])
            else:
                states[defender] = observation_ll - np.max(observation_ll)
            population_y.append(y)
            population_expected.append(expected)

    history = pd.DataFrame(rows)
    if history.duplicated(["event_date", "fight_id", "fighter_id"]).any():
        raise ValueError(f"duplicate {spec.suppression_trait} snapshots")
    return history.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def replay_rate_family(
    spec: RateFamilySpec,
    paired_rounds: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fights = build_rate_fighter_fights(spec, paired_rounds=paired_rounds)
    tendency = replay_tendency(fights, spec)
    suppression = replay_suppression(tendency, spec)
    return tendency, suppression
