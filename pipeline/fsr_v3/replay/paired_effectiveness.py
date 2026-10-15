"""Leakage-safe paired Beta-Binomial effectiveness replay for FSR V3.

Validated families:

Standing striking effectiveness
    distance landed / distance attempted
    rho=.035
    offense prior sigma=.30
    defense prior sigma=.30
    epistemic c=0

Takedown effectiveness
    TD landed / TD attempted
    rho=.12
    offense prior sigma=.35
    defense prior sigma=.50
    epistemic c=0

The validated mean equation is

    logit(p) = beta_population + O_attacker - D_defender

This implementation realizes that equation as a chronological assumed-density
filter: each attacker's offense posterior is updated conditional on the
opponent's prefight defense mean, and each defender's posterior is updated
conditional on the attacker's prefight offense mean.  Same-event updates are
delayed.  This provides leakage-safe fighter-fight prefight states while
preserving the validated likelihood, priors, signs, and population baseline.

Because chronological validation rejected epistemic sampling for both paired
families, only posterior means are simulator-facing; posterior SD is retained
for audit only with variance_multiplier=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
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


@dataclass(frozen=True)
class EffectivenessSpec:
    name: str
    offense_trait: str
    defense_trait: str
    landed_column: str
    attempted_column: str
    rho: float
    sigma_offense: float
    sigma_defense: float
    grid_min: float
    grid_max: float
    grid_points: int
    variance_multiplier: float


def standing_effectiveness_spec(config: FSRV3Config) -> EffectivenessSpec:
    return EffectivenessSpec(
        name="standing_striking_effectiveness",
        offense_trait="standing_striking_offense",
        defense_trait="standing_striking_defense",
        landed_column="distance_landed",
        attempted_column="distance_attempted",
        rho=config.standing_effectiveness_rho,
        sigma_offense=config.standing_effectiveness_sigma_offense,
        sigma_defense=config.standing_effectiveness_sigma_defense,
        grid_min=config.effectiveness_grid_min,
        grid_max=config.effectiveness_grid_max,
        grid_points=config.effectiveness_grid_points,
        variance_multiplier=config.standing_effectiveness_variance_multiplier,
    )


def takedown_effectiveness_spec(config: FSRV3Config) -> EffectivenessSpec:
    return EffectivenessSpec(
        name="takedown_effectiveness",
        offense_trait="takedown_offense",
        defense_trait="takedown_defense",
        landed_column="td_landed",
        attempted_column="td_attempted",
        rho=config.takedown_effectiveness_rho,
        sigma_offense=config.takedown_effectiveness_sigma_offense,
        sigma_defense=config.takedown_effectiveness_sigma_defense,
        grid_min=config.effectiveness_grid_min,
        grid_max=config.effectiveness_grid_max,
        grid_points=config.effectiveness_grid_points,
        variance_multiplier=config.takedown_effectiveness_variance_multiplier,
    )


def build_effectiveness_fighter_fights(
    spec: EffectivenessSpec,
    paired_rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    paired = build_paired_rounds() if paired_rounds is None else paired_rounds.copy()
    required = set(KEYS + [spec.landed_column, spec.attempted_column])
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"{spec.name} source missing columns: {sorted(missing)}")

    frame = (
        paired.groupby(KEYS, as_index=False)
        .agg(
            landed=(spec.landed_column, "sum"),
            attempted=(spec.attempted_column, "sum"),
        )
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame["opponent_id"] = frame["opponent_id"].astype(str)
    frame["landed"] = pd.to_numeric(frame["landed"], errors="raise").astype(float)
    frame["attempted"] = pd.to_numeric(frame["attempted"], errors="raise").astype(float)
    frame["landed"] = np.minimum(frame["landed"], frame["attempted"])
    if frame.duplicated(["event_date", "fight_id", "fighter_id"]).any():
        raise ValueError(f"duplicate {spec.name} fighter-fight observations")
    return frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _normal_prior(grid: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    return -0.5 * (grid / sigma) ** 2 - log(sigma)


def _fit_population_beta(
    successes: list[float],
    trials: list[float],
    rho: float,
    previous: float | None,
) -> float:
    if not trials or float(np.sum(trials)) <= 0.0:
        return 0.0 if previous is None else float(previous)
    y = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    keep = n > 0.0
    y, n = y[keep], n[keep]
    if len(n) == 0:
        return 0.0 if previous is None else float(previous)

    def objective(beta):
        p = expit(float(beta))
        return -float(beta_binomial_log_likelihood(y, n, p, rho).sum())

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success and previous is not None:
        return float(previous)
    return float(result.x)


def replay_paired_effectiveness(
    fights: pd.DataFrame,
    spec: EffectivenessSpec,
) -> pd.DataFrame:
    """Produce reciprocal offense/defense prefight rows for one paired family."""
    grid = np.linspace(spec.grid_min, spec.grid_max, spec.grid_points)
    offense_prior = _normal_prior(grid, spec.sigma_offense)
    defense_prior = _normal_prior(grid, spec.sigma_defense)
    offense_states: dict[str, np.ndarray] = {}
    defense_states: dict[str, np.ndarray] = {}
    population_y: list[float] = []
    population_n: list[float] = []
    beta: float | None = None
    rows: list[dict] = []

    for event_date, batch in fights.groupby("event_date", sort=True):
        beta = _fit_population_beta(population_y, population_n, spec.rho, beta)
        population_baseline = float(expit(beta))
        pending: list[
            tuple[str, str, float, float, np.ndarray | None, np.ndarray | None]
        ] = []

        for record in batch.to_dict("records"):
            attacker = str(record["fighter_id"])
            defender = str(record["opponent_id"])
            y = float(record["landed"])
            n = float(record["attempted"])

            off_lp = offense_prior.copy()
            if attacker in offense_states:
                off_lp += offense_states[attacker]
            def_lp = defense_prior.copy()
            if defender in defense_states:
                def_lp += defense_states[defender]

            off_w = normalize_log_weights(off_lp)
            def_w = normalize_log_weights(def_lp)
            off_pre_mean, off_pre_sd = weighted_mean_sd(grid, off_w)
            def_pre_mean, def_pre_sd = weighted_mean_sd(grid, def_w)

            off_ll = None
            def_ll = None
            if n > 0.0:
                off_p = expit(beta + grid - def_pre_mean)
                off_ll = beta_binomial_log_likelihood(y, n, off_p, spec.rho)
                off_post_w = normalize_log_weights(off_lp + off_ll)
                off_post_mean, off_post_sd = weighted_mean_sd(grid, off_post_w)

                def_p = expit(beta + off_pre_mean - grid)
                def_ll = beta_binomial_log_likelihood(y, n, def_p, spec.rho)
                def_post_w = normalize_log_weights(def_lp + def_ll)
                def_post_mean, def_post_sd = weighted_mean_sd(grid, def_post_w)
            else:
                off_post_mean, off_post_sd = off_pre_mean, off_pre_sd
                def_post_mean, def_post_sd = def_pre_mean, def_pre_sd

            expected_probability = float(expit(beta + off_pre_mean - def_pre_mean))

            rows.append({
                **{key: record[key] for key in KEYS},
                "trait": spec.offense_trait,
                "pre_rating": off_pre_mean,
                "pre_posterior_sd": off_pre_sd,
                "post_rating": off_post_mean,
                "post_posterior_sd": off_post_sd,
                "landed": y,
                "attempted": n,
                "population_baseline": population_baseline,
                "matchup_expected_probability": expected_probability,
                "rho": spec.rho,
                "prior_sigma": spec.sigma_offense,
                "variance_multiplier": spec.variance_multiplier,
                "sampling_enabled": False,
                "posterior_family": "normal_grid",
            })
            rows.append({
                "event_date": record["event_date"],
                "fight_id": record["fight_id"],
                "fighter_id": defender,
                "fighter_name": record["opponent_name"],
                "opponent_id": attacker,
                "opponent_name": record["fighter_name"],
                "trait": spec.defense_trait,
                "pre_rating": def_pre_mean,
                "pre_posterior_sd": def_pre_sd,
                "post_rating": def_post_mean,
                "post_posterior_sd": def_post_sd,
                "landed": y,
                "attempted": n,
                "population_baseline": population_baseline,
                "matchup_expected_probability": expected_probability,
                "rho": spec.rho,
                "prior_sigma": spec.sigma_defense,
                "variance_multiplier": spec.variance_multiplier,
                "sampling_enabled": False,
                "posterior_family": "normal_grid",
            })
            pending.append((attacker, defender, y, n, off_ll, def_ll))

        # Same-event delayed fighter and population updates.
        for attacker, defender, y, n, off_ll, def_ll in pending:
            if n <= 0.0 or off_ll is None or def_ll is None:
                continue
            if attacker in offense_states:
                offense_states[attacker] = offense_states[attacker] + off_ll
                offense_states[attacker] -= np.max(offense_states[attacker])
            else:
                offense_states[attacker] = off_ll - np.max(off_ll)
            if defender in defense_states:
                defense_states[defender] = defense_states[defender] + def_ll
                defense_states[defender] -= np.max(defense_states[defender])
            else:
                defense_states[defender] = def_ll - np.max(def_ll)
            population_y.append(y)
            population_n.append(n)

    history = pd.DataFrame(rows)
    keys = ["event_date", "fight_id", "fighter_id", "trait"]
    if history.duplicated(keys).any():
        raise ValueError(f"duplicate {spec.name} prefight trait rows")
    return history.sort_values(keys).reset_index(drop=True)


def replay_effectiveness_family(
    spec: EffectivenessSpec,
    paired_rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fights = build_effectiveness_fighter_fights(spec, paired_rounds=paired_rounds)
    return replay_paired_effectiveness(fights, spec)
