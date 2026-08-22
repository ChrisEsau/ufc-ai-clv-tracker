"""Validated FSR V3 knockdown-resistance replay.

Native model
------------
For defender i in a fighter-fight:

    K_i ~ BetaBinomial(N_i, p_i, rho)
    logit(p_i) = beta_population
                 + attacker_power_v3
                 + attacker_age_effect
                 + defender_age_effect
                 - resistance_i
    resistance_i ~ Normal(0, sigma^2)

K is knockdowns absorbed and N is significant strikes absorbed.  The selected
chronological audit parameters are rho=.005, sigma=.70, with full epistemic
variance (c=1).  The model is initialized from strictly pre-2022 evidence and
publishes the validated 2022+ sequential regime with same-date delayed updates.
Pre-2022 rows are neutral placeholders so historical publication is leakage-safe.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v2.physical import build_physical_observations
from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.fsr_v3.paths import POWER_HISTORY_PATH
from pipeline.fsr_v3.replay.math import beta_binomial_log_likelihood

TRAIT_NAME = "knockdown_resistance_v3"


def _age_years(dob, event_date) -> float:
    if dob is None or pd.isna(dob):
        return 30.0
    age = (pd.Timestamp(event_date) - pd.Timestamp(dob)).days / 365.2425
    return float(age) if age > 0 else 30.0


def _grid(config: ActiveTraitConfig) -> np.ndarray:
    return np.linspace(
        float(config.kd_resistance_grid_min),
        float(config.kd_resistance_grid_max),
        int(config.kd_resistance_grid_points),
    )


def _prior_logweights(grid: np.ndarray, sigma: float) -> np.ndarray:
    out = -0.5 * np.square(grid / float(sigma))
    return out - logsumexp(out)


def _moments(logw: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    weights = np.exp(logw - logsumexp(logw))
    mean = float(np.sum(weights * grid))
    var = float(np.sum(weights * np.square(grid - mean)))
    return mean, float(np.sqrt(max(var, 0.0)))


def _prepare_observations(config: ActiveTraitConfig) -> pd.DataFrame:
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    obs = build_physical_observations(rounds, master).copy()
    obs["event_date"] = pd.to_datetime(obs["date"], errors="raise").dt.normalize()
    obs["fight_id"] = obs["fight_id"].astype(str)
    obs["fighter_id"] = obs["fighter_id"].astype(str)
    obs["opponent_id"] = obs["opponent_id"].astype(str)

    power = pd.read_parquet(
        POWER_HISTORY_PATH,
        columns=["event_date", "fight_id", "fighter_id", "pre_rating"],
    ).copy()
    power["event_date"] = pd.to_datetime(power["event_date"], errors="raise").dt.normalize()
    power["fight_id"] = power["fight_id"].astype(str)
    power["fighter_id"] = power["fighter_id"].astype(str)
    attacker_power = power.rename(
        columns={"fighter_id": "opponent_id", "pre_rating": "attacker_power_v3"}
    )[["event_date", "fight_id", "opponent_id", "attacker_power_v3"]]
    obs = obs.merge(
        attacker_power,
        on=["event_date", "fight_id", "opponent_id"],
        how="inner",
        validate="one_to_one",
    )

    master["fight_id"] = master["fight_id"].astype(str)
    age_rows: list[dict[str, object]] = []
    for row in master.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        age_rows.extend(
            [
                {
                    "fight_id": str(row.fight_id),
                    "fighter_id": str(row.r_id),
                    "age": _age_years(row.r_dob, date),
                },
                {
                    "fight_id": str(row.fight_id),
                    "fighter_id": str(row.b_id),
                    "age": _age_years(row.b_dob, date),
                },
            ]
        )
    ages = pd.DataFrame(age_rows)
    obs = obs.merge(
        ages.rename(columns={"age": "defender_age"}),
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    obs = obs.merge(
        ages.rename(columns={"fighter_id": "opponent_id", "age": "attacker_age"}),
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    obs["k"] = pd.to_numeric(obs["kd_absorbed"], errors="coerce").fillna(0.0).clip(lower=0.0)
    obs["n"] = pd.to_numeric(obs["sig_absorbed"], errors="coerce").fillna(0.0).clip(lower=0.0)
    obs["k"] = np.minimum(obs["k"], obs["n"])
    obs["context_offset"] = (
        obs["attacker_power_v3"].astype(float)
        + float(config.kd_attacker_age_beta) * (obs["attacker_age"].astype(float) - 30.0)
        + float(config.kd_defender_age_beta) * (obs["defender_age"].astype(float) - 30.0)
    )
    return obs.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _fit_population_beta(train: pd.DataFrame, rho: float) -> float:
    train = train[train["n"] > 0.0]
    k = train["k"].to_numpy(float)
    n = train["n"].to_numpy(float)
    offset = train["context_offset"].to_numpy(float)

    def objective(beta: float) -> float:
        p = expit(float(beta) + offset)
        return -float(np.sum(beta_binomial_log_likelihood(k, n, p, rho)))

    fit = minimize_scalar(objective, bounds=(-9.0, -2.0), method="bounded")
    if not fit.success:
        raise RuntimeError("FSR V3 KD-resistance population beta fit failed")
    return float(fit.x)


def replay_kd_resistance(
    config: ActiveTraitConfig | None = None,
) -> pd.DataFrame:
    config = config or ActiveTraitConfig()
    cutoff = pd.Timestamp(config.kd_resistance_train_state_cutoff).normalize()
    rho = float(config.kd_resistance_rho)
    sigma = float(config.kd_resistance_sigma)
    grid = _grid(config)
    prior = _prior_logweights(grid, sigma)
    obs = _prepare_observations(config)
    beta = _fit_population_beta(obs[obs["event_date"] < cutoff], rho)

    positive = obs[obs["n"] > 0.0].copy()
    train = positive[positive["event_date"] < cutoff]
    evidence: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(grid))
    for row in train.itertuples(index=False):
        eta = beta + float(row.context_offset) - grid
        evidence[str(row.fighter_id)] += beta_binomial_log_likelihood(
            float(row.k), float(row.n), expit(eta), rho
        )

    states: dict[str, np.ndarray] = {}
    for fighter, ll in evidence.items():
        state = prior + ll
        states[fighter] = state - logsumexp(state)

    rows: list[dict[str, object]] = []
    # Neutral placeholders outside the validated forward regime.
    for row in obs[obs["event_date"] < cutoff].itertuples(index=False):
        rows.append(
            {
                "event_date": pd.Timestamp(row.event_date),
                "fight_id": str(row.fight_id),
                "fighter_id": str(row.fighter_id),
                "trait": TRAIT_NAME,
                "pre_rating": 0.0,
                "post_rating": 0.0,
                "pre_posterior_sd": sigma,
                "post_posterior_sd": sigma,
                "population_beta": beta,
                "rho": rho,
                "sigma": sigma,
                "variance_multiplier": float(config.kd_resistance_variance_multiplier),
                "sampling_enabled": True,
                "posterior_family": "normal_grid",
                "validated_regime": False,
            }
        )

    forward = obs[obs["event_date"] >= cutoff]
    for date, batch in forward.groupby("event_date", sort=True):
        pending: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(grid))
        positions: list[tuple[int, str]] = []

        for rec in batch.to_dict("records"):
            fighter = str(rec["fighter_id"])
            state = states.get(fighter, prior)
            mean, sd = _moments(state, grid)
            rows.append(
                {
                    "event_date": pd.Timestamp(date),
                    "fight_id": str(rec["fight_id"]),
                    "fighter_id": fighter,
                    "trait": TRAIT_NAME,
                    "pre_rating": mean,
                    "post_rating": np.nan,
                    "pre_posterior_sd": sd,
                    "post_posterior_sd": np.nan,
                    "population_beta": beta,
                    "rho": rho,
                    "sigma": sigma,
                    "variance_multiplier": float(config.kd_resistance_variance_multiplier),
                    "sampling_enabled": True,
                    "posterior_family": "normal_grid",
                    "validated_regime": True,
                }
            )
            positions.append((len(rows) - 1, fighter))
            if float(rec["n"]) > 0.0:
                eta = beta + float(rec["context_offset"]) - grid
                pending[fighter] += beta_binomial_log_likelihood(
                    float(rec["k"]), float(rec["n"]), expit(eta), rho
                )

        for fighter, ll in pending.items():
            current = states.get(fighter, prior)
            updated = current + ll
            states[fighter] = updated - logsumexp(updated)

        for position, fighter in positions:
            mean, sd = _moments(states.get(fighter, prior), grid)
            rows[position]["post_rating"] = mean
            rows[position]["post_posterior_sd"] = sd

    history = pd.DataFrame(rows).sort_values(
        ["event_date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)
    if history.duplicated(["event_date", "fight_id", "fighter_id"]).any():
        raise ValueError("duplicate V3 KD-resistance history rows")
    return history
