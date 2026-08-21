"""Validated FSR V3 striking-power replay.

The promoted power trait is the selected Stage-2 sequential model:

    K_if ~ BetaBinomial(N_if, p_if, rho)
    logit(p_if) = beta + power_i
    power_i ~ Normal(0, sigma^2)

where K is knockdowns scored and N is landed significant strikes.  The
persisted V3 trait is the posterior mean of ``power_i`` on its native logit
scale.  It is attacker-only, contains no KO-win bonus and no age adjustment,
and is mean-only in Monte Carlo (validated epistemic scale c=0).

Production-state rule
---------------------
The validation study used all pre-2020 UFC evidence to initialize fighter
states, then scored 2020+ sequentially with same-date delayed updates.  This
module reproduces that validated regime exactly for 2020+ publication.
Historical rows before the validation-state cutoff are published at the neutral
population effect (0.0) rather than backfilling a future-informed fighter
state.  This preserves leakage safety for old snapshots while making the
validated 2020+ state identical in construction to the research gate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, expit, gammaln, logsumexp

from pipeline.common.paths import (
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
    MASTER_PATH,
    ROUND_STATS_PATH,
)
from pipeline.fsr_v2.physical import build_physical_observations
from pipeline.fsr_v3.config import FSRV3Config

EPS = 1e-10
TRAIT_NAME = "striking_power_v3"


def _bb_loglik_p(k: float, n: float, p, rho: float):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    comb = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    if rho <= 0.0:
        return comb + k * np.log(p) + (n - k) * np.log1p(-p)
    concentration = (1.0 - rho) / rho
    a = p * concentration
    b = (1.0 - p) * concentration
    return comb + betaln(k + a, n - k + b) - betaln(a, b)


def _fit_population_beta(train: pd.DataFrame, rho: float) -> float:
    if train.empty:
        raise ValueError("power population fit requires pre-cutoff evidence")
    k = train["kd_scored"].to_numpy(float)
    n = train["sig_landed"].to_numpy(float)

    def objective(beta: float) -> float:
        return -float(np.sum(_bb_loglik_p(k, n, expit(float(beta)), rho)))

    fit = minimize_scalar(objective, bounds=(-10.0, -1.0), method="bounded")
    if not fit.success:
        raise RuntimeError("FSR V3 power population beta fit failed")
    return float(fit.x)


def _power_grid(config: FSRV3Config) -> np.ndarray:
    return np.linspace(
        float(config.power_grid_min),
        float(config.power_grid_max),
        int(config.power_grid_points),
    )


def _prior_logweights(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        out = np.full_like(grid, -np.inf, dtype=float)
        out[int(np.argmin(np.abs(grid)))] = 0.0
        return out
    out = -0.5 * np.square(grid / float(sigma))
    return out - logsumexp(out)


def _grid_loglik(
    *,
    k: float,
    n: float,
    beta: float,
    rho: float,
    grid: np.ndarray,
) -> np.ndarray:
    return np.asarray(_bb_loglik_p(k, n, expit(beta + grid), rho), dtype=float)


def _posterior_moments(logw: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    normalized = logw - logsumexp(logw)
    weights = np.exp(normalized)
    mean = float(np.sum(weights * grid))
    var = float(np.sum(weights * np.square(grid - mean)))
    return mean, float(np.sqrt(max(var, 0.0)))


def _prepare_observations() -> pd.DataFrame:
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    obs = build_physical_observations(rounds, master).copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="raise").dt.normalize()
    obs["fight_id"] = obs["fight_id"].astype(str)
    obs["fighter_id"] = obs["fighter_id"].astype(str)
    for column in ("sig_landed", "kd_scored"):
        obs[column] = pd.to_numeric(obs[column], errors="coerce").fillna(0.0)
    obs["sig_landed"] = obs["sig_landed"].clip(lower=0.0)
    obs["kd_scored"] = np.minimum(
        obs["kd_scored"].clip(lower=0.0),
        obs["sig_landed"],
    )
    return obs.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _prepare_snapshot_keys() -> pd.DataFrame:
    base = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
        columns=["event_date", "fight_id", "fighter_id"],
    ).copy()
    base["event_date"] = pd.to_datetime(base["event_date"], errors="raise").dt.normalize()
    base["fight_id"] = base["fight_id"].astype(str)
    base["fighter_id"] = base["fighter_id"].astype(str)
    if base.duplicated(["event_date", "fight_id", "fighter_id"]).any():
        raise ValueError("duplicate frozen V2 snapshot keys while building V3 power")
    return base.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def replay_power_from_frames(
    snapshot_keys: pd.DataFrame,
    observations: pd.DataFrame,
    config: FSRV3Config | None = None,
) -> pd.DataFrame:
    """Build leakage-safe V3 power states aligned to canonical snapshot keys.

    ``snapshot_keys`` must contain event_date/fight_id/fighter_id.  ``observations``
    must contain date/fight_id/fighter_id/sig_landed/kd_scored.
    """
    config = config or FSRV3Config()
    cutoff = pd.Timestamp(config.power_train_state_cutoff).normalize()
    sigma = float(config.power_sigma)
    rho = float(config.power_rho)
    grid = _power_grid(config)
    prior = _prior_logweights(grid, sigma)

    keys = snapshot_keys.copy()
    keys["event_date"] = pd.to_datetime(keys["event_date"], errors="raise").dt.normalize()
    keys["fight_id"] = keys["fight_id"].astype(str)
    keys["fighter_id"] = keys["fighter_id"].astype(str)

    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="raise").dt.normalize()
    obs["fight_id"] = obs["fight_id"].astype(str)
    obs["fighter_id"] = obs["fighter_id"].astype(str)
    for column in ("sig_landed", "kd_scored"):
        obs[column] = pd.to_numeric(obs[column], errors="coerce").fillna(0.0)
    obs["sig_landed"] = obs["sig_landed"].clip(lower=0.0)
    obs["kd_scored"] = np.minimum(obs["kd_scored"].clip(lower=0.0), obs["sig_landed"])

    model_obs = obs[obs["sig_landed"] > 0.0].copy()
    train = model_obs[model_obs["date"] < cutoff].copy()
    beta = _fit_population_beta(train, rho)

    # Initialize exactly as the selected research model: fixed population beta,
    # Normal fighter prior, all pre-cutoff positive-strike evidence accumulated.
    evidence: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(grid))
    cache: dict[tuple[float, float], np.ndarray] = {}
    for row in train.itertuples(index=False):
        key = (float(row.kd_scored), float(row.sig_landed))
        ll = cache.get(key)
        if ll is None:
            ll = _grid_loglik(k=key[0], n=key[1], beta=beta, rho=rho, grid=grid)
            cache[key] = ll
        evidence[str(row.fighter_id)] += ll

    states: dict[str, np.ndarray] = {}
    for fighter, fighter_evidence in evidence.items():
        logw = prior + fighter_evidence
        states[fighter] = logw - logsumexp(logw)

    rows: list[dict[str, object]] = []

    # Pre-cutoff publication is deliberately neutral.  Those rows are outside
    # the validated sequential regime and must not receive future-informed
    # fighter effects from the pre-2020 initialization window.
    for row in keys[keys["event_date"] < cutoff].itertuples(index=False):
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
                "population_kd_probability": float(expit(beta)),
                "variance_multiplier": float(config.power_variance_multiplier),
                "sampling_enabled": False,
                "posterior_family": "normal_grid",
                "validated_regime": False,
            }
        )

    keys_forward = keys[keys["event_date"] >= cutoff].copy()
    obs_forward = model_obs[model_obs["date"] >= cutoff].copy()
    dates = sorted(set(keys_forward["event_date"]).union(set(obs_forward["date"])))

    for date in dates:
        day_keys = keys_forward[keys_forward["event_date"] == date]
        day_obs = obs_forward[obs_forward["date"] == date]

        day_positions: list[tuple[int, str]] = []
        for row in day_keys.itertuples(index=False):
            fighter = str(row.fighter_id)
            logw = states.get(fighter, prior)
            mean, sd = _posterior_moments(logw, grid)
            rows.append(
                {
                    "event_date": pd.Timestamp(date),
                    "fight_id": str(row.fight_id),
                    "fighter_id": fighter,
                    "trait": TRAIT_NAME,
                    "pre_rating": mean,
                    "post_rating": np.nan,
                    "pre_posterior_sd": sd,
                    "post_posterior_sd": np.nan,
                    "population_beta": beta,
                    "population_kd_probability": float(expit(beta)),
                    "variance_multiplier": float(config.power_variance_multiplier),
                    "sampling_enabled": False,
                    "posterior_family": "normal_grid",
                    "validated_regime": True,
                }
            )
            day_positions.append((len(rows) - 1, fighter))

        # Same-date delayed update: every prefight state above is frozen before
        # any evidence from this date is incorporated.
        pending: dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(grid))
        for row in day_obs.itertuples(index=False):
            key = (float(row.kd_scored), float(row.sig_landed))
            ll = cache.get(key)
            if ll is None:
                ll = _grid_loglik(k=key[0], n=key[1], beta=beta, rho=rho, grid=grid)
                cache[key] = ll
            pending[str(row.fighter_id)] += ll

        for fighter, ll in pending.items():
            current = states.get(fighter, prior)
            updated = current + ll
            states[fighter] = updated - logsumexp(updated)

        for position, fighter in day_positions:
            mean, sd = _posterior_moments(states.get(fighter, prior), grid)
            rows[position]["post_rating"] = mean
            rows[position]["post_posterior_sd"] = sd

    history = pd.DataFrame(rows)
    expected = keys[["event_date", "fight_id", "fighter_id"]].copy()
    got = history[["event_date", "fight_id", "fighter_id"]].copy()
    if len(history) != len(keys) or got.duplicated().any():
        raise RuntimeError("FSR V3 power replay did not produce one row per canonical snapshot key")
    merged = expected.merge(
        got,
        on=["event_date", "fight_id", "fighter_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("FSR V3 power replay lost canonical snapshot rows")

    return history.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def replay_power(config: FSRV3Config | None = None) -> pd.DataFrame:
    return replay_power_from_frames(
        _prepare_snapshot_keys(),
        _prepare_observations(),
        config=config,
    )
