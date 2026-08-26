from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.replay.engine import aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay.ground import (
    _fit_suppression_population as ground_fit_suppression_population,
    _fit_tendency_population as ground_fit_tendency_population,
    _log_gamma_prior as ground_log_gamma_prior,
    build_ground_fighter_fights,
)
from pipeline.fsr_v3.replay.ground_effectiveness import (
    _fit_population_beta as ground_fit_population_beta,
    _normal_prior as ground_normal_prior,
)
from pipeline.fsr_v3.replay.kd_resistance import (
    _fit_population_beta as kd_fit_population_beta,
    _grid as kd_grid,
    _prepare_observations as kd_prepare_observations,
    _prior_logweights as kd_prior_logweights,
)
from pipeline.fsr_v3.replay.math import (
    beta_binomial_log_likelihood,
    nb2_log_likelihood,
    normalize_log_weights,
    weighted_mean_sd,
)
from pipeline.fsr_v3.replay.paired_effectiveness import (
    _fit_population_beta as paired_fit_population_beta,
    _normal_prior as paired_normal_prior,
    build_effectiveness_fighter_fights,
    standing_effectiveness_spec,
    takedown_effectiveness_spec,
)
from pipeline.fsr_v3.replay.power import (
    _fit_population_beta as power_fit_population_beta,
    _grid_loglik as power_grid_loglik,
    _power_grid,
    _prepare_observations as power_prepare_observations,
    _prior_logweights as power_prior_logweights,
)
from pipeline.fsr_v3.replay.rate_families import (
    _fit_suppression_population as rate_fit_suppression_population,
    _fit_tendency_population as rate_fit_tendency_population,
    _log_gamma_prior as rate_log_gamma_prior,
    build_rate_fighter_fights,
    standing_spec,
    takedown_spec,
)
from pipeline.simulation.event_clock_mc_v2.calibration.runner import run
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture


WINDOW = 3
EWM_DECAY = 0.50
EWM_CANONICAL_BLEND = 0.50
COHORT_SIZE = 150
PATHS_PER_FIGHT = 150
SHADOW_EVENT_NAME = "FSR_RECENCY_COHORT_SHADOW_20260826"
OUTDIR = Path("data/diagnostics/fsr_recency_cohort")


class EvidenceBank:
    def __init__(self, mode: str):
        if mode not in {"last3", "ewm"}:
            raise ValueError(mode)
        self.mode = mode
        self.last3: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.ewm: dict[str, np.ndarray] = {}

    def evidence(self, key: str, size: int) -> np.ndarray:
        if self.mode == "last3":
            items = self.last3[str(key)]
            if not items:
                return np.zeros(size, dtype=float)
            return np.sum(np.stack(list(items), axis=0), axis=0)
        return self.ewm.get(str(key), np.zeros(size, dtype=float))

    def add(self, key: str, ll: np.ndarray) -> None:
        x = np.asarray(ll, dtype=float)
        x = x - np.max(x)
        key = str(key)
        if self.mode == "last3":
            self.last3[key].append(x)
        else:
            old = self.ewm.get(key)
            self.ewm[key] = x if old is None else EWM_DECAY * old + x
            self.ewm[key] -= np.max(self.ewm[key])


class ScalarBank:
    """Fight-level sufficient statistics for escape recency."""
    def __init__(self, mode: str):
        self.mode = mode
        self.last3: dict[str, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.ewm: dict[str, tuple[float, float]] = {}

    def totals(self, key: str) -> tuple[float, float]:
        key = str(key)
        if self.mode == "last3":
            values = self.last3[key]
            return (sum(x for x, _ in values), sum(y for _, y in values))
        return self.ewm.get(key, (0.0, 0.0))

    def add(self, key: str, numerator: float, denominator: float) -> None:
        key = str(key)
        if self.mode == "last3":
            self.last3[key].append((float(numerator), float(denominator)))
        else:
            n0, d0 = self.ewm.get(key, (0.0, 0.0))
            self.ewm[key] = (EWM_DECAY * n0 + float(numerator), EWM_DECAY * d0 + float(denominator))


def rolling_rate(fights: pd.DataFrame, spec, mode: str) -> pd.DataFrame:
    grid = np.linspace(spec.tendency_grid_min, spec.tendency_grid_max, spec.tendency_grid_points)
    bank = EvidenceBank(mode)
    pop_y: list[float] = []
    pop_e: list[float] = []
    params = None
    rows = []
    for event_date, batch in fights.groupby("event_date", sort=True):
        params = rate_fit_tendency_population(pop_y, pop_e, params, spec)
        q_pop, alpha = params
        shape = max(q_pop * spec.tendency_prior_seconds / 900.0, 1e-9)
        prior = rate_log_gamma_prior(grid, q_pop, shape)
        pending = []
        for rec in batch.to_dict("records"):
            fid = str(rec["fighter_id"])
            y = float(rec["numerator"])
            e = float(rec["exposure_seconds"])
            lp = prior + bank.evidence(fid, len(grid))
            w = normalize_log_weights(lp)
            pre, sd = weighted_mean_sd(grid, w)
            ll = None
            post = pre
            post_sd = sd
            if e > 0:
                ll = nb2_log_likelihood(y, e / 900.0 * grid, alpha)
                post, post_sd = weighted_mean_sd(grid, normalize_log_weights(lp + ll))
            rows.append({**rec, "trait": spec.tendency_trait, "pre_rating": pre,
                         "pre_posterior_sd": sd, "post_rating": post,
                         "post_posterior_sd": post_sd, "population_rate_15m": q_pop,
                         "observation_alpha": alpha, "denominator": e, "numerator": y})
            pending.append((fid, y, e, ll))
        for fid, y, e, ll in pending:
            if e > 0 and ll is not None:
                bank.add(fid, ll)
                pop_y.append(y); pop_e.append(e)
    return pd.DataFrame(rows)


def rolling_rate_suppression(tendency: pd.DataFrame, spec, mode: str) -> pd.DataFrame:
    grid = np.linspace(spec.suppression_grid_min, spec.suppression_grid_max, spec.suppression_grid_points)
    source = tendency.copy()
    source["expected_attempts"] = source["denominator"].astype(float) / 900.0 * source["pre_rating"].astype(float)
    bank = EvidenceBank(mode)
    pop_y: list[float] = []
    pop_expected: list[float] = []
    params = None
    rows = []
    for event_date, batch in source.groupby("event_date", sort=True):
        params = rate_fit_suppression_population(pop_y, pop_expected, params, spec)
        s_pop, alpha = params
        prior = rate_log_gamma_prior(grid, s_pop, spec.suppression_prior_shape)
        pending = []
        for rec in batch.to_dict("records"):
            defender = str(rec["opponent_id"])
            y = float(rec["numerator"])
            expected = float(rec["expected_attempts"])
            lp = prior + bank.evidence(defender, len(grid))
            w = normalize_log_weights(lp)
            pre, sd = weighted_mean_sd(grid, w)
            ll = None
            post = pre; post_sd = sd
            if expected > 0:
                ll = nb2_log_likelihood(y, expected * grid, alpha)
                post, post_sd = weighted_mean_sd(grid, normalize_log_weights(lp + ll))
            rows.append({"event_date": rec["event_date"], "fight_id": str(rec["fight_id"]),
                         "fighter_id": defender, "fighter_name": rec["opponent_name"],
                         "opponent_id": str(rec["fighter_id"]), "opponent_name": rec["fighter_name"],
                         "trait": spec.suppression_trait, "pre_rating": pre,
                         "pre_posterior_sd": sd, "post_rating": post,
                         "post_posterior_sd": post_sd, "population_multiplier": s_pop,
                         "observation_alpha": alpha})
            pending.append((defender, y, expected, ll))
        for defender, y, expected, ll in pending:
            if expected > 0 and ll is not None:
                bank.add(defender, ll)
                pop_y.append(y); pop_expected.append(expected)
    return pd.DataFrame(rows)


def rolling_paired(fights: pd.DataFrame, spec, mode: str) -> pd.DataFrame:
    grid = np.linspace(spec.grid_min, spec.grid_max, spec.grid_points)
    off_prior = paired_normal_prior(grid, spec.sigma_offense)
    def_prior = paired_normal_prior(grid, spec.sigma_defense)
    off_bank, def_bank = EvidenceBank(mode), EvidenceBank(mode)
    pop_y: list[float] = []
    pop_n: list[float] = []
    beta = None
    rows = []
    for event_date, batch in fights.groupby("event_date", sort=True):
        beta = paired_fit_population_beta(pop_y, pop_n, spec.rho, beta)
        baseline = float(expit(beta))
        pending = []
        for rec in batch.to_dict("records"):
            attacker, defender = str(rec["fighter_id"]), str(rec["opponent_id"])
            y, n = float(rec["landed"]), float(rec["attempted"])
            off_lp = off_prior + off_bank.evidence(attacker, len(grid))
            def_lp = def_prior + def_bank.evidence(defender, len(grid))
            off_pre, off_sd = weighted_mean_sd(grid, normalize_log_weights(off_lp))
            def_pre, def_sd = weighted_mean_sd(grid, normalize_log_weights(def_lp))
            off_ll = def_ll = None
            off_post, def_post = off_pre, def_pre
            if n > 0:
                off_ll = beta_binomial_log_likelihood(y, n, expit(beta + grid - def_pre), spec.rho)
                def_ll = beta_binomial_log_likelihood(y, n, expit(beta + off_pre - grid), spec.rho)
                off_post, _ = weighted_mean_sd(grid, normalize_log_weights(off_lp + off_ll))
                def_post, _ = weighted_mean_sd(grid, normalize_log_weights(def_lp + def_ll))
            common = {"event_date": rec["event_date"], "fight_id": str(rec["fight_id"]),
                      "population_baseline": baseline}
            rows.append({**common, "fighter_id": attacker, "trait": spec.offense_trait,
                         "pre_rating": off_pre, "pre_posterior_sd": off_sd, "post_rating": off_post})
            rows.append({**common, "fighter_id": defender, "trait": spec.defense_trait,
                         "pre_rating": def_pre, "pre_posterior_sd": def_sd, "post_rating": def_post})
            pending.append((attacker, defender, y, n, off_ll, def_ll))
        for attacker, defender, y, n, off_ll, def_ll in pending:
            if n > 0 and off_ll is not None and def_ll is not None:
                off_bank.add(attacker, off_ll); def_bank.add(defender, def_ll)
                pop_y.append(y); pop_n.append(n)
    return pd.DataFrame(rows)


def rolling_ground_tendency(fights: pd.DataFrame, cfg: FSRV3Config, mode: str) -> pd.DataFrame:
    grid = np.linspace(cfg.ground_tendency_q_grid_min, cfg.ground_tendency_q_grid_max, cfg.ground_tendency_q_grid_points)
    bank = EvidenceBank(mode)
    pop_y: list[float] = []; pop_e: list[float] = []; params = None; rows = []
    for _, batch in fights.groupby("event_date", sort=True):
        params = ground_fit_tendency_population(pop_y, pop_e, params, cfg)
        burst, q_pop, alpha = params
        shape = max(q_pop * cfg.ground_tendency_prior_seconds / 900.0, 1e-6)
        prior = ground_log_gamma_prior(grid, q_pop, shape)
        pending = []
        for rec in batch.to_dict("records"):
            fid = str(rec["fighter_id"]); y = float(rec["ground_attempted"]); e = float(rec["own_control_seconds"])
            lp = prior + bank.evidence(fid, len(grid))
            pre, sd = weighted_mean_sd(grid, normalize_log_weights(lp))
            ll = None; post = pre
            if e > 0:
                ll = nb2_log_likelihood(y, burst + e / 900.0 * grid, alpha)
                post, _ = weighted_mean_sd(grid, normalize_log_weights(lp + ll))
            rows.append({**rec, "trait": "ground_striking_tendency", "pre_rating": pre,
                         "pre_posterior_sd": sd, "post_rating": post, "population_burst": burst,
                         "population_rate_15m": q_pop, "observation_alpha": alpha,
                         "numerator": y, "denominator": e})
            pending.append((fid, y, e, ll))
        for fid, y, e, ll in pending:
            if e > 0 and ll is not None:
                bank.add(fid, ll); pop_y.append(y); pop_e.append(e)
    return pd.DataFrame(rows)


def rolling_ground_suppression(tendency: pd.DataFrame, cfg: FSRV3Config, mode: str) -> pd.DataFrame:
    grid = np.linspace(cfg.ground_suppression_grid_min, cfg.ground_suppression_grid_max, cfg.ground_suppression_grid_points)
    source = tendency.copy()
    source["attacker_slope"] = source["denominator"].astype(float) / 900.0 * source["pre_rating"].astype(float)
    bank = EvidenceBank(mode)
    py: list[float] = []; pb: list[float] = []; ps: list[float] = []; params = None; rows = []
    for _, batch in source.groupby("event_date", sort=True):
        params = ground_fit_suppression_population(py, pb, ps, params, cfg)
        s_pop, alpha = params
        prior = ground_log_gamma_prior(grid, s_pop, cfg.ground_suppression_prior_shape)
        pending = []
        for rec in batch.to_dict("records"):
            defender = str(rec["opponent_id"]); y = float(rec["numerator"])
            burst = float(rec["population_burst"]); slope = float(rec["attacker_slope"])
            lp = prior + bank.evidence(defender, len(grid))
            pre, sd = weighted_mean_sd(grid, normalize_log_weights(lp))
            ll = None; post = pre
            if slope > 0:
                ll = nb2_log_likelihood(y, burst + grid * slope, alpha)
                post, _ = weighted_mean_sd(grid, normalize_log_weights(lp + ll))
            rows.append({"event_date": rec["event_date"], "fight_id": str(rec["fight_id"]),
                         "fighter_id": defender, "trait": "ground_striking_suppression",
                         "pre_rating": pre, "pre_posterior_sd": sd, "post_rating": post,
                         "population_multiplier": s_pop})
            pending.append((defender, y, burst, slope, ll))
        for defender, y, burst, slope, ll in pending:
            if slope > 0 and ll is not None:
                bank.add(defender, ll); py.append(y); pb.append(burst); ps.append(slope)
    return pd.DataFrame(rows)


def rolling_ground_effectiveness(fights: pd.DataFrame, cfg: FSRV3Config, mode: str) -> pd.DataFrame:
    grid = np.linspace(cfg.ground_effectiveness_grid_min, cfg.ground_effectiveness_grid_max, cfg.ground_effectiveness_grid_points)
    prior = ground_normal_prior(grid, cfg.ground_effectiveness_sigma)
    bank = EvidenceBank(mode)
    py: list[float] = []; pn: list[float] = []; beta = None; rows = []
    for event_date, batch in fights.groupby("event_date", sort=True):
        beta = ground_fit_population_beta(py, pn, beta, cfg)
        baseline = float(expit(beta))
        pending = []
        for rec in batch.to_dict("records"):
            fid = str(rec["fighter_id"]); y = float(rec["ground_landed"]); n = float(rec["ground_attempted"])
            lp = prior + bank.evidence(fid, len(grid))
            pre, sd = weighted_mean_sd(grid, normalize_log_weights(lp))
            ll = None; post = pre
            if n > 0:
                ll = beta_binomial_log_likelihood(y, n, expit(beta + grid), cfg.ground_effectiveness_rho)
                post, _ = weighted_mean_sd(grid, normalize_log_weights(lp + ll))
            rows.append({"event_date": event_date, "fight_id": str(rec["fight_id"]), "fighter_id": fid,
                         "trait": "ground_striking_offense", "pre_rating": pre,
                         "pre_posterior_sd": sd, "post_rating": post, "population_baseline": baseline})
            pending.append((fid, y, n, ll))
        for fid, y, n, ll in pending:
            if n > 0 and ll is not None:
                bank.add(fid, ll); py.append(y); pn.append(n)
    return pd.DataFrame(rows)


def rolling_power(snapshot_keys: pd.DataFrame, mode: str, cfg: FSRV3Config) -> pd.DataFrame:
    obs = power_prepare_observations()
    grid = _power_grid(cfg); prior = power_prior_logweights(grid, cfg.power_sigma)
    model_obs = obs[obs["sig_landed"] > 0].copy()
    train = model_obs[model_obs["date"] < pd.Timestamp(cfg.power_train_state_cutoff)]
    beta = power_fit_population_beta(train, cfg.power_rho)
    bank = EvidenceBank(mode); rows = []
    keys = snapshot_keys[["event_date", "fight_id", "fighter_id"]].copy()
    dates = sorted(set(keys["event_date"]).union(set(obs["date"])))
    cache: dict[tuple[float, float], np.ndarray] = {}
    for date in dates:
        day_keys = keys[keys["event_date"].eq(date)]
        for rec in day_keys.to_dict("records"):
            fid = str(rec["fighter_id"])
            lp = prior + bank.evidence(fid, len(grid))
            pre, sd = weighted_mean_sd(grid, normalize_log_weights(lp))
            rows.append({**rec, "trait": "striking_power_v3", "pre_rating": pre,
                         "pre_posterior_sd": sd, "population_beta": beta})
        day_obs = model_obs[model_obs["date"].eq(date)]
        pending = []
        for rec in day_obs.to_dict("records"):
            key = (float(rec["kd_scored"]), float(rec["sig_landed"]))
            ll = cache.get(key)
            if ll is None:
                ll = power_grid_loglik(k=key[0], n=key[1], beta=beta, rho=cfg.power_rho, grid=grid)
                cache[key] = ll
            pending.append((str(rec["fighter_id"]), ll))
        for fid, ll in pending: bank.add(fid, ll)
    return pd.DataFrame(rows)


def rolling_kd_resistance(power_history: pd.DataFrame, mode: str, cfg: ActiveTraitConfig) -> pd.DataFrame:
    obs = kd_prepare_observations(cfg).copy()
    p = power_history[["event_date", "fight_id", "fighter_id", "pre_rating"]].rename(
        columns={"fighter_id": "opponent_id", "pre_rating": "variant_attacker_power"})
    obs = obs.drop(columns=["attacker_power_v3"], errors="ignore").merge(
        p, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")
    if obs["variant_attacker_power"].isna().any():
        raise RuntimeError("variant power missing for KD context")
    obs["context_offset"] = (obs["variant_attacker_power"].astype(float)
        + float(cfg.kd_attacker_age_beta) * (obs["attacker_age"].astype(float) - 30.0)
        + float(cfg.kd_defender_age_beta) * (obs["defender_age"].astype(float) - 30.0))
    beta = kd_fit_population_beta(obs[obs["event_date"] < pd.Timestamp(cfg.kd_resistance_train_state_cutoff)], cfg.kd_resistance_rho)
    grid = kd_grid(cfg); prior = kd_prior_logweights(grid, cfg.kd_resistance_sigma)
    bank = EvidenceBank(mode); rows = []
    for date, batch in obs.groupby("event_date", sort=True):
        pending = []
        for rec in batch.to_dict("records"):
            fid = str(rec["fighter_id"])
            lp = prior + bank.evidence(fid, len(grid))
            pre, sd = weighted_mean_sd(grid, normalize_log_weights(lp))
            rows.append({"event_date": date, "fight_id": str(rec["fight_id"]), "fighter_id": fid,
                         "trait": "knockdown_resistance_v3", "pre_rating": pre,
                         "pre_posterior_sd": sd, "population_beta": beta})
            if float(rec["n"]) > 0:
                eta = beta + float(rec["context_offset"]) - grid
                ll = beta_binomial_log_likelihood(float(rec["k"]), float(rec["n"]), expit(eta), cfg.kd_resistance_rho)
                pending.append((fid, ll))
        for fid, ll in pending: bank.add(fid, ll)
    return pd.DataFrame(rows)


def rolling_escape(paired: pd.DataFrame, mode: str, cfg: ActiveTraitConfig) -> pd.DataFrame:
    fights = aggregate_fights(paired)
    suffered, inflicted = ScalarBank(mode), ScalarBank(mode)
    population_duration = 0.0; population_entries = 0.0; rows = []
    prior = float(cfg.escape_prior_entries)
    for date, batch in fights.groupby("event_date", sort=True):
        mu_pop = population_duration / population_entries if population_entries > 0 else 60.0
        pending = []
        for rec in batch.to_dict("records"):
            fid = str(rec["fighter_id"])
            sd, se = suffered.totals(fid); idur, ie = inflicted.totals(fid)
            mu_s = (sd + mu_pop * prior) / (se + prior)
            mu_i = (idur + mu_pop * prior) / (ie + prior)
            off = math.log(mu_pop / max(mu_s, 1e-9))
            defense = math.log(max(mu_i, 1e-9) / mu_pop)
            common = {"event_date": pd.Timestamp(date), "fight_id": str(rec["fight_id"]), "fighter_id": fid,
                      "population_duration_baseline_seconds": mu_pop}
            rows.append({**common, "trait": "escape_offense", "pre_rating": off})
            rows.append({**common, "trait": "escape_defense", "pre_rating": defense})
            sdur = float(rec["qualified_control_suffered_seconds"]); sentries = float(rec["opponent_ground_entries"])
            iduration = float(rec["qualified_control_inflicted_seconds"]); ientries = float(rec["ground_entries"])
            pending.append((fid, sdur, sentries, iduration, ientries))
        for fid, sd, se, idur, ie in pending:
            suffered.add(fid, sd, se); inflicted.add(fid, idur, ie)
            population_duration += sd + idur; population_entries += se + ie
    return pd.DataFrame(rows)


def build_variant(canonical: pd.DataFrame, mode: str) -> pd.DataFrame:
    cfg = FSRV3Config(); active = ActiveTraitConfig(); paired = build_paired_rounds()
    s_spec, t_spec = standing_spec(cfg), takedown_spec(cfg)
    s_eff, t_eff = standing_effectiveness_spec(cfg), takedown_effectiveness_spec(cfg)
    s_fights = build_rate_fighter_fights(s_spec, paired_rounds=paired)
    t_fights = build_rate_fighter_fights(t_spec, paired_rounds=paired)
    s_t = rolling_rate(s_fights, s_spec, mode); s_s = rolling_rate_suppression(s_t, s_spec, mode)
    t_t = rolling_rate(t_fights, t_spec, mode); t_s = rolling_rate_suppression(t_t, t_spec, mode)
    s_e = rolling_paired(build_effectiveness_fighter_fights(s_eff, paired_rounds=paired), s_eff, mode)
    t_e = rolling_paired(build_effectiveness_fighter_fights(t_eff, paired_rounds=paired), t_eff, mode)
    g_fights = build_ground_fighter_fights(paired)
    g_t = rolling_ground_tendency(g_fights, cfg, mode)
    g_s = rolling_ground_suppression(g_t, cfg, mode)
    g_e = rolling_ground_effectiveness(g_fights, cfg, mode)
    power = rolling_power(canonical, mode, cfg)
    kd = rolling_kd_resistance(power, mode, active)
    escape = rolling_escape(paired, mode, active)

    keys = ["event_date", "fight_id", "fighter_id"]
    out = canonical.copy()
    def replace(frame, trait, column, baseline_from=None, baseline_column=None):
        nonlocal out
        x = frame[frame["trait"].eq(trait)][keys + ["pre_rating"] + ([baseline_from] if baseline_from else [])].copy()
        ren = {"pre_rating": column}
        if baseline_from and baseline_column: ren[baseline_from] = baseline_column
        x = x.rename(columns=ren)
        out = out.drop(columns=[column] + ([baseline_column] if baseline_column and baseline_column in out.columns else []), errors="ignore")
        out = out.merge(x, on=keys, how="left", validate="one_to_one")

    replace(s_t, s_spec.tendency_trait, "standing_striking_tendency")
    replace(s_s, s_spec.suppression_trait, "standing_striking_suppression")
    replace(s_e, s_eff.offense_trait, "standing_striking_offense", "population_baseline", "standing_accuracy_baseline")
    replace(s_e, s_eff.defense_trait, "standing_striking_defense")
    replace(t_t, t_spec.tendency_trait, "takedown_tendency")
    replace(t_s, t_spec.suppression_trait, "takedown_suppression")
    replace(t_e, t_eff.offense_trait, "takedown_offense", "population_baseline", "takedown_completion_baseline")
    replace(t_e, t_eff.defense_trait, "takedown_defense")
    replace(g_t, "ground_striking_tendency", "ground_striking_tendency", "population_burst", "ground_striking_burst_baseline")
    out = out.drop(columns=["ground_striking_population_slope_15m"], errors="ignore").merge(
        g_t[keys + ["population_rate_15m"]].rename(columns={"population_rate_15m": "ground_striking_population_slope_15m"}),
        on=keys, how="left", validate="one_to_one")
    replace(g_s, "ground_striking_suppression", "ground_striking_suppression")
    replace(g_e, "ground_striking_offense", "ground_striking_offense", "population_baseline", "ground_accuracy_baseline")
    replace(power, "striking_power_v3", "striking_power_v3")
    replace(kd, "knockdown_resistance_v3", "knockdown_resistance_v3")
    replace(escape, "escape_offense", "escape_offense", "population_duration_baseline_seconds", "escape_population_mean_seconds")
    replace(escape, "escape_defense", "escape_defense")

    changed = [
        "standing_striking_tendency", "standing_striking_suppression", "standing_striking_offense", "standing_striking_defense",
        "takedown_tendency", "takedown_suppression", "takedown_offense", "takedown_defense",
        "ground_striking_tendency", "ground_striking_suppression", "ground_striking_offense",
        "striking_power_v3", "knockdown_resistance_v3", "escape_offense", "escape_defense",
    ]
    missing = [c for c in changed if c not in out.columns or out[c].isna().any()]
    if missing: raise RuntimeError(f"recency overlay missing values: {missing}")
    if mode == "ewm":
        base = canonical.set_index(keys)
        idx = out.set_index(keys)
        for c in changed:
            idx[c] = EWM_CANONICAL_BLEND * base[c] + (1.0 - EWM_CANONICAL_BLEND) * idx[c]
        out = idx.reset_index()
    return out.sort_values(keys).reset_index(drop=True)


def actual_red_win(row: pd.Series) -> int | None:
    for col in ("winner_id", "winner"):
        if col in row and not pd.isna(row[col]):
            value = str(row[col])
            if value in {str(row.get("r_id")), str(row.get("r_name")), "red", "R"}: return 1
            if value in {str(row.get("b_id")), str(row.get("b_name")), "blue", "B"}: return 0
    if "r_result" in row and str(row.get("r_result")).upper() == "W": return 1
    if "b_result" in row and str(row.get("b_result")).upper() == "W": return 0
    return None


def choose_mature_cohort(master: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    m = master.drop_duplicates("fight_id").copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize(); m["fight_id"] = m["fight_id"].astype(str)
    seen = defaultdict(int); rows = []
    valid_keys = set(zip(snapshots["event_date"], snapshots["fight_id"], snapshots["fighter_id"]))
    for _, row in m.sort_values(["date", "fight_id"]).iterrows():
        rid, bid = str(row["r_id"]), str(row["b_id"])
        y = actual_red_win(row)
        has_snap = ((row["date"], row["fight_id"], rid) in valid_keys and (row["date"], row["fight_id"], bid) in valid_keys)
        if seen[rid] >= 3 and seen[bid] >= 3 and y is not None and has_snap:
            rec = row.copy(); rec["actual_red_win"] = y; rec["red_prior"] = seen[rid]; rec["blue_prior"] = seen[bid]; rows.append(rec)
        seen[rid] += 1; seen[bid] += 1
    cohort = pd.DataFrame(rows).sort_values(["date", "fight_id"]).tail(COHORT_SIZE).copy()
    if len(cohort) < 50: raise RuntimeError(f"mature cohort too small: {len(cohort)}")
    return cohort


def score(record: dict, cohort: pd.DataFrame) -> dict:
    probs = record["simulator_metrics"]["fight_probabilities"]
    p = np.array([float(probs[str(fid)]["red_moneyline"]) for fid in cohort["fight_id"]], dtype=float)
    y = cohort["actual_red_win"].astype(float).to_numpy()
    pclip = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)), "brier": float(np.mean((p-y)**2)),
        "log_loss": float(-np.mean(y*np.log(pclip)+(1-y)*np.log(1-pclip))),
        "accuracy": float(np.mean((p >= .5) == (y >= .5))),
        "mean_confidence": float(np.mean(np.maximum(p, 1-p))),
        "red_probs": {str(fid): float(v) for fid, v in zip(cohort["fight_id"], p)},
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str); canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    master = pd.read_parquet(MASTER_PATH).copy()
    cohort = choose_mature_cohort(master, canonical)
    cohort_ids = set(cohort["fight_id"].astype(str))
    master.loc[master["fight_id"].astype(str).isin(cohort_ids), "event_name"] = SHADOW_EVENT_NAME
    master.to_parquet(MASTER_PATH, index=False)
    cohort[["date", "fight_id", "r_name", "b_name", "actual_red_win", "red_prior", "blue_prior"]].to_csv(OUTDIR / "cohort.csv", index=False)

    variants = {"canonical": canonical, "last3_all_v3": build_variant(canonical, "last3"), "blended_ewm_all_v3": build_variant(canonical, "ewm")}
    results = {}
    for name, snapshots in variants.items():
        snapshots.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        output = OUTDIR / f"{name}.json"
        rec = run(split="calibration", paths_per_fight=PATHS_PER_FIGHT,
                  config_path=Path("configs/event_clock_v2/calibration/default.yaml"), output=output,
                  ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2, event_name=SHADOW_EVENT_NAME)
        results[name] = score(rec, cohort)

    canonical.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
    base = results["canonical"]
    summary = {
        "design": {"mature_definition": "both fighters >=3 strictly prior UFC fights",
                   "cohort_size": int(len(cohort)), "paths_per_fight": PATHS_PER_FIGHT,
                   "last3": "same validated V3 likelihood/prior families, fighter evidence limited to three strictly prior fights; population baselines remain all-prior",
                   "ewm": f"fighter evidence exponentially decayed by {EWM_DECAY} per prior fight, then 50/50 blended with canonical full-history prefight trait means",
                   "changed_traits": "all V3-native fight-relevant traits: standing, takedown, ground striking, power, KD resistance, escape/retention",
                   "held_fixed": "inherited V2 durability, stamina and submission fields; Event Clock mechanics, age, judging, seeds"},
        "scores": {k: {kk: vv for kk, vv in v.items() if kk != "red_probs"} for k, v in results.items()},
        "delta_vs_canonical": {
            k: {"brier": v["brier"]-base["brier"], "log_loss": v["log_loss"]-base["log_loss"],
                "accuracy": v["accuracy"]-base["accuracy"]}
            for k, v in results.items() if k != "canonical"},
        "fight_probabilities": {k: v["red_probs"] for k, v in results.items()},
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
