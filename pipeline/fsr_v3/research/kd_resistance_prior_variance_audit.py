"""Measurement-only native prior/variance audit for live knockdown resistance."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v2.physical import build_physical_observations
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay.math import beta_binomial_log_likelihood
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import ShadowKOKDCalibration
from pipeline.simulation.event_mc_v1.single_fight import fighter_age_years

SEED = 20260822
VALIDATION_START = pd.Timestamp("2022-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
DEFAULT_OUT = Path("data/diagnostics/fsr_v3/active_trait_audit/kd_resistance")
SIGMA_CANDIDATES = (0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90)
RHO_CANDIDATES = (0.005, 0.01, 0.02, 0.05, 0.10)
C_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
GRID = np.linspace(-2.5, 2.5, 501)
GH_X, GH_W = hermgauss(15)
GH_W = GH_W / np.sqrt(np.pi)


def _bucket(n):
    n = int(n)
    return "0" if n <= 0 else "1" if n == 1 else "2" if n == 2 else "3plus"


def _prior_logw(sigma):
    if sigma <= 0:
        out = np.full_like(GRID, -np.inf)
        out[np.argmin(np.abs(GRID))] = 0.0
        return out
    out = -0.5 * np.square(GRID / float(sigma))
    return out - logsumexp(out)


def _moments(logw):
    w = np.exp(logw - logsumexp(logw))
    mean = float(np.sum(w * GRID))
    var = float(np.sum(w * np.square(GRID - mean)))
    return mean, float(np.sqrt(max(var, 0.0)))


def build_observations():
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    obs = build_physical_observations(rounds, master).copy()
    obs["event_date"] = pd.to_datetime(obs["date"], errors="raise").dt.normalize()
    obs["fight_id"] = obs["fight_id"].astype(str)
    obs["fighter_id"] = obs["fighter_id"].astype(str)
    obs["opponent_id"] = obs["opponent_id"].astype(str)

    fsr = pd.read_parquet(
        FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
        columns=["event_date", "fight_id", "fighter_id", "striking_power_v3", "knockdown_resistance"],
    ).copy()
    fsr["event_date"] = pd.to_datetime(fsr["event_date"], errors="raise").dt.normalize()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    own = fsr.rename(columns={"knockdown_resistance": "legacy_kdres"})[
        ["event_date", "fight_id", "fighter_id", "legacy_kdres"]
    ]
    opp = fsr.rename(columns={"fighter_id": "opponent_id", "striking_power_v3": "attacker_power_v3"})[
        ["event_date", "fight_id", "opponent_id", "attacker_power_v3"]
    ]
    obs = obs.merge(own, on=["event_date", "fight_id", "fighter_id"], how="inner", validate="one_to_one")
    obs = obs.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="inner", validate="one_to_one")

    m = master[["fight_id", "date", "r_id", "b_id", "r_dob", "b_dob"]].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    age_rows = []
    for row in m.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        age_rows += [
            {"fight_id": row.fight_id, "fighter_id": str(row.r_id), "age": fighter_age_years(row.r_dob, date)},
            {"fight_id": row.fight_id, "fighter_id": str(row.b_id), "age": fighter_age_years(row.b_dob, date)},
        ]
    ages = pd.DataFrame(age_rows)
    obs = obs.merge(ages.rename(columns={"age": "defender_age"}), on=["fight_id", "fighter_id"], how="left")
    obs = obs.merge(
        ages.rename(columns={"fighter_id": "opponent_id", "age": "attacker_age"}),
        on=["fight_id", "opponent_id"], how="left",
    )

    obs["k"] = pd.to_numeric(obs["kd_absorbed"], errors="coerce").fillna(0.0)
    obs["n"] = pd.to_numeric(obs["sig_absorbed"], errors="coerce").fillna(0.0)
    obs["k"] = np.minimum(obs["k"].clip(lower=0), obs["n"].clip(lower=0))
    obs = obs[obs["n"] > 0].copy()

    appearances = obs[["event_date", "fighter_id"]].drop_duplicates()
    counts = appearances.groupby(["fighter_id", "event_date"], as_index=False).size().sort_values(["fighter_id", "event_date"])
    counts["prior_ufc_fights"] = counts.groupby("fighter_id")["size"].cumsum() - counts["size"]
    obs = obs.merge(counts[["fighter_id", "event_date", "prior_ufc_fights"]], on=["fighter_id", "event_date"], how="left")
    obs["prior_bucket"] = obs["prior_ufc_fights"].map(_bucket)

    c = ShadowKOKDCalibration()
    obs["context_offset"] = (
        obs["attacker_power_v3"].astype(float)
        + c.kd_attacker_age_beta * (obs["attacker_age"].astype(float) - 30.0)
        + c.kd_defender_age_beta * (obs["defender_age"].astype(float) - 30.0)
    )
    return obs.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def fit_beta(train, rho):
    k = train["k"].to_numpy(float)
    n = train["n"].to_numpy(float)
    off = train["context_offset"].to_numpy(float)

    def objective(beta):
        p = expit(float(beta) + off)
        return -float(np.sum(beta_binomial_log_likelihood(k, n, p, rho)))

    fit = minimize_scalar(objective, bounds=(-9.0, -2.0), method="bounded")
    if not fit.success:
        raise RuntimeError("KD resistance population beta fit failed")
    return float(fit.x)


def predictive_ll(k, n, base_eta, mean, sd, c, rho):
    if c <= 0 or sd <= 1e-12:
        return float(beta_binomial_log_likelihood(k, n, expit(base_eta - mean), rho))
    r = mean + np.sqrt(2 * c) * sd * GH_X
    ll = beta_binomial_log_likelihood(k, n, expit(base_eta - r), rho)
    m = float(np.max(ll))
    return float(m + np.log(np.sum(GH_W * np.exp(ll - m))))


def replay(obs, sigma, rho, beta):
    prior = _prior_logw(sigma)
    states = {}
    rows = []
    legacy_beta = ShadowKOKDCalibration().kd_kdres_beta
    for _, batch in obs.groupby("event_date", sort=True):
        pending = defaultdict(lambda: np.zeros_like(GRID))
        for rec in batch.to_dict("records"):
            fighter = str(rec["fighter_id"])
            logw = states.get(fighter, prior)
            mean, sd = _moments(logw)
            base_eta = beta + float(rec["context_offset"])
            k, n = float(rec["k"]), float(rec["n"])
            plugin_p = float(expit(base_eta - mean))
            pop_p = float(expit(base_eta))
            legacy_p = float(expit(base_eta + legacy_beta * (float(rec["legacy_kdres"]) - 50.0)))
            row = dict(rec)
            row.update({
                "population_beta": beta, "rho": rho, "sigma": sigma,
                "pre_mean": mean, "pre_sd": sd,
                "plugin_p": plugin_p, "population_p": pop_p, "legacy_p": legacy_p,
                "plugin_ll": float(beta_binomial_log_likelihood(k, n, plugin_p, rho)),
                "population_ll": float(beta_binomial_log_likelihood(k, n, pop_p, rho)),
                "legacy_ll": float(beta_binomial_log_likelihood(k, n, legacy_p, rho)),
                "plugin_abs_error_kd": abs(k - n * plugin_p),
                "population_abs_error_kd": abs(k - n * pop_p),
                "legacy_abs_error_kd": abs(k - n * legacy_p),
            })
            for c in C_CANDIDATES:
                row[f"predictive_ll_c_{c:g}"] = predictive_ll(k, n, base_eta, mean, sd, c, rho)
            rows.append(row)
            pending[fighter] += beta_binomial_log_likelihood(k, n, expit(base_eta - GRID), rho)
        for fighter, ll in pending.items():
            cur = states.get(fighter, prior)
            updated = cur + ll
            states[fighter] = updated - logsumexp(updated)
    return pd.DataFrame(rows)


def window(frame, start, end, ll_col):
    x = frame[frame["event_date"] >= pd.Timestamp(start)]
    if end is not None:
        x = x[x["event_date"] < pd.Timestamp(end)]
    return {"rows": len(x), "fights": x["fight_id"].nunique(), "total_ll": x[ll_col].sum(), "mean_ll": x[ll_col].mean()}


def bootstrap(frame, a, b, draws=2000, seed=SEED):
    d = frame.groupby("fight_id")[[a, b]].sum()
    delta = (d[a] - d[b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    sims = np.array([delta[rng.integers(0, len(delta), len(delta))].sum() for _ in range(draws)])
    return float(delta.sum()), float(np.quantile(sims, .025)), float(np.quantile(sims, .975)), float(np.mean(sims > 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs = build_observations()
    train = obs[obs["event_date"] < VALIDATION_START]
    sweep = []
    for rho in RHO_CANDIDATES:
        beta = fit_beta(train, rho)
        for sigma in SIGMA_CANDIDATES:
            f = replay(obs, sigma, rho, beta)
            s = window(f, VALIDATION_START, HOLDOUT_START, "predictive_ll_c_1")
            sweep.append({"rho": rho, "sigma": sigma, "beta": beta, **s})
    sweep = pd.DataFrame(sweep)
    best = sweep.sort_values(["total_ll", "sigma"], ascending=[False, True]).iloc[0]
    rho, sigma, beta = float(best.rho), float(best.sigma), float(best.beta)
    selected = replay(obs, sigma, rho, beta)
    cs = []
    for c in C_CANDIDATES:
        for label, start, end in (
            ("validation_2022_2023", VALIDATION_START, HOLDOUT_START),
            ("holdout_2024plus", HOLDOUT_START, None),
        ):
            cs.append({"c": c, "window": label, **window(selected, start, end, f"predictive_ll_c_{c:g}")})
    cs = pd.DataFrame(cs)
    best_c = float(cs[cs.window.eq("validation_2022_2023")].sort_values(["total_ll", "c"], ascending=[False, True]).iloc[0].c)
    hold = selected[selected["event_date"] >= HOLDOUT_START].copy()
    hold["selected_ll"] = hold[f"predictive_ll_c_{best_c:g}"]
    bp = bootstrap(hold, "selected_ll", "population_ll", args.bootstrap_draws)
    bl = bootstrap(hold, "selected_ll", "legacy_ll", args.bootstrap_draws, SEED + 1)
    buckets = hold.groupby("prior_bucket").apply(lambda x: pd.Series({
        "rows": len(x), "fights": x.fight_id.nunique(),
        "delta_ll_vs_population": (x.selected_ll-x.population_ll).sum(),
        "delta_ll_vs_legacy": (x.selected_ll-x.legacy_ll).sum(),
        "plugin_mae_kd": x.plugin_abs_error_kd.mean(),
        "legacy_mae_kd": x.legacy_abs_error_kd.mean(),
    }), include_groups=False).reset_index()
    sweep.to_csv(args.output_dir/"rho_sigma_sweep.csv", index=False)
    cs.to_csv(args.output_dir/"variance_multiplier_sweep.csv", index=False)
    hold.to_csv(args.output_dir/"holdout_rows.csv", index=False)
    buckets.to_csv(args.output_dir/"holdout_prior_buckets.csv", index=False)
    print("="*120)
    print("FSR V3 ACTIVE TRAIT AUDIT — KNOCKDOWN RESISTANCE")
    print("="*120)
    print(f"rows={len(obs):,} pre-2022={len(train):,}")
    print("TOP VALIDATION RHO/SIGMA")
    print(sweep.sort_values("total_ll", ascending=False).head(12).to_string(index=False))
    print(f"selected rho={rho:g} sigma={sigma:g} beta={beta:.6f}")
    print("EPISTEMIC C")
    print(cs.to_string(index=False))
    print(f"selected c={best_c:g}")
    print(f"HOLDOUT selected vs population LL={bp[0]:+.3f} CI[{bp[1]:+.3f},{bp[2]:+.3f}] P>0={bp[3]:.3f}")
    print(f"HOLDOUT selected vs legacy KDRES LL={bl[0]:+.3f} CI[{bl[1]:+.3f},{bl[2]:+.3f}] P>0={bl[3]:.3f}")
    print(buckets.to_string(index=False))
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
