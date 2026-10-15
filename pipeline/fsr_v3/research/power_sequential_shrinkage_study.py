from __future__ import annotations

"""Measurement-only sequential shrinkage/uncertainty study for FSR V3 power.

Stage 1 selected attacker-only knockdown production and rejected adding KO wins
into the learning target.  This stage asks how that attacker signal should be
shrunk and whether its epistemic posterior should be propagated.

Observation model
-----------------
For each fighter-fight with at least one landed significant strike:

    K_if ~ BetaBinomial(N_if, p_if, rho)
    logit(p_if) = beta + power_i
    power_i ~ Normal(0, sigma^2)

where K is knockdowns scored and N is landed significant strikes.  ``rho`` is
fight-level aleatoric overdispersion.  The posterior over ``power_i`` is
represented on a fixed logit grid.  Same-date updates are delayed until every
fighter on that date has been scored.

Development folds select sigma, rho, and epistemic propagation scale c using
future predictive log likelihood.  2024+ is reserved as the outer holdout.
This script does not publish or modify any FSR trait.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, expit, gammaln, logsumexp
from sklearn.metrics import roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v2.physical import build_physical_observations


SIGMA_GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25)
RHO_GRID = (0.0, 0.01, 0.03, 0.06, 0.10)
C_GRID = (0.0, 0.35, 0.70, 1.00)
POWER_GRID = np.linspace(-4.0, 4.0, 321)
DEV_FOLDS = (
    ("2020", "2020-01-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
)
OUTER_START = pd.Timestamp("2024-01-01")
EPS = 1e-10
GH_X, GH_W = np.polynomial.hermite.hermgauss(11)
GH_LOG_W = np.log(GH_W) - 0.5 * np.log(np.pi)


@dataclass
class SequentialResult:
    metrics: dict[float, dict[str, float]]
    detail: pd.DataFrame


def _prepare() -> pd.DataFrame:
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    obs = build_physical_observations(rounds, master).copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs["fight_id"] = obs["fight_id"].astype(str)
    obs["fighter_id"] = obs["fighter_id"].astype(str)
    obs["opponent_id"] = obs["opponent_id"].astype(str)
    obs = obs.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)
    obs["prior_ufc_fights"] = obs.groupby("fighter_id").cumcount()
    for col in ("sig_landed", "kd_scored", "ko_win"):
        obs[col] = pd.to_numeric(obs[col], errors="coerce").fillna(0.0)
    obs = obs[obs["sig_landed"] > 0].copy()
    obs["kd_scored"] = np.minimum(obs["kd_scored"], obs["sig_landed"])
    return obs.reset_index(drop=True)


def _logit(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(np.log(p / (1.0 - p)))


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
    k = train["kd_scored"].to_numpy(float)
    n = train["sig_landed"].to_numpy(float)

    def objective(beta: float) -> float:
        p = expit(float(beta))
        return -float(np.sum(_bb_loglik_p(k, n, p, rho)))

    fit = minimize_scalar(objective, bounds=(-10.0, -1.0), method="bounded")
    if not fit.success:
        raise RuntimeError(f"population beta fit failed for rho={rho}")
    return float(fit.x)


def _grid_loglik(row, beta: float, rho: float) -> np.ndarray:
    p = expit(beta + POWER_GRID)
    return np.asarray(_bb_loglik_p(float(row.kd_scored), float(row.sig_landed), p, rho), float)


def _training_evidence(train: pd.DataFrame, beta: float, rho: float) -> dict[str, np.ndarray]:
    evidence: dict[str, np.ndarray] = {}
    cache: dict[tuple[float, float], np.ndarray] = {}
    for row in train.itertuples(index=False):
        key = (float(row.kd_scored), float(row.sig_landed))
        ll = cache.get(key)
        if ll is None:
            ll = _grid_loglik(row, beta, rho)
            cache[key] = ll
        fighter = str(row.fighter_id)
        if fighter not in evidence:
            evidence[fighter] = np.zeros_like(POWER_GRID)
        evidence[fighter] += ll
    return evidence


def _prior_logweights(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        out = np.full_like(POWER_GRID, -np.inf)
        out[int(np.argmin(np.abs(POWER_GRID)))] = 0.0
        return out
    out = -0.5 * np.square(POWER_GRID / float(sigma))
    return out - logsumexp(out)


def _initial_states(train_evidence: dict[str, np.ndarray], sigma: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    prior = _prior_logweights(sigma)
    states: dict[str, np.ndarray] = {}
    for fighter, evidence in train_evidence.items():
        lw = prior + evidence
        states[fighter] = lw - logsumexp(lw)
    return states, prior


def _posterior_moments(logw: np.ndarray, beta: float) -> tuple[float, float, float]:
    w = np.exp(logw - logsumexp(logw))
    mean = float(np.sum(w * POWER_GRID))
    var = float(np.sum(w * np.square(POWER_GRID - mean)))
    mean_p = float(np.sum(w * expit(beta + POWER_GRID)))
    return mean, float(np.sqrt(max(var, 0.0))), mean_p


def _predictive_ll(k: float, n: float, beta: float, mean: float, sd: float, rho: float, c: float) -> float:
    if c <= 0.0 or sd <= 1e-12:
        return float(_bb_loglik_p(k, n, expit(beta + mean), rho))
    deltas = mean + np.sqrt(2.0) * float(c) * sd * GH_X
    ll = _bb_loglik_p(k, n, expit(beta + deltas), rho)
    return float(logsumexp(GH_LOG_W + ll))


def _safe_auc(y, score) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def _sequential_score(
    test: pd.DataFrame,
    *,
    beta: float,
    rho: float,
    sigma: float,
    train_evidence: dict[str, np.ndarray],
) -> SequentialResult:
    states, prior = _initial_states(train_evidence, sigma)
    ll_by_c = {c: 0.0 for c in C_GRID}
    detail_rows: list[dict[str, object]] = []
    cache: dict[tuple[float, float], np.ndarray] = {}

    for date, date_rows in test.groupby("date", sort=True):
        pending: list[tuple[str, np.ndarray]] = []
        for row in date_rows.itertuples(index=False):
            fighter = str(row.fighter_id)
            logw = states.get(fighter, prior)
            mean, sd, mean_p = _posterior_moments(logw, beta)
            k = float(row.kd_scored)
            n = float(row.sig_landed)
            for c in C_GRID:
                ll_by_c[c] += _predictive_ll(k, n, beta, mean, sd, rho, c)

            detail_rows.append({
                "fight_id": str(row.fight_id),
                "date": pd.Timestamp(date),
                "fighter_id": fighter,
                "opponent_id": str(row.opponent_id),
                "prior_ufc_fights": int(row.prior_ufc_fights),
                "sig_landed": n,
                "kd_scored": k,
                "ko_win": int(row.ko_win),
                "posterior_mean_logit_power": mean,
                "posterior_sd_logit_power": sd,
                "posterior_mean_kd_probability": mean_p,
            })

            key = (k, n)
            ll_grid = cache.get(key)
            if ll_grid is None:
                ll_grid = _grid_loglik(row, beta, rho)
                cache[key] = ll_grid
            pending.append((fighter, ll_grid))

        # Same-date delayed update: nobody on this date can affect another
        # prefight state on the same date.
        for fighter, ll_grid in pending:
            current = states.get(fighter, prior)
            updated = current + ll_grid
            states[fighter] = updated - logsumexp(updated)

    detail = pd.DataFrame(detail_rows)
    auc = _safe_auc(detail["ko_win"], detail["posterior_mean_logit_power"])
    mean_sd = float(detail["posterior_sd_logit_power"].mean()) if len(detail) else np.nan
    metrics = {
        c: {
            "kd_ll": float(ll_by_c[c]),
            "ko_winner_auc": auc,
            "mean_posterior_sd": mean_sd,
        }
        for c in C_GRID
    }
    return SequentialResult(metrics=metrics, detail=detail)


def _development(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for fold, start_s, end_s in DEV_FOLDS:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        train = obs[obs["date"] < start].copy()
        test = obs[(obs["date"] >= start) & (obs["date"] < end)].copy()
        if train.empty or test.empty:
            continue

        for rho in RHO_GRID:
            beta = _fit_population_beta(train, rho)
            evidence = _training_evidence(train, beta, rho)
            for sigma in SIGMA_GRID:
                result = _sequential_score(test, beta=beta, rho=rho, sigma=sigma, train_evidence=evidence)
                allowed_c = (0.0,) if sigma <= 0.0 else C_GRID
                for c in allowed_c:
                    m = result.metrics[c]
                    rows.append({
                        "fold": fold,
                        "sigma": sigma,
                        "rho": rho,
                        "c": c,
                        "fighter_fights": len(test),
                        "landed_sig": float(test["sig_landed"].sum()),
                        "population_kd_rate": float(expit(beta)),
                        "kd_ll": m["kd_ll"],
                        "ko_winner_auc": m["ko_winner_auc"],
                        "mean_posterior_sd": m["mean_posterior_sd"],
                    })

    dev = pd.DataFrame(rows)
    best_pop = (
        dev[dev["sigma"].eq(0.0)]
        .sort_values(["fold", "kd_ll"], ascending=[True, False])
        .groupby("fold", as_index=False)
        .first()[["fold", "kd_ll"]]
        .rename(columns={"kd_ll": "best_population_ll"})
    )
    dev = dev.merge(best_pop, on="fold", how="left", validate="many_to_one")
    dev["ll_gain_vs_best_population"] = dev["kd_ll"] - dev["best_population_ll"]

    agg = dev.groupby(["sigma", "rho", "c"], as_index=False).agg(
        folds=("fold", "nunique"),
        kd_ll=("kd_ll", "sum"),
        ll_gain_vs_best_population=("ll_gain_vs_best_population", "sum"),
        worst_fold_gain=("ll_gain_vs_best_population", "min"),
        folds_beating_population=("ll_gain_vs_best_population", lambda s: int((s > 0).sum())),
        mean_ko_winner_auc=("ko_winner_auc", "mean"),
        mean_posterior_sd=("mean_posterior_sd", "mean"),
        landed_sig=("landed_sig", "sum"),
    )
    agg["kd_ll_per_landed"] = agg["kd_ll"] / agg["landed_sig"]
    agg = agg.sort_values(["kd_ll", "mean_ko_winner_auc"], ascending=[False, False]).reset_index(drop=True)
    return dev, agg


def _outer(obs: pd.DataFrame, selected: pd.Series, best_population: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = obs[obs["date"] < OUTER_START].copy()
    test = obs[obs["date"] >= OUTER_START].copy()

    selected_rho = float(selected.rho)
    selected_sigma = float(selected.sigma)
    selected_c = float(selected.c)
    beta = _fit_population_beta(train, selected_rho)
    evidence = _training_evidence(train, beta, selected_rho)
    result = _sequential_score(test, beta=beta, rho=selected_rho, sigma=selected_sigma, train_evidence=evidence)

    pop_rho = float(best_population.rho)
    pop_beta = _fit_population_beta(train, pop_rho)
    pop_evidence: dict[str, np.ndarray] = {}
    pop_result = _sequential_score(test, beta=pop_beta, rho=pop_rho, sigma=0.0, train_evidence=pop_evidence)
    pop_ll = pop_result.metrics[0.0]["kd_ll"]

    rows = [{
        "model": "best_population_dev_selected",
        "sigma": 0.0,
        "rho": pop_rho,
        "c": 0.0,
        "fighter_fights": len(test),
        "fights": test["fight_id"].nunique(),
        "landed_sig": float(test["sig_landed"].sum()),
        "kd_ll": pop_ll,
        "ll_gain_vs_population": 0.0,
        "ko_winner_auc": 0.5,
        "mean_posterior_sd": 0.0,
    }]

    for c in C_GRID:
        m = result.metrics[c]
        rows.append({
            "model": "selected_power" if c == selected_c else "selected_power_c_ablation",
            "sigma": selected_sigma,
            "rho": selected_rho,
            "c": c,
            "fighter_fights": len(test),
            "fights": test["fight_id"].nunique(),
            "landed_sig": float(test["sig_landed"].sum()),
            "kd_ll": m["kd_ll"],
            "ll_gain_vs_population": m["kd_ll"] - pop_ll,
            "ko_winner_auc": m["ko_winner_auc"],
            "mean_posterior_sd": m["mean_posterior_sd"],
        })

    detail = result.detail.copy()
    try:
        fsr2 = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH).copy()
        fsr2["fight_id"] = fsr2["fight_id"].astype(str)
        fsr2["fighter_id"] = fsr2["fighter_id"].astype(str)
        detail = detail.merge(
            fsr2[["fight_id", "fighter_id", "striking_power"]],
            on=["fight_id", "fighter_id"], how="left", validate="one_to_one",
        )
        if detail["striking_power"].notna().all():
            v2_auc = _safe_auc(detail["ko_win"], detail["striking_power"])
            rows.append({
                "model": "published_fsr_v2_power_auc_only",
                "sigma": np.nan, "rho": np.nan, "c": np.nan,
                "fighter_fights": len(test), "fights": test["fight_id"].nunique(),
                "landed_sig": float(test["sig_landed"].sum()),
                "kd_ll": np.nan, "ll_gain_vs_population": np.nan,
                "ko_winner_auc": v2_auc, "mean_posterior_sd": np.nan,
            })
    except FileNotFoundError:
        pass

    return pd.DataFrame(rows), detail


def main(out_dir: str = "data/diagnostics/fsr_v3_power") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs = _prepare()

    print("=" * 120)
    print("FSR V3 POWER — SEQUENTIAL SHRINKAGE / UNCERTAINTY STUDY")
    print("=" * 120)
    print(f"fighter-fight observations: {len(obs):,}")
    print(f"date range: {obs.date.min().date()} to {obs.date.max().date()}")
    print("signal: attacker knockdowns / landed significant strikes")
    print("KO wins are NOT added to the learning target (rejected by structural study)")
    print("same-date updates: delayed")

    dev, selection = _development(obs)
    selected = selection.iloc[0]
    best_population = selection[selection["sigma"].eq(0.0)].iloc[0]

    print("\nTOP DEVELOPMENT CANDIDATES")
    print(selection.head(20).to_string(index=False))
    print("\nSELECTED")
    print(selected.to_string())
    print("\nBEST DEVELOPMENT POPULATION")
    print(best_population.to_string())

    outer, detail = _outer(obs, selected, best_population)
    print("\nRESERVED OUTER 2024+ RESULTS")
    print(outer.to_string(index=False))

    dev.to_csv(out / "power_sequential_dev_folds.csv", index=False)
    selection.to_csv(out / "power_sequential_selection.csv", index=False)
    outer.to_csv(out / "power_sequential_outer_metrics.csv", index=False)
    detail.to_csv(out / "power_sequential_outer_detail.csv", index=False)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
