"""Measurement-only FSR V3 escape/retention prior and variance study.

This study audits only the escape/retention family because it is an active
Event Clock V2 consumer.  It does not modify canonical FSR V3 publication or
Event Clock mechanics.

Native amount target
--------------------
For a directional ground-control observation with n inferred/landed ground
entries and y > 0 qualified control seconds, model the total positive control
amount as the sum of n Gamma entry durations:

    y ~ Gamma(shape=n*kappa, mean=n*mu_entry)

    log(mu_entry) = log(mu_population)
                    + controller_retention_effect
                    - bottom_escape_effect

Both fighter effects have zero-centered Normal population priors with shared
sigma.  Same-event evidence is delayed.  Population mean control/entry is also
strictly prior-date.

The amount model deliberately conditions on positive qualified control. Event
Clock already owns a separate control-occurrence hurdle, so zero-control bouts
are not reused here to avoid double-counting the hurdle signal.

Selection protocol
------------------
* 2020-2023: chronological calibration window for sigma and Gamma dispersion.
* 2024+: untouched holdout for the chosen prior/dispersion.
* posterior variance multiplier c is then measured on the holdout. c=0 is the
  posterior-mean plug-in; c=1 propagates the full inferred epistemic variance.
* the frozen V2 escape_prior_entries=5 plug-in is reported as a baseline.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log, pi
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import gammaln, logsumexp

from pipeline.fsr_v2.replay.engine import aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds

OUT_DIR = Path("data/diagnostics/fsr_v3_active_trait_audit/escape")
EVAL_START = pd.Timestamp("2020-01-01")
CAL_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
INITIAL_POP_SECONDS_PER_ENTRY = 60.0
V2_PRIOR_ENTRIES = 5.0

SIGMA_CANDIDATES = (0.15, 0.30, 0.45, 0.60, 0.80, 1.00)
KAPPA_CANDIDATES = (0.5, 1.0, 2.0, 4.0, 8.0)
C_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
GRID = np.linspace(-2.5, 2.5, 501)
GH_X, GH_W = hermgauss(21)
GH_LOG_W = np.log(GH_W / np.sqrt(pi))
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20260822


def _prior_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3plus"


def _normal_log_prior(grid: np.ndarray, sigma: float) -> np.ndarray:
    s = max(float(sigma), 1e-8)
    return -0.5 * (grid / s) ** 2 - log(s)


def _normalize(lp: np.ndarray) -> np.ndarray:
    z = lp - np.max(lp)
    w = np.exp(z)
    return w / w.sum()


def _moments(lp: np.ndarray, prior_lp: np.ndarray) -> tuple[float, float]:
    w = _normalize(prior_lp + lp)
    mean = float(np.sum(GRID * w))
    var = float(np.sum((GRID - mean) ** 2 * w))
    return mean, max(var, 0.0)


def _gamma_total_ll(y: float, entries: float, mean_entry, kappa: float):
    """Log likelihood for total duration from iid Gamma entry durations."""
    y = max(float(y), 1e-12)
    n = max(float(entries), 1e-12)
    k = max(float(kappa), 1e-8)
    mean = np.maximum(np.asarray(mean_entry, dtype=float), 1e-9)
    shape = n * k
    rate = k / mean
    return (
        shape * np.log(rate)
        + (shape - 1.0) * np.log(y)
        - rate * y
        - gammaln(shape)
    )


def _pair_predictive_ll(
    y: float,
    entries: float,
    mu_pop: float,
    d_mean: float,
    d_var: float,
    o_mean: float,
    o_var: float,
    kappa: float,
    c: float,
) -> float:
    diff_mean = float(d_mean - o_mean)
    diff_var = max(float(c), 0.0) * max(float(d_var + o_var), 0.0)
    if diff_var <= 1e-12:
        return float(_gamma_total_ll(y, entries, mu_pop * np.exp(diff_mean), kappa))
    draws = diff_mean + np.sqrt(2.0 * diff_var) * GH_X
    ll = _gamma_total_ll(y, entries, mu_pop * np.exp(draws), kappa)
    return float(logsumexp(GH_LOG_W + ll))


def _with_prior_fight_counts(fights: pd.DataFrame) -> pd.DataFrame:
    x = fights.copy().sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
    counts: dict[str, int] = {}
    prior = np.zeros(len(x), dtype=int)
    for _, idx in x.groupby("event_date", sort=True).groups.items():
        idx = list(idx)
        for i in idx:
            prior[i] = counts.get(str(x.at[i, "fighter_id"]), 0)
        for i in idx:
            fid = str(x.at[i, "fighter_id"])
            counts[fid] = counts.get(fid, 0) + 1
    x["prior_ufc_fights"] = prior
    return x


def build_observations() -> pd.DataFrame:
    """Create leakage-safe directional positive-control amount observations."""
    paired = build_paired_rounds()
    fights = _with_prior_fight_counts(aggregate_fights(paired))

    # The old V2 plug-in state is computed on all directional entry observations
    # first, including zero-duration entry cases, then positive amount rows are
    # selected for the native amount target.
    inflicted: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    suffered: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    pop_duration = 0.0
    pop_entries = 0.0
    rows: list[dict[str, object]] = []

    for date, batch in fights.groupby("event_date", sort=True):
        mu_pop = (
            pop_duration / pop_entries
            if pop_entries > 0.0
            else INITIAL_POP_SECONDS_PER_ENTRY
        )
        pending: list[tuple[str, str, float, float]] = []
        for r in batch.to_dict("records"):
            controller = str(r["fighter_id"])
            bottom = str(r["opponent_id"])
            y = float(r["qualified_control_inflicted_seconds"])
            n = float(r["ground_entries"])

            i_dur, i_ent = inflicted[controller]
            s_dur, s_ent = suffered[bottom]
            mu_inflicted = (i_dur + mu_pop * V2_PRIOR_ENTRIES) / (i_ent + V2_PRIOR_ENTRIES)
            mu_suffered = (s_dur + mu_pop * V2_PRIOR_ENTRIES) / (s_ent + V2_PRIOR_ENTRIES)
            v2_mean_entry = mu_inflicted * mu_suffered / max(mu_pop, 1e-9)

            if n > 0.0 and y > 0.0:
                rows.append({
                    "event_date": pd.Timestamp(date),
                    "year": int(pd.Timestamp(date).year),
                    "fight_id": str(r["fight_id"]),
                    "controller_id": controller,
                    "controller_name": r["fighter_name"],
                    "bottom_id": bottom,
                    "bottom_name": r["opponent_name"],
                    "controller_prior_ufc_fights": int(r["prior_ufc_fights"]),
                    "controller_prior_bucket": _prior_bucket(int(r["prior_ufc_fights"])),
                    "control_seconds": y,
                    "ground_entries": n,
                    "observed_seconds_per_entry": y / n,
                    "population_seconds_per_entry": float(mu_pop),
                    "v2_mean_seconds_per_entry": float(v2_mean_entry),
                })
            if n > 0.0:
                pending.append((controller, bottom, y, n))

        # Same-event delayed updates for both V2 baseline and population mean.
        for controller, bottom, y, n in pending:
            inflicted[controller][0] += y
            inflicted[controller][1] += n
            suffered[bottom][0] += y
            suffered[bottom][1] += n
            if y > 0.0:
                pop_duration += y
                pop_entries += n

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("escape study produced no positive-control observations")
    return out.sort_values(["event_date", "fight_id", "controller_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class CandidateResult:
    sigma: float
    kappa: float
    rows: pd.DataFrame


def replay_candidate(obs: pd.DataFrame, sigma: float, kappa: float, c: float = 1.0) -> CandidateResult:
    prior_lp = _normal_log_prior(GRID, sigma)
    zero = np.zeros_like(GRID)
    defense_evidence: dict[str, np.ndarray] = {}
    escape_evidence: dict[str, np.ndarray] = {}
    scored: list[dict[str, object]] = []

    for date, batch in obs.groupby("event_date", sort=True):
        pending_d: dict[str, np.ndarray] = {}
        pending_o: dict[str, np.ndarray] = {}
        for r in batch.to_dict("records"):
            controller = str(r["controller_id"])
            bottom = str(r["bottom_id"])
            d_lp = defense_evidence.get(controller, zero)
            o_lp = escape_evidence.get(bottom, zero)
            d_mean, d_var = _moments(d_lp, prior_lp)
            o_mean, o_var = _moments(o_lp, prior_lp)
            y = float(r["control_seconds"])
            n = float(r["ground_entries"])
            mu_pop = float(r["population_seconds_per_entry"])

            predictive_ll = _pair_predictive_ll(
                y, n, mu_pop, d_mean, d_var, o_mean, o_var, kappa, c
            )
            plugin_mean = mu_pop * np.exp(d_mean - o_mean)
            plugin_ll = float(_gamma_total_ll(y, n, plugin_mean, kappa))
            population_ll = float(_gamma_total_ll(y, n, mu_pop, kappa))
            v2_ll = float(_gamma_total_ll(y, n, r["v2_mean_seconds_per_entry"], kappa))

            row = dict(r)
            row.update({
                "sigma": float(sigma),
                "kappa": float(kappa),
                "variance_multiplier": float(c),
                "controller_defense_mean": d_mean,
                "controller_defense_sd": float(np.sqrt(d_var)),
                "bottom_escape_mean": o_mean,
                "bottom_escape_sd": float(np.sqrt(o_var)),
                "model_mean_seconds_per_entry": float(plugin_mean),
                "plugin_ll": plugin_ll,
                "predictive_ll": predictive_ll,
                "population_ll": population_ll,
                "v2_plugin_ll": v2_ll,
            })
            scored.append(row)

            # Conditional same-event-delayed posterior updates. Each side sees
            # the opponent's prefight posterior mean, never the current outcome.
            d_update = _gamma_total_ll(
                y, n, mu_pop * np.exp(GRID - o_mean), kappa
            )
            o_update = _gamma_total_ll(
                y, n, mu_pop * np.exp(d_mean - GRID), kappa
            )
            pending_d[controller] = pending_d.get(controller, 0.0) + d_update
            pending_o[bottom] = pending_o.get(bottom, 0.0) + o_update

        for fighter, update in pending_d.items():
            cur = defense_evidence.get(fighter)
            nxt = update if cur is None else cur + update
            defense_evidence[fighter] = nxt - np.max(nxt)
        for fighter, update in pending_o.items():
            cur = escape_evidence.get(fighter)
            nxt = update if cur is None else cur + update
            escape_evidence[fighter] = nxt - np.max(nxt)

    return CandidateResult(float(sigma), float(kappa), pd.DataFrame(scored))


def calibration_grid(obs: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    records = []
    for sigma in SIGMA_CANDIDATES:
        for kappa in KAPPA_CANDIDATES:
            result = replay_candidate(obs, sigma, kappa, c=1.0).rows
            cal = result[(result["event_date"] >= EVAL_START) & (result["event_date"] <= CAL_END)]
            records.append({
                "sigma": sigma,
                "kappa": kappa,
                "rows": len(cal),
                "fights": cal["fight_id"].nunique(),
                "predictive_ll": float(cal["predictive_ll"].sum()),
                "plugin_ll": float(cal["plugin_ll"].sum()),
                "population_ll": float(cal["population_ll"].sum()),
                "predictive_gain_vs_population": float((cal["predictive_ll"] - cal["population_ll"]).sum()),
                "plugin_gain_vs_population": float((cal["plugin_ll"] - cal["population_ll"]).sum()),
            })
    table = pd.DataFrame(records).sort_values(
        ["predictive_ll", "sigma", "kappa"], ascending=[False, True, True]
    ).reset_index(drop=True)
    best = table.iloc[0]
    return table, (float(best["sigma"]), float(best["kappa"]))


def variance_sweep(obs: pd.DataFrame, sigma: float, kappa: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    holdout_rows = []
    for c in C_CANDIDATES:
        scored = replay_candidate(obs, sigma, kappa, c=c).rows
        h = scored[scored["event_date"] >= HOLDOUT_START].copy()
        h["predictive_gain_vs_population"] = h["predictive_ll"] - h["population_ll"]
        h["predictive_gain_vs_v2"] = h["predictive_ll"] - h["v2_plugin_ll"]
        h["plugin_gain_vs_population"] = h["plugin_ll"] - h["population_ll"]
        h["plugin_gain_vs_v2"] = h["plugin_ll"] - h["v2_plugin_ll"]
        holdout_rows.append(h)
        summaries.append({
            "variance_multiplier": c,
            "rows": len(h),
            "fights": h["fight_id"].nunique(),
            "predictive_ll": float(h["predictive_ll"].sum()),
            "plugin_ll": float(h["plugin_ll"].sum()),
            "population_ll": float(h["population_ll"].sum()),
            "v2_plugin_ll": float(h["v2_plugin_ll"].sum()),
            "predictive_gain_vs_population": float(h["predictive_gain_vs_population"].sum()),
            "predictive_gain_vs_v2": float(h["predictive_gain_vs_v2"].sum()),
            "plugin_gain_vs_population": float(h["plugin_gain_vs_population"].sum()),
            "plugin_gain_vs_v2": float(h["plugin_gain_vs_v2"].sum()),
            "mae_seconds_per_entry": float(np.mean(np.abs(h["model_mean_seconds_per_entry"] - h["observed_seconds_per_entry"]))),
        })
    summary = pd.DataFrame(summaries).sort_values("predictive_ll", ascending=False).reset_index(drop=True)
    return summary, pd.concat(holdout_rows, ignore_index=True)


def _bootstrap_best(rows: pd.DataFrame, best_c: float) -> pd.DataFrame:
    x = rows[np.isclose(rows["variance_multiplier"], best_c)].copy()
    fight = x.groupby("fight_id", as_index=False).agg(
        gain_population=("predictive_gain_vs_population", "sum"),
        gain_v2=("predictive_gain_vs_v2", "sum"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(fight)
    if n == 0:
        return pd.DataFrame()
    draws = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    gp = fight["gain_population"].to_numpy(float)[draws].sum(axis=1)
    gv = fight["gain_v2"].to_numpy(float)[draws].sum(axis=1)
    return pd.DataFrame([{
        "variance_multiplier": best_c,
        "fights": n,
        "gain_vs_population": float(fight["gain_population"].sum()),
        "gain_vs_population_ci_2_5": float(np.quantile(gp, 0.025)),
        "gain_vs_population_ci_97_5": float(np.quantile(gp, 0.975)),
        "p_gain_vs_population_gt0": float(np.mean(gp > 0.0)),
        "gain_vs_v2": float(fight["gain_v2"].sum()),
        "gain_vs_v2_ci_2_5": float(np.quantile(gv, 0.025)),
        "gain_vs_v2_ci_97_5": float(np.quantile(gv, 0.975)),
        "p_gain_vs_v2_gt0": float(np.mean(gv > 0.0)),
    }])


def _bucket_summary(rows: pd.DataFrame, best_c: float) -> pd.DataFrame:
    x = rows[np.isclose(rows["variance_multiplier"], best_c)].copy()
    return (
        x.groupby("controller_prior_bucket", as_index=False)
        .agg(
            rows=("fight_id", "size"),
            fights=("fight_id", "nunique"),
            predictive_ll=("predictive_ll", "sum"),
            population_ll=("population_ll", "sum"),
            v2_plugin_ll=("v2_plugin_ll", "sum"),
            gain_vs_population=("predictive_gain_vs_population", "sum"),
            gain_vs_v2=("predictive_gain_vs_v2", "sum"),
            mean_controller_sd=("controller_defense_sd", "mean"),
            mean_bottom_sd=("bottom_escape_sd", "mean"),
        )
    )


def main() -> None:
    print("=" * 132)
    print("FSR V3 ACTIVE TRAIT AUDIT — ESCAPE / RETENTION — MEASUREMENT ONLY")
    print("=" * 132)
    print("Building qualified positive-control amount observations...")
    obs = build_observations()
    print(
        f"positive amount rows={len(obs):,} | fights={obs['fight_id'].nunique():,} | "
        f"range={obs['event_date'].min().date()}..{obs['event_date'].max().date()}"
    )

    print("Selecting population-prior sigma and Gamma observation dispersion on 2020-2023...")
    grid, (best_sigma, best_kappa) = calibration_grid(obs)
    print(grid.head(12).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"selected sigma={best_sigma:.3f} | kappa={best_kappa:.3f}")

    print("Scoring posterior variance multiplier on untouched 2024+ holdout...")
    variance, rows = variance_sweep(obs, best_sigma, best_kappa)
    print(variance.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    best_c = float(variance.iloc[0]["variance_multiplier"])
    boot = _bootstrap_best(rows, best_c)
    buckets = _bucket_summary(rows, best_c)

    print()
    print("SELECTED HOLDOUT RESULT")
    print(boot.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print("HOLDOUT BY CONTROLLER PRIOR UFC FIGHTS")
    print(buckets.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT_DIR / "escape_observations.csv", index=False)
    grid.to_csv(OUT_DIR / "escape_calibration_grid.csv", index=False)
    variance.to_csv(OUT_DIR / "escape_holdout_variance_sweep.csv", index=False)
    rows.to_csv(OUT_DIR / "escape_holdout_row_scores.csv", index=False)
    boot.to_csv(OUT_DIR / "escape_holdout_bootstrap.csv", index=False)
    buckets.to_csv(OUT_DIR / "escape_holdout_prior_buckets.csv", index=False)

    print()
    print(f"selected prior sigma={best_sigma:.3f}; selected observation kappa={best_kappa:.3f}; selected c={best_c:.2f}")
    print("DONE — measurement only; no canonical FSR or Event Clock values changed.")


if __name__ == "__main__":
    main()
