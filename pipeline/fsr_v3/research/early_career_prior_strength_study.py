"""Measurement-only FSR V3 early-career tendency prior-strength study.

Purpose
-------
Test whether the validated standing- and takedown-tendency population-prior
strengths move too quickly after a fighter's first UFC appearance.

This study does NOT modify production FSR configuration or Event Clock mechanics.
It reuses the locked NB2 tendency likelihood, population parameter sequence, and
same-event-delayed evidence semantics, then varies only the Gamma prior evidence
strength K.  Candidates are scored on the next fight's native count/exposure
outcome, stratified by 0 / 1 / 2 / 3+ prior UFC appearances.

Primary diagnostic bucket: exactly one prior UFC fight.
"""
from __future__ import annotations

from dataclasses import replace
from math import lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.math import nb2_log_likelihood, normalize_log_weights, weighted_mean_sd
from pipeline.fsr_v3.replay.rate_families import (
    RateFamilySpec,
    build_rate_fighter_fights,
    replay_tendency,
    standing_spec,
    takedown_spec,
)

OUT_DIR = Path("data/diagnostics/fsr_v3_early_career_prior")
EVAL_START = pd.Timestamp("2020-01-01")
K_MULTIPLIERS = (0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260821


def _bucket(prior: int) -> str:
    if prior <= 0:
        return "0"
    if prior == 1:
        return "1"
    if prior == 2:
        return "2"
    return "3plus"


def _with_prior_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count prior UFC appearances with same-date delayed semantics."""
    x = frame.copy().sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
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
    x["prior_bucket"] = [_bucket(v) for v in prior]
    return x


def _log_gamma_prior(grid: np.ndarray, mean: float, shape: float) -> np.ndarray:
    mean = max(float(mean), 1e-9)
    shape = max(float(shape), 1e-9)
    rate = shape / mean
    return shape * log(rate) - lgamma(shape) + (shape - 1.0) * np.log(grid) - rate * grid


def _candidate_rows(
    baseline: pd.DataFrame,
    spec: RateFamilySpec,
    multipliers: tuple[float, ...] = K_MULTIPLIERS,
) -> pd.DataFrame:
    """Score all K candidates while sharing the candidate-independent evidence state."""
    grid = np.linspace(spec.tendency_grid_min, spec.tendency_grid_max, spec.tendency_grid_points)
    candidates = np.asarray(multipliers, dtype=float)
    ks = candidates * float(spec.tendency_prior_seconds)
    states: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    source = _with_prior_counts(baseline)
    for event_date, batch in source.groupby("event_date", sort=True):
        pending: list[tuple[str, np.ndarray | None]] = []
        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            y = float(record["numerator"])
            exposure = float(record["denominator"])
            q_pop = float(record["population_rate_15m"])
            alpha = float(record["observation_alpha"])
            state_lp = states.get(fighter)

            observation_ll = None
            if exposure > 0.0:
                mu_grid = exposure / 900.0 * grid
                observation_ll = nb2_log_likelihood(y, mu_grid, alpha)

            population_mu = exposure / 900.0 * q_pop if exposure > 0 else np.nan
            population_ll = (
                float(nb2_log_likelihood(y, population_mu, alpha)) if exposure > 0 else np.nan
            )

            for multiplier, k_seconds in zip(candidates, ks):
                prior_shape = max(q_pop * float(k_seconds) / 900.0, 1e-9)
                lp = _log_gamma_prior(grid, q_pop, prior_shape)
                if state_lp is not None:
                    lp = lp + state_lp
                weights = normalize_log_weights(lp)
                pre_mean, pre_sd = weighted_mean_sd(grid, weights)

                if exposure > 0.0 and observation_ll is not None:
                    plugin_mu = exposure / 900.0 * pre_mean
                    plugin_ll = float(nb2_log_likelihood(y, plugin_mu, alpha))
                    predictive_ll = float(logsumexp(lp + observation_ll) - logsumexp(lp))
                else:
                    plugin_ll = np.nan
                    predictive_ll = np.nan

                rows.append({
                    "family": spec.name,
                    "event_date": record["event_date"],
                    "year": int(pd.Timestamp(record["event_date"]).year),
                    "fight_id": str(record["fight_id"]),
                    "fighter_id": fighter,
                    "fighter_name": record["fighter_name"],
                    "opponent_id": str(record["opponent_id"]),
                    "opponent_name": record["opponent_name"],
                    "prior_ufc_fights": int(record["prior_ufc_fights"]),
                    "prior_bucket": record["prior_bucket"],
                    "k_multiplier": float(multiplier),
                    "k_seconds": float(k_seconds),
                    "is_current_k": bool(np.isclose(multiplier, 1.0)),
                    "pre_rating": float(pre_mean),
                    "pre_posterior_sd": float(pre_sd),
                    "observed_count": y,
                    "exposure_seconds": exposure,
                    "observed_rate_15m": y / exposure * 900.0 if exposure > 0 else np.nan,
                    "population_rate_15m": q_pop,
                    "observation_alpha": alpha,
                    "plugin_log_likelihood": plugin_ll,
                    "posterior_predictive_log_likelihood": predictive_ll,
                    "population_log_likelihood": population_ll,
                })
            pending.append((fighter, observation_ll))

        # Same-event delayed update. Evidence likelihood is independent of K.
        for fighter, observation_ll in pending:
            if observation_ll is None:
                continue
            if fighter in states:
                states[fighter] = states[fighter] + observation_ll
                states[fighter] -= np.max(states[fighter])
            else:
                states[fighter] = observation_ll - np.max(observation_ll)

    return pd.DataFrame(rows)


def _summaries(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[(rows["event_date"] >= EVAL_START) & rows["exposure_seconds"].gt(0)].copy()
    current = x[x["is_current_k"]][
        ["family", "fight_id", "fighter_id", "plugin_log_likelihood", "posterior_predictive_log_likelihood"]
    ].rename(columns={
        "plugin_log_likelihood": "current_plugin_ll",
        "posterior_predictive_log_likelihood": "current_predictive_ll",
    })
    x = x.merge(current, on=["family", "fight_id", "fighter_id"], how="left", validate="many_to_one")
    x["plugin_delta_vs_current"] = x["plugin_log_likelihood"] - x["current_plugin_ll"]
    x["predictive_delta_vs_current"] = x["posterior_predictive_log_likelihood"] - x["current_predictive_ll"]
    x["plugin_gain_vs_population"] = x["plugin_log_likelihood"] - x["population_log_likelihood"]
    x["predictive_gain_vs_population"] = x["posterior_predictive_log_likelihood"] - x["population_log_likelihood"]

    pieces = []
    for bucket in ("ALL", "0", "1", "2", "3plus"):
        g = x if bucket == "ALL" else x[x["prior_bucket"] == bucket]
        agg = (
            g.groupby(["family", "k_multiplier", "k_seconds"], as_index=False)
            .agg(
                rows=("fight_id", "size"),
                fights=("fight_id", "nunique"),
                plugin_ll=("plugin_log_likelihood", "sum"),
                predictive_ll=("posterior_predictive_log_likelihood", "sum"),
                population_ll=("population_log_likelihood", "sum"),
                plugin_delta_vs_current=("plugin_delta_vs_current", "sum"),
                predictive_delta_vs_current=("predictive_delta_vs_current", "sum"),
                plugin_gain_vs_population=("plugin_gain_vs_population", "sum"),
                predictive_gain_vs_population=("predictive_gain_vs_population", "sum"),
                mean_pre_rating=("pre_rating", "mean"),
                mean_posterior_sd=("pre_posterior_sd", "mean"),
            )
        )
        agg["prior_bucket"] = bucket
        pieces.append(agg)
    return pd.concat(pieces, ignore_index=True)


def _annual(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[(rows["event_date"] >= EVAL_START) & rows["exposure_seconds"].gt(0) & rows["prior_bucket"].eq("1")].copy()
    cur = x[x["is_current_k"]][
        ["family", "fight_id", "fighter_id", "posterior_predictive_log_likelihood", "plugin_log_likelihood"]
    ].rename(columns={
        "posterior_predictive_log_likelihood": "current_predictive_ll",
        "plugin_log_likelihood": "current_plugin_ll",
    })
    x = x.merge(cur, on=["family", "fight_id", "fighter_id"], validate="many_to_one")
    x["predictive_delta_vs_current"] = x["posterior_predictive_log_likelihood"] - x["current_predictive_ll"]
    x["plugin_delta_vs_current"] = x["plugin_log_likelihood"] - x["current_plugin_ll"]
    return (
        x.groupby(["family", "k_multiplier", "k_seconds", "year"], as_index=False)
        .agg(
            rows=("fight_id", "size"),
            fights=("fight_id", "nunique"),
            predictive_delta_vs_current=("predictive_delta_vs_current", "sum"),
            plugin_delta_vs_current=("plugin_delta_vs_current", "sum"),
        )
    )


def _bootstrap(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[(rows["event_date"] >= EVAL_START) & rows["exposure_seconds"].gt(0) & rows["prior_bucket"].eq("1")].copy()
    cur = x[x["is_current_k"]][
        ["family", "fight_id", "fighter_id", "posterior_predictive_log_likelihood", "plugin_log_likelihood"]
    ].rename(columns={
        "posterior_predictive_log_likelihood": "current_predictive_ll",
        "plugin_log_likelihood": "current_plugin_ll",
    })
    x = x.merge(cur, on=["family", "fight_id", "fighter_id"], validate="many_to_one")
    x["predictive_delta"] = x["posterior_predictive_log_likelihood"] - x["current_predictive_ll"]
    x["plugin_delta"] = x["plugin_log_likelihood"] - x["current_plugin_ll"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    for (family, multiplier, k_seconds), g in x.groupby(["family", "k_multiplier", "k_seconds"], sort=False):
        fight = g.groupby("fight_id", as_index=False).agg(
            predictive_delta=("predictive_delta", "sum"),
            plugin_delta=("plugin_delta", "sum"),
        )
        vals_pred = fight["predictive_delta"].to_numpy(float)
        vals_plugin = fight["plugin_delta"].to_numpy(float)
        n = len(fight)
        if n == 0:
            continue
        draws = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
        boot_pred = vals_pred[draws].sum(axis=1)
        boot_plugin = vals_plugin[draws].sum(axis=1)
        records.append({
            "family": family,
            "k_multiplier": float(multiplier),
            "k_seconds": float(k_seconds),
            "fights": n,
            "predictive_delta_vs_current": float(vals_pred.sum()),
            "predictive_ci_2_5": float(np.quantile(boot_pred, 0.025)),
            "predictive_ci_97_5": float(np.quantile(boot_pred, 0.975)),
            "plugin_delta_vs_current": float(vals_plugin.sum()),
            "plugin_ci_2_5": float(np.quantile(boot_plugin, 0.025)),
            "plugin_ci_97_5": float(np.quantile(boot_plugin, 0.975)),
        })
    return pd.DataFrame(records)


def _print_headline(summary: pd.DataFrame, bootstrap: pd.DataFrame, annual: pd.DataFrame) -> None:
    print("=" * 132)
    print("FSR V3 EARLY-CAREER TENDENCY PRIOR-STRENGTH STUDY — MEASUREMENT ONLY")
    print("=" * 132)
    print(f"evaluation start: {EVAL_START.date()} | bootstrap reps: {BOOTSTRAP_REPS}")
    print(f"K multipliers: {', '.join(f'{x:g}x' for x in K_MULTIPLIERS)}")
    print()
    for family in ("standing_striking", "takedown"):
        one = summary[(summary["family"] == family) & (summary["prior_bucket"] == "1")].sort_values(
            "predictive_delta_vs_current", ascending=False
        )
        print(f"{family.upper()} — EXACTLY ONE PRIOR UFC FIGHT")
        print(one[[
            "k_multiplier", "k_seconds", "rows", "predictive_ll", "predictive_delta_vs_current",
            "plugin_delta_vs_current", "predictive_gain_vs_population", "mean_posterior_sd",
        ]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()
        best = one.iloc[0]
        b = bootstrap[(bootstrap["family"] == family) & np.isclose(bootstrap["k_multiplier"], best["k_multiplier"])].iloc[0]
        print(
            f"best 1-prior predictive K: {best['k_seconds']:.2f}s ({best['k_multiplier']:.2f}x current) | "
            f"delta vs current={best['predictive_delta_vs_current']:+.3f} | "
            f"95% fight-bootstrap CI [{b['predictive_ci_2_5']:+.3f}, {b['predictive_ci_97_5']:+.3f}]"
        )
        annual_best = annual[(annual["family"] == family) & np.isclose(annual["k_multiplier"], best["k_multiplier"])]
        if not annual_best.empty:
            improved = int((annual_best["predictive_delta_vs_current"] > 0).sum())
            print(f"annual 1-prior periods improved: {improved}/{len(annual_best)}")
        print()


def main() -> None:
    config = FSRV3Config()
    all_rows = []
    for family, base_spec in (
        ("standing_striking", standing_spec(config)),
        ("takedown", takedown_spec(config)),
    ):
        print(f"Building {family} native fighter-fight observations...")
        fights = build_rate_fighter_fights(base_spec)
        print(f"Replaying locked current {family} state to recover leakage-safe population sequence...")
        baseline = replay_tendency(fights, base_spec)
        print(f"Scoring K candidates around current K={base_spec.tendency_prior_seconds:.2f}s...")
        all_rows.append(_candidate_rows(baseline, base_spec))

    rows = pd.concat(all_rows, ignore_index=True)
    summary = _summaries(rows)
    annual = _annual(rows)
    bootstrap = _bootstrap(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "early_career_prior_row_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "early_career_prior_summary.csv", index=False)
    annual.to_csv(OUT_DIR / "early_career_prior_annual_one_prior.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "early_career_prior_bootstrap_one_prior.csv", index=False)

    _print_headline(summary, bootstrap, annual)
    print("Outputs:")
    for path in sorted(OUT_DIR.glob("*.csv")):
        print(path)
    print("DONE — no production FSR or Event Clock settings changed.")


if __name__ == "__main__":
    main()
