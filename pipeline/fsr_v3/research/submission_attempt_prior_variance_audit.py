"""Measurement-only prior/variance audit for live submission attempt tendency/suppression."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

from pipeline.fsr_v2.replay.engine import SUBMISSION_TENDENCY_INITIAL_RATE
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay.math import nb2_log_likelihood
from pipeline.fsr_v3.replay.rate_families import (
    RateFamilySpec, build_rate_fighter_fights, replay_tendency, replay_suppression,
)

SEED = 20260822
VALIDATION_START = pd.Timestamp("2022-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
DEFAULT_OUT = Path("data/diagnostics/fsr_v3/active_trait_audit/submission_attempts")
K_CANDIDATES = (900.0, 1800.0, 2700.0, 3600.0, 5400.0)
SHAPE_CANDIDATES = (1.0, 2.0, 3.0, 5.0, 8.0)
C_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
GH_X, GH_W = hermgauss(15)
GH_W = GH_W / np.sqrt(np.pi)


def _bucket(n):
    n = int(n)
    return "0" if n <= 0 else "1" if n == 1 else "2" if n == 2 else "3plus"


def spec(k, shape):
    return RateFamilySpec(
        name="submission_attempt",
        tendency_trait="submission_tendency_v3_candidate",
        suppression_trait="submission_suppression_v3_candidate",
        numerator_column="effective_submission_attempts",
        exposure_column="round_elapsed_seconds",
        tendency_prior_seconds=float(k),
        tendency_initial_population_rate_15m=float(SUBMISSION_TENDENCY_INITIAL_RATE * 900.0),
        tendency_initial_alpha=1.0,
        tendency_grid_min=0.001,
        tendency_grid_max=12.0,
        tendency_grid_points=1200,
        tendency_variance_multiplier=1.0,
        suppression_prior_shape=float(shape),
        suppression_initial_population=1.0,
        suppression_initial_alpha=1.0,
        suppression_grid_min=0.03,
        suppression_grid_max=5.0,
        suppression_grid_points=1200,
        suppression_variance_multiplier=1.0,
    )


def _lognormal_variance(mean, sd):
    mean = max(float(mean), 1e-12)
    sd = max(float(sd), 0.0)
    return float(np.log1p((sd / mean) ** 2))


def _row_nb2_ll(y, mu, alpha):
    return float(nb2_log_likelihood(float(y), float(mu), float(alpha)))


def predictive_ll(y, exposure, q, qsd, s, ssd, alpha, c):
    base_mean = exposure / 900.0 * q * s
    if c <= 0:
        return _row_nb2_ll(y, base_mean, alpha)
    v = c * (_lognormal_variance(q, qsd) + _lognormal_variance(s, ssd))
    if v <= 1e-14:
        return _row_nb2_ll(y, base_mean, alpha)
    log_mu = np.log(max(base_mean, 1e-12)) - 0.5 * v
    means = np.exp(log_mu + np.sqrt(2.0 * v) * GH_X)
    lls = nb2_log_likelihood(float(y), means, float(alpha))
    m = float(np.max(lls))
    return float(m + np.log(np.sum(GH_W * np.exp(lls - m))))


def add_prior_counts(frame):
    apps = frame[["event_date", "fighter_id"]].drop_duplicates()
    c = apps.groupby(["fighter_id", "event_date"], as_index=False).size().sort_values(["fighter_id", "event_date"])
    c["prior_ufc_fights"] = c.groupby("fighter_id")["size"].cumsum() - c["size"]
    out = frame.merge(c[["fighter_id", "event_date", "prior_ufc_fights"]], on=["fighter_id", "event_date"], how="left")
    out["prior_bucket"] = out["prior_ufc_fights"].map(_bucket)
    return out


def score_candidate(tendency, suppression, fsr):
    # Suppression rows are defender-centric: fighter_id is the defender whose
    # multiplier applies to the tendency row's opponent. Select only that key
    # before renaming so the original attacker opponent_id is never duplicated.
    sup = suppression[[
        "event_date", "fight_id", "fighter_id", "pre_rating",
        "pre_posterior_sd", "population_multiplier",
    ]].rename(columns={
        "fighter_id": "opponent_id", "pre_rating": "suppression_mean",
        "pre_posterior_sd": "suppression_sd", "population_multiplier": "suppression_population",
    })
    x = tendency.merge(sup, on=["event_date", "fight_id", "opponent_id"], how="inner", validate="one_to_one")
    legacy_self = fsr[["event_date", "fight_id", "fighter_id", "submission_tendency"]].copy()
    legacy_opp = fsr[["event_date", "fight_id", "fighter_id", "submission_suppression"]].rename(
        columns={"fighter_id": "opponent_id", "submission_suppression": "legacy_suppression"}
    )
    x = x.merge(legacy_self, on=["event_date", "fight_id", "fighter_id"], how="left", validate="one_to_one")
    x = x.merge(legacy_opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")
    x = add_prior_counts(x)
    x["new_mean"] = x["denominator"] / 900.0 * x["pre_rating"] * x["suppression_mean"]
    x["population_mean"] = x["denominator"] / 900.0 * x["population_rate_15m"] * x["suppression_population"]
    x["legacy_mean"] = x["denominator"] * x["submission_tendency"] * x["legacy_suppression"]

    # The shared NB2 helper intentionally takes one scalar alpha. Observation
    # dispersion varies by replay date, so score each row with its own scalar.
    x["population_ll"] = [
        _row_nb2_ll(y, mu, a)
        for y, mu, a in zip(x["numerator"], x["population_mean"], x["observation_alpha"])
    ]
    x["legacy_ll"] = [
        _row_nb2_ll(y, mu, a)
        for y, mu, a in zip(x["numerator"], x["legacy_mean"], x["observation_alpha"])
    ]
    x["plugin_ll"] = [
        _row_nb2_ll(y, mu, a)
        for y, mu, a in zip(x["numerator"], x["new_mean"], x["observation_alpha"])
    ]
    x["plugin_abs_error"] = (x["numerator"] - x["new_mean"]).abs()
    x["legacy_abs_error"] = (x["numerator"] - x["legacy_mean"]).abs()
    for c in C_CANDIDATES:
        x[f"predictive_ll_c_{c:g}"] = [
            predictive_ll(y, e, q, qsd, s, ssd, a, c)
            for y, e, q, qsd, s, ssd, a in zip(
                x.numerator, x.denominator, x.pre_rating, x.pre_posterior_sd,
                x.suppression_mean, x.suppression_sd, x.observation_alpha,
            )
        ]
    return x


def window(frame, start, end, col):
    x = frame[frame.event_date >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.event_date < pd.Timestamp(end)]
    return {"rows": len(x), "fights": x.fight_id.nunique(), "total_ll": x[col].sum(), "mean_ll": x[col].mean(), "mae": x.plugin_abs_error.mean()}


def bootstrap(frame, a, b, draws, seed):
    d = frame.groupby("fight_id")[[a, b]].sum()
    diff = (d[a] - d[b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    sims = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(draws)])
    return diff.sum(), np.quantile(sims, .025), np.quantile(sims, .975), np.mean(sims > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired = build_paired_rounds()
    fsr = pd.read_parquet(
        FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
        columns=["event_date", "fight_id", "fighter_id", "submission_tendency", "submission_suppression"],
    ).copy()
    fsr.event_date = pd.to_datetime(fsr.event_date, errors="raise").dt.normalize()
    fsr.fight_id = fsr.fight_id.astype(str)
    fsr.fighter_id = fsr.fighter_id.astype(str)
    results = []
    candidate_frames = {}
    tendency_cache = {}
    for k in K_CANDIDATES:
        sp = spec(k, SHAPE_CANDIDATES[0])
        fights = build_rate_fighter_fights(sp, paired_rounds=paired)
        tendency_cache[k] = replay_tendency(fights, sp)
        for shape in SHAPE_CANDIDATES:
            sp = spec(k, shape)
            suppression = replay_suppression(tendency_cache[k], sp)
            scored = score_candidate(tendency_cache[k], suppression, fsr)
            candidate_frames[(k, shape)] = scored
            results.append({"k_seconds": k, "suppression_shape": shape, **window(scored, VALIDATION_START, HOLDOUT_START, "predictive_ll_c_1")})
    sweep = pd.DataFrame(results)
    best = sweep.sort_values(["total_ll", "k_seconds", "suppression_shape"], ascending=[False, True, True]).iloc[0]
    k = float(best.k_seconds)
    shape = float(best.suppression_shape)
    selected = candidate_frames[(k, shape)]
    cs = []
    for c in C_CANDIDATES:
        for label, start, end in (
            ("validation_2022_2023", VALIDATION_START, HOLDOUT_START),
            ("holdout_2024plus", HOLDOUT_START, None),
        ):
            cs.append({"c": c, "window": label, **window(selected, start, end, f"predictive_ll_c_{c:g}")})
    cs = pd.DataFrame(cs)
    best_c = float(cs[cs.window.eq("validation_2022_2023")].sort_values(["total_ll", "c"], ascending=[False, True]).iloc[0].c)
    hold = selected[selected.event_date >= HOLDOUT_START].copy()
    hold["selected_ll"] = hold[f"predictive_ll_c_{best_c:g}"]
    bp = bootstrap(hold, "selected_ll", "population_ll", args.bootstrap_draws, SEED)
    bl = bootstrap(hold, "selected_ll", "legacy_ll", args.bootstrap_draws, SEED + 1)
    buckets = hold.groupby("prior_bucket").apply(lambda x: pd.Series({
        "rows": len(x), "fights": x.fight_id.nunique(),
        "delta_ll_vs_population": (x.selected_ll - x.population_ll).sum(),
        "delta_ll_vs_legacy": (x.selected_ll - x.legacy_ll).sum(),
        "plugin_mae": x.plugin_abs_error.mean(), "legacy_mae": x.legacy_abs_error.mean(),
    }), include_groups=False).reset_index()
    sweep.to_csv(args.output_dir / "prior_sweep.csv", index=False)
    cs.to_csv(args.output_dir / "variance_multiplier_sweep.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_rows.csv", index=False)
    buckets.to_csv(args.output_dir / "holdout_prior_buckets.csv", index=False)
    print("=" * 120)
    print("FSR V3 ACTIVE TRAIT AUDIT — SUBMISSION ATTEMPTS")
    print("=" * 120)
    print("TOP VALIDATION PRIOR SETTINGS")
    print(sweep.sort_values("total_ll", ascending=False).head(12).to_string(index=False))
    print(f"selected K={k:g}s suppression_shape={shape:g}")
    print("EPISTEMIC C")
    print(cs.to_string(index=False))
    print(f"selected c={best_c:g}")
    print(f"HOLDOUT vs population LL={bp[0]:+.3f} CI[{bp[1]:+.3f},{bp[2]:+.3f}] P>0={bp[3]:.3f}")
    print(f"HOLDOUT vs inherited V2 submission traits LL={bl[0]:+.3f} CI[{bl[1]:+.3f},{bl[2]:+.3f}] P>0={bl[3]:.3f}")
    print(buckets.to_string(index=False))
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
