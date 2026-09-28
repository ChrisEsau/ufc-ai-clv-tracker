"""Research-only bantamweight uncertainty-compression A/B.

Tests whether path-level FSR epistemic uncertainty is responsible for favorite
probability compression. Uses the locked research mechanics candidate i10_b0
(fresh-power offset 10-t/12, KD sequence disabled) on the exact priced fights
from the completed dog-compression diagnostic.

Arms:
  current: canonical V3 epistemic sampling, including KD-resistance posterior draw
  means:   posterior means only for sampled V3 traits and KD resistance

No fitting, tuning, market inputs to simulation, or frozen mechanics changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit

DIVISION = "bantamweight"
INTERCEPT = 10.0
DENOMINATOR = 12.0
LOWER_CAP = -40.0
SEED = 20260823


def _install_i10_b0() -> None:
    seq.INTERCEPT = INTERCEPT
    seq.DENOMINATOR = DENOMINATOR
    seq.LOWER_CAP = LOWER_CAP
    seq.UPPER_CAP = INTERCEPT
    seq.ARMS = {"i10_b0": None}
    seq._MODE = "i10_b0"
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path


def _market_fights(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    f["fight_id"] = f["fight_id"].astype(str)
    return f.drop_duplicates("fight_id").copy()


def _metrics(summary: pd.DataFrame, market_fights: pd.DataFrame, arm: str, subset: str) -> dict:
    x = summary.merge(
        market_fights[["fight_id", "favorite_side", "market_favorite_fair_p", "favorite_won", "red_prior_ufc_fights", "blue_prior_ufc_fights"]],
        on="fight_id", how="inner", suffixes=("", "_market")
    )
    if subset == "high_evidence":
        x = x[(x["red_prior_ufc_fights_market"] >= 3) & (x["blue_prior_ufc_fights_market"] >= 3)].copy()
    if x.empty:
        return {"arm": arm, "subset": subset, "fights": 0}

    x["mc_favorite_p"] = np.where(x["favorite_side"].eq("red"), x["p_red_win"], 1.0 - x["p_red_win"])
    y = x["favorite_won"].astype(int).to_numpy()
    p = x["mc_favorite_p"].astype(float).to_numpy()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    return {
        "arm": arm,
        "subset": subset,
        "fights": len(x),
        "mean_market_favorite_fair_p": float(x["market_favorite_fair_p"].mean()),
        "mean_mc_favorite_p": float(x["mc_favorite_p"].mean()),
        "actual_favorite_win_rate": float(x["favorite_won"].mean()),
        "mean_compression_pp": float(100.0 * (x["market_favorite_fair_p"] - x["mc_favorite_p"]).mean()),
        "favorite_accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "favorite_auc": auc,
        "favorite_brier": float(np.mean((p - y) ** 2)),
        "favorite_logloss": float(-np.mean(y*np.log(np.clip(p,1e-9,1)) + (1-y)*np.log(np.clip(1-p,1e-9,1)))),
    }


def _bucket(summary: pd.DataFrame, market_fights: pd.DataFrame, arm: str) -> pd.DataFrame:
    x = summary.merge(market_fights[["fight_id","favorite_side","market_favorite_fair_p","favorite_won"]], on="fight_id", how="inner")
    x["mc_favorite_p"] = np.where(x["favorite_side"].eq("red"), x["p_red_win"], 1.0-x["p_red_win"])
    x["bucket"] = pd.cut(x["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.01], labels=["50-60","60-70","70-80","80-90","90+"], right=False)
    out = x.groupby("bucket", observed=True).agg(
        fights=("fight_id","size"),
        market_favorite_fair_p=("market_favorite_fair_p","mean"),
        mc_favorite_p=("mc_favorite_p","mean"),
        actual_favorite_win_rate=("favorite_won","mean"),
    ).reset_index()
    out["compression_pp"] = 100.0*(out["market_favorite_fair_p"]-out["mc_favorite_p"])
    out.insert(0,"arm",arm)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced-fights-path", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    market_fights = _market_fights(args.priced_fights_path)
    cohort, _ = wc_audit.select_cohort(DIVISION, 100)
    cohort["fight_id"] = cohort["fight_id"].astype(str)
    ids = set(market_fights["fight_id"])
    cohort = cohort[cohort["fight_id"].isin(ids)].reset_index(drop=True)
    if len(cohort) != len(ids):
        missing = sorted(ids - set(cohort["fight_id"]))
        raise RuntimeError(f"priced cohort mismatch: cohort={len(cohort)} ids={len(ids)} missing={missing}")

    _install_i10_b0()
    orig_init = canonical.initialize_path_matchup
    orig_kd = canonical.sample_kd_resistance_latent

    summaries = []
    metrics = []
    buckets = []

    # Current canonical epistemic sampling.
    current = canonical._simulate_c(cohort, args.paths, args.seed)
    current["arm"] = "current_epistemic"
    summaries.append(current)
    for subset in ("all", "high_evidence"):
        metrics.append(_metrics(current, market_fights, "current_epistemic", subset))
    buckets.append(_bucket(current, market_fights, "current_epistemic"))

    # Posterior means only: preserve identical base/detailed RNG; suppress only
    # the epistemic trait and KD-resistance draws.
    def init_means(red_row, blue_row, red_unc, blue_unc, *, rng, sample_epistemic=True):
        return orig_init(red_row, blue_row, red_unc, blue_unc, rng=rng, sample_epistemic=False)

    def kd_mean(row, rng):
        return float(row["pre_rating"])

    canonical.initialize_path_matchup = init_means
    canonical.sample_kd_resistance_latent = kd_mean
    means = canonical._simulate_c(cohort, args.paths, args.seed)
    means["arm"] = "posterior_means"
    summaries.append(means)
    for subset in ("all", "high_evidence"):
        metrics.append(_metrics(means, market_fights, "posterior_means", subset))
    buckets.append(_bucket(means, market_fights, "posterior_means"))

    canonical.initialize_path_matchup = orig_init
    canonical.sample_kd_resistance_latent = orig_kd

    m = pd.DataFrame(metrics)
    b = pd.concat(buckets, ignore_index=True)
    s = pd.concat(summaries, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_dir/"uncertainty_ab_metrics.csv", index=False)
    b.to_csv(args.out_dir/"uncertainty_ab_market_buckets.csv", index=False)
    s.to_csv(args.out_dir/"uncertainty_ab_fight_summaries.csv", index=False)

    print("BANTAMWEIGHT UNCERTAINTY COMPRESSION A/B")
    print(f"priced fights: {len(cohort)} | paths/fight/arm: {args.paths}")
    print("mechanics: i10_b0 fixed | only epistemic sampling differs")
    print("\nMETRICS")
    print(m.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMARKET-STRENGTH BUCKETS")
    print(b.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
