"""Measurement-only canonical C audit for one UFC weight class.

This diagnostic does not tune parameters, apply weight-class overrides, or change
Event Clock mechanics. It evaluates only post-training-cutoff fights so the
current direct inference model is assessed out of sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import CUTOFF
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import METHODS, normalize_method
from pipeline.simulation.event_clock_mc_v2.diagnostics.canonical_c_validation import _simulate_c
from pipeline.simulation.event_clock_mc_v2.diagnostics.method_market_abc_validation import (
    _fight_bucket,
    _prior_ufc_counts,
)


def select_cohort(division: str, target_n: int) -> tuple[pd.DataFrame, int]:
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
    master["division_norm"] = master["division"].astype(str).str.strip().str.lower()

    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    counts = fsr.groupby("fight_id").size()
    valid_fsr_ids = set(counts[counts == 2].index.astype(str))

    rows = []
    cutoff = pd.Timestamp(CUTOFF).normalize()
    division_norm = division.strip().lower()
    for _, row in master.iterrows():
        method = normalize_method(row["method"])
        ok = (
            row["division_norm"] == division_norm
            and pd.notna(row["event_date"])
            and row["event_date"] > cutoff
            and row["winner"] in {row["r_name"], row["b_name"]}
            and method in METHODS
            and pd.notna(row["total_rounds"])
            and int(row["total_rounds"]) in {3, 5}
            and pd.notna(row["match_time_sec"])
            and str(row["fight_id"]) in valid_fsr_ids
        )
        if ok:
            rows.append(row)

    eligible = pd.DataFrame(rows)
    if eligible.empty:
        raise RuntimeError(f"No eligible post-cutoff fights for division={division!r}")

    cohort = (
        eligible.sort_values(["event_date", "fight_id"], ascending=[False, False])
        .head(int(target_n))
        .sort_values(["event_date", "fight_id"])
        .reset_index(drop=True)
    )

    prior = _prior_ufc_counts(master)
    cohort["red_prior_ufc_fights"] = [
        prior.get((str(r.fight_id), str(r.r_id)), 0) for r in cohort.itertuples(index=False)
    ]
    cohort["blue_prior_ufc_fights"] = [
        prior.get((str(r.fight_id), str(r.b_id)), 0) for r in cohort.itertuples(index=False)
    ]
    cohort["fight_evidence_bucket"] = [
        _fight_bucket(rp, bp)
        for rp, bp in zip(cohort["red_prior_ufc_fights"], cohort["blue_prior_ufc_fights"])
    ]
    return cohort, len(eligible)


def audit(summary: pd.DataFrame, cohort: pd.DataFrame, division: str, paths: int) -> dict[str, pd.DataFrame]:
    y_red = summary["actual_winner"].eq("red").astype(float).to_numpy()
    p_red = summary["p_red_win"].to_numpy(float)
    actual_winner_p = np.where(y_red > 0.5, p_red, 1.0 - p_red)

    method_cols = {"DEC": "p_fight_dec", "KO_TKO": "p_fight_ko_tko", "SUB": "p_fight_sub"}
    method_loss = []
    method_brier_rows = []
    method_order = tuple(method_cols)
    for row in summary.itertuples(index=False):
        probs = np.array([getattr(row, method_cols[m]) for m in method_order], dtype=float)
        yi = np.array([1.0 if row.actual_method == m else 0.0 for m in method_order])
        method_loss.append(-np.log(max(probs[method_order.index(row.actual_method)], 1e-9)))
        method_brier_rows.append(float(np.sum((probs - yi) ** 2)))

    headline = pd.DataFrame([{
        "division": division,
        "n_fights": len(summary),
        "paths_per_fight": int(paths),
        "training_cutoff_exclusive": str(pd.Timestamp(CUTOFF).date()),
        "first_event_date": str(cohort["event_date"].min().date()),
        "last_event_date": str(cohort["event_date"].max().date()),
        "ml_accuracy": float(summary["ml_correct"].mean()),
        "ml_brier": float(np.mean((p_red - y_red) ** 2)),
        "ml_logloss": float(-np.mean(np.log(np.clip(actual_winner_p, 1e-9, 1.0)))),
        "mean_actual_winner_probability": float(actual_winner_p.mean()),
        "method_accuracy": float(summary["method_correct"].mean()),
        "method_brier": float(np.mean(method_brier_rows)),
        "method_logloss": float(np.mean(method_loss)),
    }])

    method_share = pd.DataFrame([
        {
            "method": method,
            "actual_share": float(summary["actual_method"].eq(method).mean()),
            "simulated_share": float(summary[col].mean()),
            "bias_sim_minus_actual": float(
                summary[col].mean() - summary["actual_method"].eq(method).mean()
            ),
        }
        for method, col in method_cols.items()
    ])

    specs = [
        ("sig_attempted", "sim_red_sig_attempted", "sim_blue_sig_attempted", "hist_red_sig_attempted", "hist_blue_sig_attempted"),
        ("sig_landed", "sim_red_sig_landed", "sim_blue_sig_landed", "hist_red_sig_landed", "hist_blue_sig_landed"),
        ("td_attempted", "sim_red_td_attempted", "sim_blue_td_attempted", "hist_red_td_attempted", "hist_blue_td_attempted"),
        ("td_landed", "sim_red_td_landed", "sim_blue_td_landed", "hist_red_td_landed", "hist_blue_td_landed"),
        ("knockdowns", "sim_red_kd", "sim_blue_kd", "hist_red_kd", "hist_blue_kd"),
        ("sub_attempts", "sim_red_sub_attempts", "sim_blue_sub_attempts", "hist_red_sub_attempts", "hist_blue_sub_attempts"),
        ("control_seconds", "sim_red_control_seconds", "sim_blue_control_seconds", "hist_red_control_seconds", "hist_blue_control_seconds"),
    ]
    mechanics_rows = []
    for name, sr, sb, hr, hb in specs:
        simulated = float((summary[sr] + summary[sb]).mean())
        historical = float((summary[hr] + summary[hb]).mean())
        mechanics_rows.append({
            "metric": name,
            "historical_mean_per_fight": historical,
            "simulated_mean_per_fight": simulated,
            "absolute_bias": simulated - historical,
            "relative_bias": ((simulated - historical) / historical if abs(historical) > 1e-12 else np.nan),
        })
    historical = float(summary["actual_elapsed"].mean())
    simulated = float(summary["sim_mean_elapsed"].mean())
    mechanics_rows.append({
        "metric": "elapsed_seconds",
        "historical_mean_per_fight": historical,
        "simulated_mean_per_fight": simulated,
        "absolute_bias": simulated - historical,
        "relative_bias": (simulated - historical) / historical,
    })
    mechanics = pd.DataFrame(mechanics_rows)

    rng = np.random.default_rng(2026082301)
    n = len(summary)
    bootstrap_rows = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        sy = y_red[idx]
        sp = p_red[idx]
        swp = np.where(sy > 0.5, sp, 1.0 - sp)
        sub = summary.iloc[idx]
        rec = {
            "ml_brier": float(np.mean((sp - sy) ** 2)),
            "ml_logloss": float(-np.mean(np.log(np.clip(swp, 1e-9, 1.0)))),
        }
        for method, col in method_cols.items():
            rec[f"{method}_bias"] = float(
                sub[col].mean() - sub["actual_method"].eq(method).mean()
            )
        bootstrap_rows.append(rec)
    bootstrap_raw = pd.DataFrame(bootstrap_rows)
    bootstrap_ci = pd.DataFrame([
        {
            "metric": col,
            "ci_2_5": float(bootstrap_raw[col].quantile(0.025)),
            "ci_97_5": float(bootstrap_raw[col].quantile(0.975)),
        }
        for col in bootstrap_raw.columns
    ])

    return {
        "headline": headline,
        "method_share": method_share,
        "mechanics": mechanics,
        "bootstrap_ci": bootstrap_ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--division", required=True)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort, eligible_count = select_cohort(args.division, args.target_n)
    print("=" * 120)
    print(f"EVENT CLOCK MC V2 — {args.division.upper()} WEIGHT-CLASS AUDIT")
    print("=" * 120)
    print(
        f"training cutoff: {pd.Timestamp(CUTOFF).date()} | eligible post-cutoff: {eligible_count} | "
        f"selected: {len(cohort)} | paths/fight: {args.paths}"
    )
    print(f"date range: {cohort['event_date'].min().date()} through {cohort['event_date'].max().date()}")
    print("NO TUNING / NO OVERRIDES / CURRENT CANONICAL C ONLY")

    summary = _simulate_c(cohort, args.paths, args.seed)
    outputs = audit(summary, cohort, args.division, args.paths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.out_dir / "cohort.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)

    print("\nHEADLINE")
    print(outputs["headline"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMETHOD CALIBRATION")
    print(outputs["method_share"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMECHANICS")
    print(outputs["mechanics"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBOOTSTRAP 95% CI")
    print(outputs["bootstrap_ci"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
