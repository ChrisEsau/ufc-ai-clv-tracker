"""Full men's A/B validation of the KD-triggered finishing-sequence candidate.

Research-only comparison:
  control : validated continuous consequence curve only
  seq_b3  : same curve plus one post-KD finishing-sequence resolver at +3 logit

No FSR refit, no strike-budget/cadence change, no event-time change, and no
frozen V1 source modification. Uses all eligible fights in the eight audited
men's divisions and common random seeds across arms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import shared_power_decay_grid as shared
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit

ARMS = ("control", "seq_b3")


def _install_summary_wrapper() -> None:
    original = canonical.summarize_fight

    def wrapped(fight_id, pair, rows, master_row):
        out = original(fight_id, pair, rows, master_row)
        p = pd.DataFrame(rows)
        nondec = p["method"].ne("DEC")
        for threshold in (300, 600, 900):
            out[f"p_nondec_by_{threshold}"] = float((nondec & p["elapsed"].le(threshold)).mean())
        for round_no in (1, 2, 3):
            out[f"p_ko_r{round_no}"] = float(
                (p["method"].eq("KO_TKO") & p["finish_round"].eq(round_no)).mean()
            )
        out["mean_sequence_opportunities"] = float(p["sequence_opportunities"].mean())
        out["p_sequence_finish"] = float(p["sequence_finishes"].mean())
        return out

    canonical.summarize_fight = wrapped


def _historical(target_n: int):
    cohorts = []
    for division in shared.MEN_DIVISIONS:
        c, _ = wc_audit.select_cohort(division, target_n)
        c = c.copy()
        c["division_audit"] = division
        cohorts.append(c)
    cohort = pd.concat(cohorts, ignore_index=True)
    method = cohort["method"].map(wc_audit.normalize_method)
    elapsed = pd.to_numeric(cohort["match_time_sec"], errors="raise")
    finish_round = pd.to_numeric(cohort["finish_round"], errors="coerce")
    targets = {
        "ko_share": float(method.eq("KO_TKO").mean()),
        "mean_elapsed": float(elapsed.mean()),
    }
    for t in (300, 600, 900):
        targets[f"nondec_by_{t}"] = float(((method != "DEC") & elapsed.le(t)).mean())
    for r in (1, 2, 3):
        targets[f"ko_r{r}"] = float((method.eq("KO_TKO") & finish_round.eq(r)).mean())
    return cohort, targets


def _simulate_arm(target_n: int, paths: int, seed: int, arm: str) -> pd.DataFrame:
    seq._MODE = arm
    seq.runner.simulate_detailed_path = seq.sequence_simulate_detailed_path
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path
    frames = []
    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort, eligible = wc_audit.select_cohort(division, target_n)
        print(f"ARM {arm} | {division} | eligible={eligible} selected={len(cohort)} paths={paths}")
        s = canonical._simulate_c(cohort, paths, seed + i * 100_000_000)
        s["division"] = division
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _metrics(summary: pd.DataFrame, hist: dict, paths: int, arm: str) -> dict:
    y = summary["actual_winner"].eq("red").astype(float).to_numpy()
    p = summary["p_red_win"].to_numpy(float)
    winner_p = np.where(y > 0.5, p, 1.0 - p)
    rec = {
        "arm": arm,
        "n_fights": int(len(summary)),
        "paths_per_fight": int(paths),
        "ml_accuracy": float(summary["ml_correct"].mean()),
        "ml_brier": float(np.mean((p - y) ** 2)),
        "ml_logloss": float(-np.mean(np.log(np.clip(winner_p, 1e-9, 1.0)))),
        "mean_actual_winner_probability": float(winner_p.mean()),
        "method_accuracy": float(summary["method_correct"].mean()),
        "historical_ko_share": hist["ko_share"],
        "simulated_ko_share": float(summary["p_fight_ko_tko"].mean()),
        "ko_share_bias": float(summary["p_fight_ko_tko"].mean() - hist["ko_share"]),
        "historical_mean_elapsed": hist["mean_elapsed"],
        "simulated_mean_elapsed": float(summary["sim_mean_elapsed"].mean()),
        "duration_relative_bias": float(summary["sim_mean_elapsed"].mean() / hist["mean_elapsed"] - 1.0),
        "mean_sequence_opportunities_per_path": float(summary["mean_sequence_opportunities"].mean()),
        "sequence_finish_share": float(summary["p_sequence_finish"].mean()),
    }
    for t in (300, 600, 900):
        sim = float(summary[f"p_nondec_by_{t}"].mean())
        rec[f"hist_nondec_by_{t}"] = hist[f"nondec_by_{t}"]
        rec[f"sim_nondec_by_{t}"] = sim
        rec[f"bias_nondec_by_{t}"] = sim - hist[f"nondec_by_{t}"]
    for r in (1, 2, 3):
        sim = float(summary[f"p_ko_r{r}"].mean())
        rec[f"hist_ko_r{r}"] = hist[f"ko_r{r}"]
        rec[f"sim_ko_r{r}"] = sim
        rec[f"bias_ko_r{r}"] = sim - hist[f"ko_r{r}"]
    return rec


def _division_metrics(summary: pd.DataFrame, paths: int, arm: str) -> pd.DataFrame:
    rows = []
    for division, g in summary.groupby("division", sort=False):
        y = g["actual_winner"].eq("red").astype(float).to_numpy()
        p = g["p_red_win"].to_numpy(float)
        winner_p = np.where(y > 0.5, p, 1.0 - p)
        rows.append({
            "arm": arm,
            "division": division,
            "n_fights": int(len(g)),
            "paths_per_fight": int(paths),
            "ml_accuracy": float(g["ml_correct"].mean()),
            "ml_brier": float(np.mean((p-y)**2)),
            "ml_logloss": float(-np.mean(np.log(np.clip(winner_p,1e-9,1.0)))),
            "method_accuracy": float(g["method_correct"].mean()),
            "simulated_ko_share": float(g["p_fight_ko_tko"].mean()),
            "simulated_mean_elapsed": float(g["sim_mean_elapsed"].mean()),
            "sequence_finish_share": float(g["p_sequence_finish"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--paths", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/full_mens_kd_sequence_validation"))
    args = ap.parse_args()

    _install_summary_wrapper()
    _, hist = _historical(args.target_n)
    all_metrics = []
    all_divisions = []
    all_summaries = []

    for arm in ARMS:
        s = _simulate_arm(args.target_n, args.paths, args.seed, arm)
        s["arm"] = arm
        all_summaries.append(s)
        all_metrics.append(_metrics(s, hist, args.paths, arm))
        all_divisions.append(_division_metrics(s, args.paths, arm))

    metrics = pd.DataFrame(all_metrics)
    divisions = pd.concat(all_divisions, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)

    control = metrics.loc[metrics.arm.eq("control")].iloc[0]
    candidate = metrics.loc[metrics.arm.eq("seq_b3")].iloc[0]
    delta_cols = [
        "ml_accuracy", "ml_brier", "ml_logloss", "mean_actual_winner_probability",
        "method_accuracy", "simulated_ko_share", "simulated_mean_elapsed",
        "duration_relative_bias", "sim_nondec_by_300", "sim_nondec_by_600",
        "sim_nondec_by_900", "sim_ko_r1", "sim_ko_r2", "sim_ko_r3",
    ]
    deltas = pd.DataFrame([{
        "candidate_minus_control_" + c: float(candidate[c] - control[c])
        for c in delta_cols
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "overall_ab_metrics.csv", index=False)
    divisions.to_csv(args.out_dir / "division_ab_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "candidate_minus_control.csv", index=False)
    summaries.to_csv(args.out_dir / "fight_summaries.csv", index=False)

    print("\nHISTORICAL TARGETS")
    for k, v in hist.items(): print(f"{k}: {v:.6f}")
    print("\nOVERALL A/B")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCANDIDATE MINUS CONTROL")
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nDIVISION A/B")
    print(divisions.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
