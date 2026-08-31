"""Research-only men's flyweight joint lethality tuning screen.

Tunes only two consequence-side parameters around the validated men's architecture:
  1) fresh-power intercept in offset(t) = clip(intercept - t/12, -40, intercept)
  2) post-KD finishing-sequence logit bonus

The fatigue slope, strike/takedown/submission budgets, timing, FSR, judging, and
frozen V1 source are unchanged. This screen reports method shares/timing plus
ML and method probability quality so selection is not based on KO share alone.
"""
from __future__ import annotations

import argparse
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1 import run_event_or_fight as runner
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit

DIVISION = "flyweight"
DENOMINATOR = 12.0
LOWER_CAP = -40.0
SEED = 20260823

# Compact coarse grid plus the current all-men reference.
ARMS = []
for intercept in (15.0, 20.0, 25.0):
    for bonus in (None, 1.0, 2.0, 3.0):
        ARMS.append((f"i{int(intercept)}_b{'0' if bonus is None else int(bonus)}", intercept, bonus))
ARMS.append(("mens_ref_i35_b3", 35.0, 3.0))


def _method_probs(summary: pd.DataFrame) -> np.ndarray:
    return summary[["p_fight_dec", "p_fight_ko_tko", "p_fight_sub"]].to_numpy(float)


def _method_targets(summary: pd.DataFrame) -> np.ndarray:
    classes = {"DEC": 0, "KO_TKO": 1, "SUB": 2}
    y = np.zeros((len(summary), 3), dtype=float)
    for i, m in enumerate(summary["actual_method"].astype(str)):
        y[i, classes[m]] = 1.0
    return y


def _metrics(summary: pd.DataFrame, hist: dict, arm: str, intercept: float, bonus: float | None, paths: int) -> dict:
    y_red = summary["actual_winner"].eq("red").astype(float).to_numpy()
    p_red = summary["p_red_win"].to_numpy(float)
    winner_p = np.where(y_red > 0.5, p_red, 1.0 - p_red)

    pm = np.clip(_method_probs(summary), 1e-9, 1.0)
    ym = _method_targets(summary)
    actual_method_p = np.sum(pm * ym, axis=1)

    rec = {
        "arm": arm,
        "intercept": intercept,
        "sequence_bonus": 0.0 if bonus is None else float(bonus),
        "sequence_enabled": bonus is not None,
        "n_fights": len(summary),
        "paths_per_fight": paths,
        "ml_accuracy": float(summary["ml_correct"].mean()),
        "ml_brier": float(np.mean((p_red - y_red) ** 2)),
        "ml_logloss": float(-np.mean(np.log(np.clip(winner_p, 1e-9, 1.0)))),
        "mean_actual_winner_probability": float(winner_p.mean()),
        "method_accuracy": float(summary["method_correct"].mean()),
        "method_brier_multiclass": float(np.mean(np.sum((pm - ym) ** 2, axis=1))),
        "method_logloss": float(-np.mean(np.log(np.clip(actual_method_p, 1e-9, 1.0)))),
        "mean_actual_method_probability": float(actual_method_p.mean()),
        "sim_dec": float(summary["p_fight_dec"].mean()),
        "sim_ko": float(summary["p_fight_ko_tko"].mean()),
        "sim_sub": float(summary["p_fight_sub"].mean()),
        "sim_mean_elapsed": float(summary["sim_mean_elapsed"].mean()),
        "duration_bias": float(summary["sim_mean_elapsed"].mean() / hist["mean_elapsed"] - 1.0),
        "sim_nondec_by_300": float(summary["p_nondec_by_300"].mean()),
        "sim_nondec_by_600": float(summary["p_nondec_by_600"].mean()),
        "sim_nondec_by_900": float(summary["p_nondec_by_900"].mean()),
        "sim_ko_r1": float(summary["p_ko_r1"].mean()),
        "sim_ko_r2": float(summary["p_ko_r2"].mean()),
        "sim_ko_r3": float(summary["p_ko_r3"].mean()),
        "sequence_finish_share": float(summary["p_sequence_finish"].mean()),
    }
    for key in ("dec", "ko", "sub", "nondec_by_300", "nondec_by_600", "nondec_by_900", "ko_r1", "ko_r2", "ko_r3"):
        rec[f"hist_{key}"] = hist[key]
        rec[f"bias_{key}"] = rec[f"sim_{key}"] - hist[key]

    # Screening score: method calibration and finish timing dominate; ML quality is a guardrail.
    rec["screen_score"] = float(
        3.0 * rec["bias_ko"] ** 2
        + 1.5 * rec["bias_dec"] ** 2
        + 1.5 * rec["bias_sub"] ** 2
        + 2.0 * rec["bias_nondec_by_300"] ** 2
        + 1.0 * rec["bias_nondec_by_600"] ** 2
        + 1.5 * rec["bias_ko_r1"] ** 2
        + 1.0 * rec["bias_ko_r2"] ** 2
        + 0.5 * rec["bias_ko_r3"] ** 2
        + 0.25 * rec["ml_brier"]
        + 0.25 * rec["method_brier_multiclass"]
    )
    return rec


def _historical(cohort: pd.DataFrame) -> dict:
    method = cohort["method"].map(wc_audit.normalize_method)
    elapsed = pd.to_numeric(cohort["match_time_sec"], errors="raise")
    out = {
        "dec": float(method.eq("DEC").mean()),
        "ko": float(method.eq("KO_TKO").mean()),
        "sub": float(method.eq("SUB").mean()),
        "mean_elapsed": float(elapsed.mean()),
    }
    for t in (300, 600, 900):
        out[f"nondec_by_{t}"] = float(((method != "DEC") & elapsed.le(t)).mean())
    fr = pd.to_numeric(cohort["finish_round"], errors="coerce")
    for r in (1, 2, 3):
        out[f"ko_r{r}"] = float((method.eq("KO_TKO") & fr.eq(r)).mean())
    return out


def _install_round_wrapper() -> None:
    # Sequence module wrapper adds finish timing and sequence diagnostics.
    seq._install_summary_wrapper()
    original = canonical.summarize_fight

    def wrapped(fight_id, pair, rows, master_row):
        out = original(fight_id, pair, rows, master_row)
        p = pd.DataFrame(rows)
        for r in (1, 2, 3):
            out[f"p_ko_r{r}"] = float((p["method"].eq("KO_TKO") & p["finish_round"].eq(r)).mean())
        return out

    canonical.summarize_fight = wrapped


def _run_arm(cohort: pd.DataFrame, paths: int, seed: int, arm: str, intercept: float, bonus: float | None) -> pd.DataFrame:
    seq.INTERCEPT = float(intercept)
    seq.DENOMINATOR = DENOMINATOR
    seq.LOWER_CAP = LOWER_CAP
    seq.UPPER_CAP = float(intercept)
    seq.ARMS = {arm: bonus}
    seq._MODE = arm
    runner.simulate_detailed_path = seq.sequence_simulate_detailed_path
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path
    print(f"ARM {arm} | intercept={intercept:.1f} bonus={bonus} | fights={len(cohort)} paths={paths}")
    return canonical._simulate_c(cohort, paths, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/flyweight_joint_tuning_screen"))
    args = ap.parse_args()

    cohort, _ = wc_audit.select_cohort(DIVISION, args.target_n)
    cohort = cohort.reset_index(drop=True)
    hist = _historical(cohort)
    _install_round_wrapper()

    metrics = []
    summaries = []
    for arm, intercept, bonus in ARMS:
        summary = _run_arm(cohort, args.paths, args.seed, arm, intercept, bonus)
        summary["arm"] = arm
        summary["intercept"] = intercept
        summary["sequence_bonus"] = 0.0 if bonus is None else bonus
        summaries.append(summary)
        metrics.append(_metrics(summary, hist, arm, intercept, bonus, args.paths))

    m = pd.DataFrame(metrics).sort_values("screen_score").reset_index(drop=True)
    s = pd.concat(summaries, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_dir / "arm_metrics.csv", index=False)
    s.to_csv(args.out_dir / "fight_summaries.csv", index=False)

    print("\nHISTORICAL FLYWEIGHT TARGETS")
    for k, v in hist.items():
        print(f"{k}: {v:.6f}")
    print("\nRANKED ARMS")
    cols = [
        "arm","screen_score","ml_accuracy","ml_brier","ml_logloss","method_accuracy",
        "method_brier_multiclass","method_logloss","sim_dec","sim_ko","sim_sub",
        "duration_bias","sim_nondec_by_300","sim_nondec_by_600","sim_ko_r1","sim_ko_r2","sim_ko_r3",
        "sequence_finish_share",
    ]
    print(m[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
