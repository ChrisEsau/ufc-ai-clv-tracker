"""Research-only men's flyweight power-decay shape screen.

Purpose: reduce excess R1 KO concentration while preserving more consequence later
in the fight. The post-KD finishing sequence is disabled for every arm.

Only the consequence-side power offset curve changes:
    offset(t) = clip(intercept - t / denominator, -40, intercept)

Strike/takedown/submission budgets, event timing, FSR, judging, and frozen V1
source remain unchanged. The prior flyweight best coarse arm (15 / 12, sequence
off) is included as the reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.simulation.event_clock_mc_v1 import run_event_or_fight as runner
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import flyweight_joint_tuning_screen as fw
from pipeline.simulation.event_clock_mc_v2.diagnostics import kd_finishing_sequence_screen as seq
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit

DIVISION = "flyweight"
LOWER_CAP = -40.0
SEED = 20260823

# Lower fresh boost + slower fade, with the prior i15/d12 winner as reference.
ARMS = [
    ("ref_i15_d12", 15.0, 12.0),
    ("i15_d18", 15.0, 18.0),
    ("i15_d24", 15.0, 24.0),
    ("i10_d18", 10.0, 18.0),
    ("i10_d24", 10.0, 24.0),
    ("i10_d30", 10.0, 30.0),
    ("i5_d18", 5.0, 18.0),
    ("i5_d24", 5.0, 24.0),
    ("i5_d30", 5.0, 30.0),
]


def _run_arm(cohort: pd.DataFrame, paths: int, seed: int, arm: str, intercept: float, denominator: float) -> pd.DataFrame:
    seq.INTERCEPT = float(intercept)
    seq.DENOMINATOR = float(denominator)
    seq.LOWER_CAP = LOWER_CAP
    seq.UPPER_CAP = float(intercept)
    seq.ARMS = {arm: None}
    seq._MODE = arm
    runner.simulate_detailed_path = seq.sequence_simulate_detailed_path
    canonical.simulate_detailed_path = seq.sequence_simulate_detailed_path
    print(f"ARM {arm} | intercept={intercept:.1f} denominator={denominator:.1f} | fights={len(cohort)} paths={paths}")
    return canonical._simulate_c(cohort, paths, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/flyweight_decay_shape_screen"))
    args = ap.parse_args()

    cohort, _ = wc_audit.select_cohort(DIVISION, args.target_n)
    cohort = cohort.reset_index(drop=True)
    hist = fw._historical(cohort)
    fw._install_round_wrapper()

    metrics = []
    summaries = []
    for arm, intercept, denominator in ARMS:
        summary = _run_arm(cohort, args.paths, args.seed, arm, intercept, denominator)
        summary["arm"] = arm
        summary["intercept"] = intercept
        summary["denominator"] = denominator
        summaries.append(summary)
        rec = fw._metrics(summary, hist, arm, intercept, None, args.paths)
        rec["denominator"] = denominator
        metrics.append(rec)

    m = pd.DataFrame(metrics).sort_values("screen_score").reset_index(drop=True)
    s = pd.concat(summaries, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_dir / "arm_metrics.csv", index=False)
    s.to_csv(args.out_dir / "fight_summaries.csv", index=False)

    print("\nHISTORICAL FLYWEIGHT TARGETS")
    for k, v in hist.items():
        print(f"{k}: {v:.6f}")

    print("\nRANKED DECAY ARMS")
    cols = [
        "arm", "intercept", "denominator", "screen_score",
        "ml_accuracy", "ml_brier", "ml_logloss", "mean_actual_winner_probability",
        "method_accuracy", "method_brier_multiclass", "method_logloss", "mean_actual_method_probability",
        "sim_dec", "sim_ko", "sim_sub", "duration_bias",
        "sim_nondec_by_300", "sim_nondec_by_600", "sim_nondec_by_900",
        "sim_ko_r1", "sim_ko_r2", "sim_ko_r3",
    ]
    print(m[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
