"""Stage-1b ablation for the from-scratch KO/KD research system.

Uses the Stage-1 raw leakage-safe dataset builder, then isolates which components
actually add out-of-sample signal. Still no FSR traits and no MC mechanics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1

DEFAULT_OUT = Path("data/research/ko_v3_from_scratch_stage1_ablation")


def kd_arms():
    return [
        s1.Arm("age_only", ("attacker_age", "defender_age")),
        s1.Arm("age_division", ("attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm50_rates_only", ("ewm50_att_kd_rate", "ewm50_def_kd_suscept")),
        s1.Arm("ewm50_exposure_only", ("ewm50_att_log_sig_landed", "ewm50_def_log_sig_absorbed")),
        s1.Arm("ewm50_rates_age_division", ("ewm50_att_kd_rate", "ewm50_def_kd_suscept", "attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm50_exposure_age_division", ("ewm50_att_log_sig_landed", "ewm50_def_log_sig_absorbed", "attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm50_full_age_division", ("ewm50_att_kd_rate", "ewm50_def_kd_suscept", "ewm50_att_log_sig_landed", "ewm50_def_log_sig_absorbed", "attacker_age", "defender_age"), ("division_cat",)),
    ]


def post_arms():
    return [
        s1.Arm("age_only", ("attacker_age", "defender_age")),
        s1.Arm("age_division", ("attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm75_conversion_only", ("ewm75_att_post_kd_conversion",)),
        s1.Arm("ewm75_recovery_only", ("ewm75_def_post_kd_recovery",)),
        s1.Arm("ewm75_conversion_recovery_only", ("ewm75_att_post_kd_conversion", "ewm75_def_post_kd_recovery")),
        s1.Arm("ewm75_exposure_age_division", ("ewm75_att_log_fights", "ewm75_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm75_conversion_recovery_age_division_no_exposure", ("ewm75_att_post_kd_conversion", "ewm75_def_post_kd_recovery", "attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm75_full_age_division", ("ewm75_att_post_kd_conversion", "ewm75_def_post_kd_recovery", "ewm75_att_log_fights", "ewm75_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)),
    ]


def direct_arms():
    return [
        s1.Arm("age_only", ("attacker_age", "defender_age")),
        s1.Arm("age_division", ("attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm90_att_direct_only", ("ewm90_att_direct_ko_rate",)),
        s1.Arm("ewm90_def_direct_only", ("ewm90_def_direct_ko_loss_rate",)),
        s1.Arm("ewm90_direct_both_only", ("ewm90_att_direct_ko_rate", "ewm90_def_direct_ko_loss_rate")),
        s1.Arm("ewm90_exposure_age_division", ("ewm90_att_log_fights", "ewm90_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm90_direct_age_division_no_exposure", ("ewm90_att_direct_ko_rate", "ewm90_def_direct_ko_loss_rate", "attacker_age", "defender_age"), ("division_cat",)),
        s1.Arm("ewm90_full_age_division", ("ewm90_att_direct_ko_rate", "ewm90_def_direct_ko_loss_rate", "ewm90_att_log_fights", "ewm90_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)),
    ]


def delta(summary: pd.DataFrame, a: str, b: str, metric: str) -> dict | None:
    if summary.empty or not {a, b}.issubset(set(summary.arm)):
        return None
    av = float(summary.loc[summary.arm.eq(a), metric].iloc[0])
    bv = float(summary.loc[summary.arm.eq(b), metric].iloc[0])
    return {"candidate": a, "reference": b, "metric": metric, "candidate_value": av, "reference_value": bv, "candidate_minus_reference": av - bv}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-path", type=Path, default=ROUND_STATS_PATH)
    p.add_argument("--master-path", type=Path, default=MASTER_PATH)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--first-test-year", type=int, default=2020)
    return p.parse_args()


def main():
    args = parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    ff, audit = s1.load_raw_fighter_fights(args.round_path, args.master_path)
    frame = s1.build_matchup_frame(s1.build_prefight_states(ff))

    original = s1._kd_arms
    try:
        s1._kd_arms = kd_arms
        kd_detail, kd_year, kd_coef = s1.run_kd_walkforward(frame, args.first_test_year)
    finally:
        s1._kd_arms = original
    kd_summary = s1._aggregate_kd(kd_detail)

    post_detail, post_year, post_coef = s1.run_binary_walkforward(
        frame, first_test_year=args.first_test_year,
        eligible_mask=frame["post_kd_opportunity"].gt(0), target="post_kd_finish",
        arms=post_arms(), label="post_kd_finish")
    post_summary = s1._aggregate_binary(post_detail, "post_kd_finish", "p_post_kd_finish")

    direct_detail, direct_year, direct_coef = s1.run_binary_walkforward(
        frame, first_test_year=args.first_test_year,
        eligible_mask=pd.Series(True, index=frame.index), target="direct_ko_win",
        arms=direct_arms(), label="direct_ko")
    direct_summary = s1._aggregate_binary(direct_detail, "direct_ko_win", "p_direct_ko")

    for name, df in {
        "kd_ablation_summary.csv": kd_summary, "kd_ablation_by_year.csv": kd_year, "kd_ablation_coefficients.csv": kd_coef,
        "post_kd_ablation_summary.csv": post_summary, "post_kd_ablation_by_year.csv": post_year, "post_kd_ablation_coefficients.csv": post_coef,
        "direct_ko_ablation_summary.csv": direct_summary, "direct_ko_ablation_by_year.csv": direct_year, "direct_ko_ablation_coefficients.csv": direct_coef,
    }.items():
        df.to_csv(args.out_dir / name, index=False)

    report = {
        "stage": "KO V3 from scratch — Stage 1b component ablation",
        "uses_fsr_traits": False, "changes_mc_mechanics": False, "same_date_delayed": True,
        "raw_data_audit": audit,
        "kd_full_vs_age_division": delta(kd_summary, "ewm50_full_age_division", "age_division", "strike_log_loss"),
        "kd_rates_vs_age_division": delta(kd_summary, "ewm50_rates_age_division", "age_division", "strike_log_loss"),
        "kd_exposure_vs_age_division": delta(kd_summary, "ewm50_exposure_age_division", "age_division", "strike_log_loss"),
        "post_full_vs_age_division": delta(post_summary, "ewm75_full_age_division", "age_division", "log_loss"),
        "post_conversion_no_exposure_vs_age_division": delta(post_summary, "ewm75_conversion_recovery_age_division_no_exposure", "age_division", "log_loss"),
        "post_exposure_vs_age_division": delta(post_summary, "ewm75_exposure_age_division", "age_division", "log_loss"),
        "direct_full_vs_age_division": delta(direct_summary, "ewm90_full_age_division", "age_division", "log_loss"),
        "direct_rates_no_exposure_vs_age_division": delta(direct_summary, "ewm90_direct_age_division_no_exposure", "age_division", "log_loss"),
        "direct_exposure_vs_age_division": delta(direct_summary, "ewm90_exposure_age_division", "age_division", "log_loss"),
    }
    (args.out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KO V3 FROM SCRATCH — STAGE 1B ABLATION")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nKD"); print(kd_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPOST-KD"); print(post_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nDIRECT KO PROXY"); print(direct_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
