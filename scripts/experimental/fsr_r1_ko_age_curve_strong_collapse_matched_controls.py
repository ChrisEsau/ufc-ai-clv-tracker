"""Matched-control replay for strong KD-collapse age-curve calibration.

Purpose
-------
Compare the strong KD-collapse shadow simulator with and without the leading
age adjustment (linear_on30_s2) on a balanced sample of actual R1 KO/TKO bouts
and deterministic age-matched non-R1-KO controls.

This asks whether the age curve increases simulated R1-KO probability more on
true R1-KO fights than on age-matched controls.

No stored FSR values or simulator constants are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.experimental import fsr_age_decay_curve_search_kd_durability_2020plus_mature as curves
from scripts.experimental import fsr_r1_ko_age_curve_mc_replay_2020plus_mature as replay
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "r1_ko_age_curve_strong_collapse_matched_controls.csv"
)
DEFAULT_PATHS = 50
DEFAULT_SEED = 20260810

STRONG = collapse.CollapseCandidate("strong", 5.0, 2.0)
VARIANTS = {
    "baseline": curves.Curve("none"),
    "linear_on30_s2": curves.Curve("linear", onset=30.0, slope=2.0),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _effective_profile(profile: pd.Series, age: float, curve: curves.Curve) -> pd.Series:
    out = profile.copy()
    age_series = pd.Series([age], dtype=float)
    for trait in ("knockdown_resistance", "damage_durability"):
        original = pd.Series([pd.to_numeric(out.get(trait), errors="coerce")], dtype=float)
        out[trait] = float(curves._apply_curve(original, age_series, curve).iloc[0])
    return out


def _simulate_bout(
    bout: pd.Series,
    pair: tuple[pd.Series, pd.Series],
    curve: curves.Curve,
    path_seeds: np.ndarray,
) -> dict[str, object]:
    red, blue = pair
    red_eff = _effective_profile(red, float(bout["r_age"]), curve)
    blue_eff = _effective_profile(blue, float(bout["b_age"]), curve)

    r_r1_ko = 0
    b_r1_ko = 0
    for seed in path_seeds:
        sim = collapse.StaticFSRMCKOTKOV2KDCollapse(
            red_eff,
            blue_eff,
            collapse=STRONG,
            rounds=3,
            seed=int(seed),
        )
        result = sim.run()
        finish = result.finish
        if finish is None or finish.round != 1:
            continue
        if finish.winner == 0:
            r_r1_ko += 1
        else:
            b_r1_ko += 1

    n = len(path_seeds)
    p_r = r_r1_ko / n
    p_b = b_r1_ko / n
    return {
        "bout_id": str(bout["bout_id"]),
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "sample_class": bout["sample_class"],
        "mean_age": float(bout["mean_age"]),
        "max_age": float(bout["max_age"]),
        "p_r_r1_ko": p_r,
        "p_b_r1_ko": p_b,
        "p_any_r1_ko": p_r + p_b,
    }


def _print_summary(results: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("STRONG KD-COLLAPSE — AGE CURVE MATCHED-CONTROL REPLAY")
    print("=" * 120)

    rows = []
    for variant, g in results.groupby("variant", observed=True):
        pos = g.loc[g["actual_r1_ko"].eq(1)]
        ctl = g.loc[g["actual_r1_ko"].eq(0)]
        y = g["actual_r1_ko"].astype(int)
        p = g["p_any_r1_ko"].clip(1e-6, 1 - 1e-6)
        rows.append({
            "variant": variant,
            "n": len(g),
            "mean_p_positive": pos["p_any_r1_ko"].mean(),
            "mean_p_control": ctl["p_any_r1_ko"].mean(),
            "separation": pos["p_any_r1_ko"].mean() - ctl["p_any_r1_ko"].mean(),
            "auc": roc_auc_score(y, p),
            "brier_balanced": brier_score_loss(y, p),
            "logloss_balanced": log_loss(y, p),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    pivot = results.pivot(index="bout_id", columns="variant", values="p_any_r1_ko")
    meta = results.drop_duplicates("bout_id").set_index("bout_id")
    pivot["actual_r1_ko"] = meta["actual_r1_ko"]
    pivot["delta"] = pivot["linear_on30_s2"] - pivot["baseline"]

    print("\nPAIRWISE AGE-CURVE DELTA")
    for target, label in ((1, "actual R1 KO"), (0, "age-matched control")):
        g = pivot.loc[pivot["actual_r1_ko"].eq(target)]
        print(
            f"{label}: n={len(g)}  mean delta P(any R1 KO)={g['delta'].mean():+.4f}  "
            f"median={g['delta'].median():+.4f}  increased={(g['delta'] > 0).mean():.2%}",
        )

    pos_delta = pivot.loc[pivot["actual_r1_ko"].eq(1), "delta"].mean()
    ctl_delta = pivot.loc[pivot["actual_r1_ko"].eq(0), "delta"].mean()
    print(f"delta separation gain: {(pos_delta - ctl_delta):+.4f}")


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = replay._build_cohort()
    sample = replay._match_controls(cohort)
    positives = int(sample["actual_r1_ko"].sum())
    controls = len(sample) - positives

    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"actual R1 KO/TKO bouts: {positives:,}")
    print(f"age-matched controls: {controls:,}")
    print(f"variants: {len(VARIANTS)}")
    print(f"paths per bout/variant: {args.paths:,}")
    print("Strong KD-collapse: scale=5.0, curvature=2.0")
    print("Common random-number seeds are reused across variants for each bout.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    total_jobs = len(sample) * len(VARIANTS)
    total_paths = total_jobs * args.paths
    completed_paths = 0
    jobs = 0

    for _, bout in sample.iterrows():
        bout_id = str(bout["bout_id"])
        path_seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        pair = pairs[bout_id]
        for label, curve in VARIANTS.items():
            row = _simulate_bout(bout, pair, curve, path_seeds)
            row["variant"] = label
            rows.append(row)
            jobs += 1
            completed_paths += args.paths
            if completed_paths % 1000 == 0 or jobs == total_jobs:
                print(
                    f"[matched control replay] paths {completed_paths:,}/{total_paths:,}; "
                    f"jobs {jobs:,}/{total_jobs:,}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    _print_summary(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nWrote {len(results):,} bout-variant rows to {args.output}")
    print("No stored FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
