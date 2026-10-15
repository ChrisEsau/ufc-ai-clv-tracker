"""Head-to-head R1 KO replay: strong KD-collapse baseline vs age-adjusted traits.

Diagnostic only. Uses the mature 2020+ cohort's actual R1 KO/TKO bouts and
compares two shadow variants with shared path seeds:

1. strong KD-collapse, stored pre-fight FSR traits unchanged
2. strong KD-collapse, linear age adjustment after 30 at -2 points/year applied
   independently to knockdown_resistance and damage_durability

No stored FSR values or simulator constants are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_age_decay_curve_search_kd_durability_2020plus_mature as age_curves
from scripts.experimental import fsr_r1_ko_age_curve_mc_replay_2020plus_mature as replay
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "r1_ko_age_curve_strong_collapse_head_to_head.csv"
)
DEFAULT_PATHS = 50
DEFAULT_SEED = 20260810

STRONG = collapse_mod.CollapseCandidate("strong", 5.0, 2.0)
AGE_CURVE = age_curves.Curve("linear", onset=30.0, slope=2.0)
VARIANTS = ("baseline", "linear_on30_s2")

AGE_BINS = [-np.inf, 30.999, 33.999, 36.999, 39.999, np.inf]
AGE_LABELS = ["<=30", "31-33", "34-36", "37-39", "40+"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _effective_profile(profile: pd.Series, age: float, use_age_curve: bool) -> pd.Series:
    if not use_age_curve:
        return profile.copy()
    out = profile.copy()
    ages = pd.Series([age], dtype=float)
    for trait in ("knockdown_resistance", "damage_durability"):
        raw = pd.Series([pd.to_numeric(out.get(trait), errors="coerce")], dtype=float)
        out[trait] = float(age_curves._apply_curve(raw, ages, AGE_CURVE).iloc[0])
    return out


def _simulate_variant(
    bout: pd.Series,
    pair: tuple[pd.Series, pd.Series],
    *,
    variant: str,
    path_seeds: np.ndarray,
) -> dict[str, object]:
    red, blue = pair
    use_age = variant == "linear_on30_s2"
    red_eff = _effective_profile(red, float(bout["r_age"]), use_age)
    blue_eff = _effective_profile(blue, float(bout["b_age"]), use_age)

    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    winner_id = str(bout["winner_id"])
    r_ko = 0
    b_ko = 0

    for seed in path_seeds:
        sim = collapse_mod.StaticFSRMCKOTKOV2KDCollapse(
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
            r_ko += 1
        else:
            b_ko += 1

    n = len(path_seeds)
    p_r = r_ko / n
    p_b = b_ko / n
    if winner_id == r_id:
        p_winner, p_loser = p_r, p_b
        winner_age, loser_age = float(bout["r_age"]), float(bout["b_age"])
    elif winner_id == b_id:
        p_winner, p_loser = p_b, p_r
        winner_age, loser_age = float(bout["b_age"]), float(bout["r_age"])
    else:
        p_winner = p_loser = np.nan
        winner_age = loser_age = np.nan

    return {
        "variant": variant,
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "winner_id": winner_id,
        "winner_age": winner_age,
        "loser_age": loser_age,
        "p_r_r1_ko": p_r,
        "p_b_r1_ko": p_b,
        "p_any_r1_ko": p_r + p_b,
        "p_actual_winner_r1_ko": p_winner,
        "p_actual_loser_r1_ko": p_loser,
        "winner_direction_hit": int(p_winner > p_loser),
        "direction_tie": int(p_winner == p_loser),
    }


def _print_summary(results: pd.DataFrame) -> None:
    print("\n" + "=" * 122)
    print("STRONG KD-COLLAPSE — BASELINE VS AGE CURVE")
    print("=" * 122)
    overall = results.groupby("variant", observed=True).agg(
        bouts=("bout_id", "size"),
        mean_p_any_r1_ko=("p_any_r1_ko", "mean"),
        mean_p_actual_winner_r1_ko=("p_actual_winner_r1_ko", "mean"),
        mean_p_actual_loser_r1_ko=("p_actual_loser_r1_ko", "mean"),
        winner_direction_hit_rate=("winner_direction_hit", "mean"),
        direction_tie_rate=("direction_tie", "mean"),
    ).reset_index()
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    temp = results.copy()
    temp["loser_age_band"] = pd.cut(temp["loser_age"], AGE_BINS, labels=AGE_LABELS)
    print("\nBY ACTUAL LOSER AGE")
    by_age = temp.groupby(["variant", "loser_age_band"], observed=True).agg(
        bouts=("bout_id", "size"),
        mean_loser_age=("loser_age", "mean"),
        mean_p_any_r1_ko=("p_any_r1_ko", "mean"),
        mean_p_actual_winner_r1_ko=("p_actual_winner_r1_ko", "mean"),
        mean_p_actual_loser_r1_ko=("p_actual_loser_r1_ko", "mean"),
        winner_direction_hit_rate=("winner_direction_hit", "mean"),
    ).reset_index()
    print(by_age.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pivot = results.pivot(index="bout_id", columns="variant")
    base_hit = pivot["winner_direction_hit"]["baseline"]
    age_hit = pivot["winner_direction_hit"]["linear_on30_s2"]
    wrong_to_right = int(((base_hit == 0) & (age_hit == 1)).sum())
    right_to_wrong = int(((base_hit == 1) & (age_hit == 0)).sum())
    unchanged_right = int(((base_hit == 1) & (age_hit == 1)).sum())
    unchanged_wrong = int(((base_hit == 0) & (age_hit == 0)).sum())

    base_pw = pivot["p_actual_winner_r1_ko"]["baseline"]
    age_pw = pivot["p_actual_winner_r1_ko"]["linear_on30_s2"]
    base_pa = pivot["p_any_r1_ko"]["baseline"]
    age_pa = pivot["p_any_r1_ko"]["linear_on30_s2"]

    print("\nPAIRWISE EFFECT OF AGE CURVE")
    print(f"wrong -> right direction: {wrong_to_right}")
    print(f"right -> wrong direction: {right_to_wrong}")
    print(f"unchanged right:           {unchanged_right}")
    print(f"unchanged wrong/tie:       {unchanged_wrong}")
    print(f"mean delta P(actual winner R1 KO): {(age_pw - base_pw).mean():+.4f}")
    print(f"mean delta P(any R1 KO):           {(age_pa - base_pa).mean():+.4f}")

    loser_age = results.loc[results["variant"].eq("baseline")].set_index("bout_id")["loser_age"]
    old_ids = loser_age.index[loser_age >= 37.0]
    if len(old_ids):
        print("\nACTUAL LOSER AGE >=37 — PAIRWISE")
        print(f"bouts: {len(old_ids)}")
        print(f"baseline mean P(actual winner): {base_pw.loc[old_ids].mean():.4f}")
        print(f"age-curve mean P(actual winner): {age_pw.loc[old_ids].mean():.4f}")
        print(f"delta: {(age_pw.loc[old_ids] - base_pw.loc[old_ids]).mean():+.4f}")
        print(f"baseline direction hit: {base_hit.loc[old_ids].mean():.4f}")
        print(f"age-curve direction hit: {age_hit.loc[old_ids].mean():.4f}")


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = replay._build_cohort()
    sample = cohort.loc[cohort["actual_r1_ko"].eq(1)].copy().reset_index(drop=True)
    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"actual R1 KO/TKO bouts: {len(sample):,}")
    print("variants: baseline strong-collapse vs linear_on30_s2 strong-collapse")
    print(f"paths per bout/variant: {args.paths:,}")
    print("Shared path seeds are reused across both variants for every bout.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    total_paths = len(sample) * len(VARIANTS) * args.paths
    completed_paths = 0
    jobs = 0
    total_jobs = len(sample) * len(VARIANTS)

    for _, bout in sample.iterrows():
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        pair = pairs[str(bout["bout_id"])]
        for variant in VARIANTS:
            rows.append(_simulate_variant(bout, pair, variant=variant, path_seeds=seeds))
            jobs += 1
            completed_paths += args.paths
            if completed_paths % 1000 == 0 or jobs == total_jobs:
                print(
                    f"[strong-collapse age H2H] paths {completed_paths:,}/{total_paths:,}; "
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
