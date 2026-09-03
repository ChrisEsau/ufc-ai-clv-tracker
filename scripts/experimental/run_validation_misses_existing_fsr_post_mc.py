"""PRE vs direct observed-POST FSR MC audit for frozen validation misses.

This script DOES NOT rebuild/replay the FSR rating database.

For every missed fight in the frozen 34-fight validation baseline it:
1. reads the existing aligned FSR-32 prefight profiles;
2. derives immediate post-fight canonical FSR by applying only that fight's
   RFS/round/master evidence with calculate_postfight_fsr_from_existing.py;
3. runs PRE and POST Monte Carlo arms with identical seeds;
4. measures movement in probability of the actual winner.

The direct updater caches leakage-safe population observation contexts by event
 date and validates separately against the already-generated Ricci/Kline
sentinel result before this batch should be trusted.

Diagnostic only: POST arms are intentionally leaky/counterfactual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import calculate_postfight_fsr_from_existing as postcalc
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import run_single_historical_full_fight_bout as historical


BASELINE_PATH = Path(
    "data/experimental/validation_baselines/fsr_mc_card_validation_prechange_v1.csv"
)
OUTPUT_PATH = Path(
    "data/experimental/validation_misses_direct_post_fsr_mc.csv"
)
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811


def _overlay_post(profile: pd.Series, direct_row: pd.Series) -> pd.Series:
    out = profile.copy()
    for trait in canonical.CANONICAL_RATINGS:
        out[trait] = float(direct_row[f"{trait}_post"])
    out["distance_precision"] = float(out["distance_striking_precision"])
    out["distance_defense"] = float(out["distance_striking_defense"])
    out["stamina_depletion_resistance"] = float(out["fatigue_accumulation_resistance"])
    out["stamina_performance_resilience"] = float(out["fatigue_performance_resilience"])
    out["stamina_capacity"] = float(canonical.STAMINA_CAPACITY)
    return out


def _run_arm(
    red: pd.Series,
    blue: pd.Series,
    *,
    seeds: np.ndarray,
    red_age: float | None,
    blue_age: float | None,
) -> dict[str, float]:
    winners = [0, 0]
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    for seed in seeds:
        sim = full.StaticFSRMCFullFightV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()
        winners[int(path.winner)] += 1
        methods[str(path.method)] += 1
    n = float(len(seeds))
    return {
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n,
        "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if not BASELINE_PATH.exists():
        raise RuntimeError(f"required input not found: {BASELINE_PATH}")

    baseline = pd.read_csv(BASELINE_PATH)
    misses = baseline.loc[pd.to_numeric(baseline["mc_correct"], errors="coerce").eq(0)].copy()
    print(f"[direct-post batch] frozen misses: {len(misses):,}", flush=True)

    print("[direct-post batch] loading aligned cohort...", flush=True)
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    bout_meta = cohort.set_index("bout_id", drop=False)

    print("[direct-post batch] initializing cached one-fight FSR calculator...", flush=True)
    calculator = postcalc.PostfightFSRCalculator()

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)
    results: list[dict[str, object]] = []

    total = len(misses)
    for idx, miss in enumerate(misses.itertuples(index=False), start=1):
        bout_id = str(miss.bout_id)
        print(
            f"\n[direct-post batch] bout {idx}/{total} | {bout_id} | "
            f"{miss.red} vs {miss.blue}",
            flush=True,
        )
        if bout_id not in pairs or bout_id not in bout_meta.index:
            raise RuntimeError(f"missed bout {bout_id} is not in aligned mature cohort")

        bout = bout_meta.loc[bout_id]
        pre_red, pre_blue = pairs[bout_id]
        red_id = str(pre_red["fighter_id"])
        blue_id = str(pre_blue["fighter_id"])

        direct = calculator.calculate(bout_id)
        direct["fighter_id"] = direct["fighter_id"].astype(str)
        rrow = direct.loc[direct["fighter_id"].eq(red_id)]
        brow = direct.loc[direct["fighter_id"].eq(blue_id)]
        if len(rrow) != 1 or len(brow) != 1:
            raise RuntimeError(f"direct post result does not align to bout {bout_id}")

        post_red = _overlay_post(pre_red, rrow.iloc[0])
        post_blue = _overlay_post(pre_blue, brow.iloc[0])
        red_age = historical._age(bout, "r_age")
        blue_age = historical._age(bout, "b_age")

        print(f"  running PRE  {args.paths:,} paths...", flush=True)
        pre = _run_arm(pre_red, pre_blue, seeds=seeds, red_age=red_age, blue_age=blue_age)
        print(f"  running POST {args.paths:,} paths...", flush=True)
        post = _run_arm(post_red, post_blue, seeds=seeds, red_age=red_age, blue_age=blue_age)

        red_name = str(miss.red)
        blue_name = str(miss.blue)
        actual = str(miss.actual_winner)
        if actual == red_name:
            pre_actual = pre["p_red_win"]
            post_actual = post["p_red_win"]
            post_favorite = red_name if post["p_red_win"] >= 0.5 else blue_name
        elif actual == blue_name:
            pre_actual = pre["p_blue_win"]
            post_actual = post["p_blue_win"]
            post_favorite = blue_name if post["p_blue_win"] >= 0.5 else red_name
        else:
            raise RuntimeError(f"actual winner {actual!r} does not match bout sides")

        delta = post_actual - pre_actual
        flipped = post_favorite == actual
        print(
            f"  ACTUAL {actual}: {pre_actual:.1%} -> {post_actual:.1%} "
            f"({100*delta:+.1f} pp) | flipped={flipped}",
            flush=True,
        )

        results.append({
            "bout_id": bout_id,
            "event_date": miss.event_date,
            "red": red_name,
            "blue": blue_name,
            "actual_winner": actual,
            "actual_method": miss.actual_method,
            "red_age": red_age,
            "blue_age": blue_age,
            "red_prior_ufc_fights": pre_red.get("prior_ufc_fights", np.nan),
            "blue_prior_ufc_fights": pre_blue.get("prior_ufc_fights", np.nan),
            "pre_p_red_win": pre["p_red_win"],
            "pre_p_blue_win": pre["p_blue_win"],
            "post_p_red_win": post["p_red_win"],
            "post_p_blue_win": post["p_blue_win"],
            "pre_p_actual_winner": pre_actual,
            "post_p_actual_winner": post_actual,
            "actual_winner_delta": delta,
            "moved_toward_actual": delta > 0.0,
            "post_flipped_to_actual": flipped,
            "pre_p_ko_tko": pre["p_ko_tko"],
            "post_p_ko_tko": post["p_ko_tko"],
            "pre_p_sub": pre["p_sub"],
            "post_p_sub": post["p_sub"],
            "pre_p_dec": pre["p_dec"],
            "post_p_dec": post["p_dec"],
        })

    out = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 104)
    print("DIRECT ONE-FIGHT POST-FSR VALIDATION MISS SUMMARY")
    print("=" * 104)
    moved = int(out["moved_toward_actual"].sum())
    flipped = int(out["post_flipped_to_actual"].sum())
    print(f"bouts:                     {len(out)}/{len(misses)}")
    print(f"moved toward actual winner: {moved}/{len(out)} ({moved/len(out):.1%})")
    print(f"flipped to actual winner:   {flipped}/{len(out)} ({flipped/len(out):.1%})")
    print(f"mean actual-winner change:   {100*out['actual_winner_delta'].mean():+.2f} pp")
    print(f"median actual-winner change: {100*out['actual_winner_delta'].median():+.2f} pp")

    compact = out[[
        "red", "blue", "actual_winner", "pre_p_actual_winner",
        "post_p_actual_winner", "actual_winner_delta", "post_flipped_to_actual",
    ]].copy()
    compact["pre_p_actual_winner"] = compact["pre_p_actual_winner"].map(lambda x: f"{x:.1%}")
    compact["post_p_actual_winner"] = compact["post_p_actual_winner"].map(lambda x: f"{x:.1%}")
    compact["actual_winner_delta"] = compact["actual_winner_delta"].map(lambda x: f"{100*x:+.1f} pp")
    print("\n" + compact.to_string(index=False))
    print(f"\nwrote: {args.output}")
    print("No FSR rating replay was performed.")


if __name__ == "__main__":
    main()
