"""Attribute wrong-direction poly2 MC moves to individual FSR traits.

Uses the corrected include-target-prefight poly2 construction. For every fight in
which the 34-fight poly2 diagnostic moved probability AWAY from the actual winner:

1. rebuild the full poly2 fight-night profiles;
2. rank traits by the largest absolute red/blue forecast delta;
3. for the top-N changed traits, revert that trait for BOTH fighters to the stored
   target-fight prefight FSR while leaving every other poly2 trait unchanged;
4. rerun the matchup with the SAME seed stream as the full poly2 attribution run;
5. measure the paired change in actual-winner probability.

Interpretation:
    impact_on_actual_winner = P(actual | full poly2) - P(actual | trait reverted)

Negative impact means the poly2 forecast for that trait HURT the actual winner.
Positive impact means it HELPED the actual winner even though the fight moved the
wrong direction overall.

Research only. No stored FSR, simulator config, or production state is modified.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Importing this module installs the corrected include-target-prefight forecast
# function onto the base experiment module.
from scripts.experimental import run_34fight_poly2_include_target_prefight_mc_test as corrected
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import build_fsr_canonical_database as canonical

RESULT_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "fsr_mc_card_validation_poly2_include_target_v1.csv"
)
OUTPUT_DIR = Path("data/experimental/poly2_wrong_direction_trait_attribution")
STRUCTURAL_PATH = OUTPUT_DIR / "wrong_direction_trait_shifts.csv"
ATTRIBUTION_PATH = OUTPUT_DIR / "wrong_direction_trait_attribution.csv"
FIGHT_SUMMARY_PATH = OUTPUT_DIR / "wrong_direction_fight_summary.csv"
DEFAULT_PATHS = 200
DEFAULT_TOP_TRAITS = 8
DEFAULT_SEED = 20260812


def _actual_prob(actual: str, red_name: str, p_red: float) -> float:
    if actual == red_name:
        return float(p_red)
    return float(1.0 - p_red)


def _simulate_p_red(red: pd.Series, blue: pd.Series, seeds: np.ndarray) -> float:
    wins_red = 0
    for seed in seeds:
        sim = full.StaticFSRMCFullFightV1(red, blue, rounds=3, seed=int(seed))
        path = sim.run()
        wins_red += int(path.winner == 0)
    return wins_red / float(len(seeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--top-traits", type=int, default=DEFAULT_TOP_TRAITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.top_traits <= 0:
        raise ValueError("--top-traits must be positive")
    if not args.results.exists():
        raise FileNotFoundError(args.results)

    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results).copy()
    results["bout_id"] = results["bout_id"].astype(str)
    results["event_date"] = pd.to_datetime(results["event_date"], errors="raise")
    wrong = results.loc[results["delta_actual_winner_probability"] < 0].copy()
    wrong = wrong.sort_values("delta_actual_winner_probability")
    print(
        f"[poly2 attribution] wrong-direction fights={len(wrong)} | "
        f"paths={args.paths} | top traits/fight={args.top_traits}",
        flush=True,
    )

    fsr = experiment._prepare_fsr_history()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    structural_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    fight_rows: list[dict[str, object]] = []

    for fight_idx, (_, row) in enumerate(wrong.iterrows(), start=1):
        fight_started = time.perf_counter()
        bout_id = str(row["bout_id"])
        target_date = pd.Timestamp(row["event_date"])
        red_base, blue_base = pairs[bout_id]
        red_name = base._display_name(red_base)
        blue_name = base._display_name(blue_base)
        actual = str(row["actual_winner"])

        red_poly, red_audit = corrected._forecast_profile_include_target(
            red_base, fsr, bout_id, target_date, red_name
        )
        blue_poly, blue_audit = corrected._forecast_profile_include_target(
            blue_base, fsr, bout_id, target_date, blue_name
        )
        ra = pd.DataFrame(red_audit).set_index("trait")
        ba = pd.DataFrame(blue_audit).set_index("trait")

        ranking: list[tuple[str, float]] = []
        for trait in canonical.CANONICAL_RATINGS:
            rd = float(ra.loc[trait, "mc_delta"])
            bd = float(ba.loc[trait, "mc_delta"])
            score = max(abs(rd), abs(bd))
            ranking.append((trait, score))
            structural_rows.append({
                "bout_id": bout_id,
                "red": red_name,
                "blue": blue_name,
                "actual_winner": actual,
                "card_delta_actual_winner_probability": float(row["delta_actual_winner_probability"]),
                "trait": trait,
                "red_prefight_fsr": float(ra.loc[trait, "aligned_latest_fsr"]),
                "red_poly2_fsr": float(ra.loc[trait, "mc_fsr"]),
                "red_delta": rd,
                "blue_prefight_fsr": float(ba.loc[trait, "aligned_latest_fsr"]),
                "blue_poly2_fsr": float(ba.loc[trait, "mc_fsr"]),
                "blue_delta": bd,
                "max_abs_fighter_delta": score,
            })

        ranking.sort(key=lambda x: x[1], reverse=True)
        candidates = [trait for trait, _ in ranking[: args.top_traits]]

        # Paired reference using exactly the same seeds as every reversion below.
        full_p_red = _simulate_p_red(red_poly, blue_poly, seeds)
        full_actual_p = _actual_prob(actual, red_name, full_p_red)

        print(
            f"\n[{fight_idx:02d}/{len(wrong)}] {red_name} vs {blue_name} | actual={actual} | "
            f"card Δ={float(row['delta_actual_winner_probability']):+.1%} | "
            f"paired full P(actual)={full_actual_p:.1%}",
            flush=True,
        )

        for trait_idx, trait in enumerate(candidates, start=1):
            red_revert = red_poly.copy(deep=True)
            blue_revert = blue_poly.copy(deep=True)
            red_revert[trait] = red_base[trait]
            blue_revert[trait] = blue_base[trait]

            revert_p_red = _simulate_p_red(red_revert, blue_revert, seeds)
            revert_actual_p = _actual_prob(actual, red_name, revert_p_red)
            impact = full_actual_p - revert_actual_p

            rd = float(ra.loc[trait, "mc_delta"])
            bd = float(ba.loc[trait, "mc_delta"])
            attribution_rows.append({
                "bout_id": bout_id,
                "red": red_name,
                "blue": blue_name,
                "actual_winner": actual,
                "card_delta_actual_winner_probability": float(row["delta_actual_winner_probability"]),
                "trait": trait,
                "red_delta": rd,
                "blue_delta": bd,
                "paired_full_actual_probability": full_actual_p,
                "trait_reverted_actual_probability": revert_actual_p,
                "impact_on_actual_winner_probability": impact,
                "impact_pp": 100.0 * impact,
                "classification": "HURT" if impact < 0 else ("HELPED" if impact > 0 else "NEUTRAL"),
                "paths": args.paths,
            })
            print(
                f"  {trait_idx:02d}. {trait:34s} red Δ={rd:+6.2f} blue Δ={bd:+6.2f} | "
                f"revert P(actual)={revert_actual_p:6.1%} | impact={impact:+6.1%} "
                f"{'HURT' if impact < 0 else 'HELP' if impact > 0 else 'FLAT'}",
                flush=True,
            )

        fight_attr = [r for r in attribution_rows if r["bout_id"] == bout_id]
        worst = sorted(fight_attr, key=lambda r: float(r["impact_on_actual_winner_probability"]))[:3]
        fight_rows.append({
            "bout_id": bout_id,
            "red": red_name,
            "blue": blue_name,
            "actual_winner": actual,
            "card_delta_actual_winner_probability": float(row["delta_actual_winner_probability"]),
            "paired_full_actual_probability": full_actual_p,
            "worst_trait_1": worst[0]["trait"] if len(worst) > 0 else None,
            "worst_trait_1_impact_pp": worst[0]["impact_pp"] if len(worst) > 0 else None,
            "worst_trait_2": worst[1]["trait"] if len(worst) > 1 else None,
            "worst_trait_2_impact_pp": worst[1]["impact_pp"] if len(worst) > 1 else None,
            "worst_trait_3": worst[2]["trait"] if len(worst) > 2 else None,
            "worst_trait_3_impact_pp": worst[2]["impact_pp"] if len(worst) > 2 else None,
        })
        print(
            f"  completed fight in {time.perf_counter() - fight_started:.1f}s | "
            f"total {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    structural = pd.DataFrame(structural_rows)
    attribution = pd.DataFrame(attribution_rows)
    fight_summary = pd.DataFrame(fight_rows)
    structural.to_csv(STRUCTURAL_PATH, index=False)
    attribution.to_csv(ATTRIBUTION_PATH, index=False)
    fight_summary.to_csv(FIGHT_SUMMARY_PATH, index=False)

    print("\n" + "=" * 125)
    print("WRONG-DIRECTION POLY2 TRAIT ATTRIBUTION — MOST HARMFUL TRAITS")
    print("=" * 125)
    show = attribution.sort_values(
        ["bout_id", "impact_on_actual_winner_probability"], ascending=[True, True]
    ).groupby("bout_id", group_keys=False).head(5)
    print(
        show[[
            "red", "blue", "actual_winner", "trait", "red_delta", "blue_delta",
            "impact_pp", "classification",
        ]].to_string(index=False, float_format=lambda v: f"{v:.2f}"),
        flush=True,
    )

    print("\nTRAITS MOST OFTEN HARMFUL ACROSS WRONG-DIRECTION FIGHTS")
    harmful = attribution.loc[attribution["impact_on_actual_winner_probability"] < 0].copy()
    if harmful.empty:
        print("none")
    else:
        agg = harmful.groupby("trait", as_index=False).agg(
            harmful_fights=("bout_id", "nunique"),
            mean_harm_pp=("impact_pp", "mean"),
            total_harm_pp=("impact_pp", "sum"),
        ).sort_values(["harmful_fights", "total_harm_pp"], ascending=[False, True])
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\nwrote: {STRUCTURAL_PATH}")
    print(f"wrote: {ATTRIBUTION_PATH}")
    print(f"wrote: {FIGHT_SUMMARY_PATH}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")
    print("Negative impact_pp means that trait's poly2 forecast hurt the actual winner.")


if __name__ == "__main__":
    main()
