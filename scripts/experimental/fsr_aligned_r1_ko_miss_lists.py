"""Build aligned R1 KO/TKO occurrence and direction miss lists.

This diagnostic replays only the historical bouts that actually ended by R1
KO/TKO, while consuming the same per-bout seed stream as the mature 2020+
10-path population audit.  That makes its 10-path probabilities directly
comparable to the aligned recovery population run.

Two failure sets are printed:
1. occurrence misses: actual R1 KO/TKO, but zero simulated R1 KO paths;
2. directional misses: among actual R1 KO/TKOs, the actual loser has more
   simulated R1-KO wins than the actual winner.

Ties are reported separately.  Stored FSR values and simulator constants are
never modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_historical_corner_alignment as alignment
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as population
from scripts.experimental import fsr_mature_2020plus_mc_10path_round_recovery_audit as recovery_audit
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/aligned_r1_ko_miss_lists.csv"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _name(profile: pd.Series) -> str:
    return base._display_name(profile)


def _simulate_r1_bout(
    bout: pd.Series,
    pair: tuple[pd.Series, pd.Series],
    *,
    paths: int,
    rounds: int,
    seeds: np.ndarray,
) -> dict[str, object]:
    red, blue = pair
    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    winner_id = str(bout.get("winner_id", ""))
    r_age = float(bout["r_age"]) if pd.notna(bout["r_age"]) else None
    b_age = float(bout["b_age"]) if pd.notna(bout["b_age"]) else None

    red_r1_ko = 0
    blue_r1_ko = 0
    red_any_ko = 0
    blue_any_ko = 0

    for seed in seeds:
        sim = recovery.StaticFSRMCKOTKOV2RoundRecovery(
            red,
            blue,
            collapse=STRONG_COLLAPSE,
            rounds=rounds,
            seed=int(seed),
            red_age=r_age,
            blue_age=b_age,
        )
        result = sim.run()
        finish = result.finish
        if finish is None:
            continue
        if finish.winner == 0:
            red_any_ko += 1
            if int(finish.round) == 1:
                red_r1_ko += 1
        else:
            blue_any_ko += 1
            if int(finish.round) == 1:
                blue_r1_ko += 1

    p_red_r1 = red_r1_ko / paths
    p_blue_r1 = blue_r1_ko / paths
    p_any_r1 = p_red_r1 + p_blue_r1

    if winner_id == r_id:
        winner_name = _name(red)
        loser_name = _name(blue)
        p_winner_r1 = p_red_r1
        p_loser_r1 = p_blue_r1
        p_winner_any = red_any_ko / paths
        p_loser_any = blue_any_ko / paths
    elif winner_id == b_id:
        winner_name = _name(blue)
        loser_name = _name(red)
        p_winner_r1 = p_blue_r1
        p_loser_r1 = p_red_r1
        p_winner_any = blue_any_ko / paths
        p_loser_any = red_any_ko / paths
    else:
        raise ValueError(
            f"Bout {bout['bout_id']}: winner_id={winner_id} not in aligned corners "
            f"r_id={r_id}, b_id={b_id}"
        )

    if p_winner_r1 > p_loser_r1:
        direction = "RIGHT"
    elif p_winner_r1 < p_loser_r1:
        direction = "WRONG"
    else:
        direction = "TIE"

    return {
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "actual_winner_name": winner_name,
        "actual_loser_name": loser_name,
        "r_name": _name(red),
        "b_name": _name(blue),
        "winner_id": winner_id,
        "r_id": r_id,
        "b_id": b_id,
        "p_r1_ko": p_any_r1,
        "p_actual_winner_r1_ko": p_winner_r1,
        "p_actual_loser_r1_ko": p_loser_r1,
        "winner_minus_loser_r1_margin": p_winner_r1 - p_loser_r1,
        "p_actual_winner_any_ko": p_winner_any,
        "p_actual_loser_any_ko": p_loser_any,
        "occurrence_miss_zero_r1_paths": int(p_any_r1 == 0.0),
        "r1_direction": direction,
    }


def _print_table(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        print("none")
        return
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = population._build_cohort()
    pairs = alignment.align_pair_dict_to_master_corners(cohort, pairs)
    actual_r1 = cohort[cohort["actual_r1_ko"].eq(1)].copy()
    actual_r1_ids = set(actual_r1["bout_id"].astype(str))

    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"historical R1 KO/TKO bouts: {len(actual_r1):,}")
    print(f"paths per selected bout: {args.paths}")
    print("engine: strong KD collapse + locked age + between-round recovery")
    print("alignment: strict fighter_id -> master r_id/b_id")

    # Consume the exact same seed block for every cohort bout as the full audit.
    # Only execute the 220 actual R1 KO/TKO bouts. This preserves comparable
    # per-bout seeds without paying to rerun all 1,565 fights.
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    selected_done = 0
    for _, bout in cohort.iterrows():
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        bout_id = str(bout["bout_id"])
        if bout_id not in actual_r1_ids:
            continue
        rows.append(
            _simulate_r1_bout(
                bout,
                pairs[bout_id],
                paths=args.paths,
                rounds=args.rounds,
                seeds=seeds,
            )
        )
        selected_done += 1
        if selected_done % 25 == 0 or selected_done == len(actual_r1):
            print(
                f"[aligned R1 miss audit] bouts {selected_done:,}/{len(actual_r1):,}; "
                f"paths {selected_done * args.paths:,}/{len(actual_r1) * args.paths:,}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    occurrence = result[result["occurrence_miss_zero_r1_paths"].eq(1)].copy()
    directional = result[result["r1_direction"].eq("WRONG")].copy()
    ties = result[result["r1_direction"].eq("TIE")].copy()
    right = result[result["r1_direction"].eq("RIGHT")].copy()

    # Worst occurrence misses first: lowest actual-winner full-fight KO support,
    # then highest actual-loser full-fight KO support.
    occurrence = occurrence.sort_values(
        ["p_actual_winner_any_ko", "p_actual_loser_any_ko"],
        ascending=[True, False],
    )
    directional = directional.sort_values(
        "winner_minus_loser_r1_margin", ascending=True
    )
    ties = ties.sort_values(
        ["p_actual_winner_r1_ko", "p_actual_winner_any_ko"], ascending=[True, True]
    )

    print("\n" + "=" * 132)
    print("ALIGNED HISTORICAL R1 KO/TKO MISS AUDIT")
    print("=" * 132)
    print(f"actual R1 KO/TKO bouts: {len(result):,}")
    print(
        f"OCCURRENCE MISS (0/{args.paths} simulated R1 KOs): "
        f"{len(occurrence):,} ({len(occurrence)/len(result):.2%})"
    )
    print(
        f"R1 DIRECTION RIGHT: {len(right):,} ({len(right)/len(result):.2%}) | "
        f"WRONG: {len(directional):,} ({len(directional)/len(result):.2%}) | "
        f"TIE: {len(ties):,} ({len(ties)/len(result):.2%})"
    )
    non_tie = len(right) + len(directional)
    if non_tie:
        print(f"R1 direction hit rate among non-ties: {len(right)/non_tie:.2%}")

    display_cols = [
        "event_date",
        "actual_winner_name",
        "actual_loser_name",
        "p_r1_ko",
        "p_actual_winner_r1_ko",
        "p_actual_loser_r1_ko",
        "winner_minus_loser_r1_margin",
        "p_actual_winner_any_ko",
        "p_actual_loser_any_ko",
        "bout_id",
    ]

    print("\nOCCURRENCE MISSES — ACTUAL R1 KO/TKO BUT MC PRODUCED ZERO R1 KO PATHS")
    print("-" * 132)
    _print_table(occurrence, display_cols)

    print("\nR1 DIRECTIONAL MISSES — ACTUAL LOSER HAD MORE SIMULATED R1-KO WINS")
    print("-" * 132)
    _print_table(directional, display_cols)

    print("\nR1 DIRECTIONAL TIES")
    print("-" * 132)
    _print_table(ties, display_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    occurrence.to_csv(args.output.with_name(args.output.stem + "_occurrence_misses.csv"), index=False)
    directional.to_csv(args.output.with_name(args.output.stem + "_direction_misses.csv"), index=False)
    ties.to_csv(args.output.with_name(args.output.stem + "_direction_ties.csv"), index=False)

    print(f"\nWrote all {len(result):,} aligned actual-R1-KO rows to {args.output}")
    print("Also wrote occurrence-miss, direction-miss, and direction-tie CSV subsets.")
    print("No simulator constants or stored FSR values were changed.")


if __name__ == "__main__":
    main()
