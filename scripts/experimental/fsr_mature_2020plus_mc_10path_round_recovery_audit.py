"""Test provisional between-round reservoir recovery on the mature 2020+ cohort.

Uses the same cohort, path count, seed stream, strong KD-collapse architecture,
and locked age mechanic as ``fsr_mature_2020plus_mc_10path_population_audit.py``.
The only added mechanic is recovery of missing reservoir between completed rounds,
scaled by the existing ``recovery_ability`` FSR trait.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as baseline
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery


DEFAULT_PATHS = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_round_recovery_audit.csv"
)
BASELINE_OUTPUT_PATH = baseline.OUTPUT_PATH
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _simulate_one_bout(
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

    ko_count = 0
    r1_ko_count = 0
    r_ko = 0
    b_ko = 0
    actual_winner_ko = 0
    finish_rounds: list[int] = []
    red_recovered = 0.0
    blue_recovered = 0.0

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
        red_recovered += sim.total_round_recovery[0]
        blue_recovered += sim.total_round_recovery[1]
        finish = result.finish
        if finish is None:
            continue

        ko_count += 1
        finish_rounds.append(int(finish.round))
        if finish.round == 1:
            r1_ko_count += 1
        if finish.winner == 0:
            r_ko += 1
            if winner_id == r_id:
                actual_winner_ko += 1
        else:
            b_ko += 1
            if winner_id == b_id:
                actual_winner_ko += 1

    p_r_ko = r_ko / paths
    p_b_ko = b_ko / paths
    predicted_ko_winner = "tie"
    if p_r_ko > p_b_ko:
        predicted_ko_winner = r_id
    elif p_b_ko > p_r_ko:
        predicted_ko_winner = b_id

    return {
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "r_id": r_id,
        "b_id": b_id,
        "winner_id": winner_id,
        "r_age": r_age,
        "b_age": b_age,
        "max_age": np.nanmax([r_age if r_age is not None else np.nan, b_age if b_age is not None else np.nan]),
        "actual_ko_tko": int(bout["actual_ko_tko"]),
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "actual_finish_round": bout["actual_finish_round"],
        "p_any_ko": ko_count / paths,
        "p_r1_ko": r1_ko_count / paths,
        "p_r_ko": p_r_ko,
        "p_b_ko": p_b_ko,
        "p_actual_winner_ko": actual_winner_ko / paths if winner_id in {r_id, b_id} else np.nan,
        "predicted_ko_winner": predicted_ko_winner,
        "ko_winner_direction_hit": (
            int(predicted_ko_winner == winner_id)
            if int(bout["actual_ko_tko"]) == 1 and predicted_ko_winner != "tie" and winner_id in {r_id, b_id}
            else np.nan
        ),
        "ko_winner_direction_tie": (
            int(predicted_ko_winner == "tie")
            if int(bout["actual_ko_tko"]) == 1 and winner_id in {r_id, b_id}
            else np.nan
        ),
        "mean_sim_finish_round": float(np.mean(finish_rounds)) if finish_rounds else np.nan,
        "mean_red_round_recovery": red_recovered / paths,
        "mean_blue_round_recovery": blue_recovered / paths,
    }


def _print_summary(frame: pd.DataFrame, paths: int, rounds: int) -> None:
    print("\n" + "=" * 124)
    print("MATURE 2020+ MC AUDIT — STRONG KD COLLAPSE + AGE + BETWEEN-ROUND RECOVERY")
    print("=" * 124)
    print(f"bouts: {len(frame):,}")
    print(f"paths per bout: {paths}")
    print(f"total paths: {len(frame) * paths:,}")
    print(f"simulation horizon: {rounds} rounds")
    print(f"actual KO/TKO rate: {frame['actual_ko_tko'].mean():.2%}")
    print(f"simulated any-KO rate: {frame['p_any_ko'].mean():.2%}")
    print(f"actual R1 KO/TKO rate: {frame['actual_r1_ko'].mean():.2%}")
    print(f"simulated R1-KO rate: {frame['p_r1_ko'].mean():.2%}")

    metrics = []
    for label, target, prob in [
        ("Any KO/TKO", "actual_ko_tko", "p_any_ko"),
        ("R1 KO/TKO", "actual_r1_ko", "p_r1_ko"),
    ]:
        metrics.append({"target": label, **baseline._binary_metrics(frame, target, prob)})
    print("\nKO OCCURRENCE METRICS")
    print(pd.DataFrame(metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ko_bouts = frame.loc[frame["actual_ko_tko"].eq(1)].copy()
    valid_dir = ko_bouts["ko_winner_direction_hit"].notna()
    print("\nACTUAL KO/TKO WINNER DIRECTION")
    print(f"actual KO bouts: {len(ko_bouts):,}")
    print(f"non-tie directional calls: {int(valid_dir.sum()):,}")
    print(f"direction hit rate among non-ties: {ko_bouts.loc[valid_dir, 'ko_winner_direction_hit'].mean():.2%}")
    print(f"direction tie rate: {ko_bouts['ko_winner_direction_tie'].mean():.2%}")
    print(f"mean P(actual KO winner scores KO): {ko_bouts['p_actual_winner_ko'].mean():.2%}")

    actual_ko_rounds = frame.loc[frame["actual_ko_tko"].eq(1), "actual_finish_round"].dropna()
    sim_finish_rounds = frame["mean_sim_finish_round"].dropna()
    print("\nFINISH ROUND DIAGNOSTIC")
    print(f"actual KO mean finish round: {actual_ko_rounds.mean():.3f}")
    print(f"mean bout-level simulated finish round: {sim_finish_rounds.mean():.3f}")
    print(
        "mean reservoir restored per bout-path: "
        f"red={frame['mean_red_round_recovery'].mean():.2f}, "
        f"blue={frame['mean_blue_round_recovery'].mean():.2f}"
    )

    if BASELINE_OUTPUT_PATH.exists():
        old = pd.read_csv(BASELINE_OUTPUT_PATH)
        if len(old) == len(frame) and set(old["bout_id"].astype(str)) == set(frame["bout_id"].astype(str)):
            merged = old[["bout_id", "p_any_ko", "p_r1_ko", "mean_sim_finish_round"]].merge(
                frame[["bout_id", "p_any_ko", "p_r1_ko", "mean_sim_finish_round"]],
                on="bout_id",
                suffixes=("_baseline", "_recovery"),
                validate="one_to_one",
            )
            print("\nPAIRED CHANGE VS SAVED NO-RECOVERY AUDIT")
            print(f"baseline any-KO: {merged['p_any_ko_baseline'].mean():.2%}")
            print(f"recovery any-KO: {merged['p_any_ko_recovery'].mean():.2%}")
            print(f"delta any-KO: {merged['p_any_ko_recovery'].mean() - merged['p_any_ko_baseline'].mean():+.2%}")
            print(f"baseline R1-KO: {merged['p_r1_ko_baseline'].mean():.2%}")
            print(f"recovery R1-KO: {merged['p_r1_ko_recovery'].mean():.2%}")
            print(
                "delta mean simulated finish round: "
                f"{merged['mean_sim_finish_round_recovery'].mean() - merged['mean_sim_finish_round_baseline'].mean():+.3f}"
            )

    print("\nNOTE: this is a mechanism test. Recovery percentages are provisional and no KD-based suppression is active.")


def main() -> None:
    args = _parse_args()
    cohort, pairs = baseline._build_cohort()
    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {len(cohort) * args.paths:,}")
    print("engine: strong KD collapse + locked age + between-round recovery")
    print("recovery: 5%-35% of missing reservoir based on recovery_ability; 20% at rating 50")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    completed_paths = 0
    total_paths = len(cohort) * args.paths

    for bout_no, (_, bout) in enumerate(cohort.iterrows(), start=1):
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        rows.append(
            _simulate_one_bout(
                bout,
                pairs[str(bout["bout_id"])],
                paths=args.paths,
                rounds=args.rounds,
                seeds=seeds,
            )
        )
        completed_paths += args.paths
        if completed_paths % 1000 == 0 or bout_no == len(cohort):
            print(
                f"[round recovery MC] paths {completed_paths:,}/{total_paths:,}; bouts {bout_no:,}/{len(cohort):,}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    _print_summary(result, args.paths, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {len(result):,} bout rows to {args.output}")
    print("Stored FSR ratings and the no-recovery simulator remain unchanged.")


if __name__ == "__main__":
    main()
