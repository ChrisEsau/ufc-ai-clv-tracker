"""Aligned mature 2020+ population audit for phase-stamina V3.2.

Preserves the existing population-audit cohort, locked age mechanic, strong KD
collapse candidate, and paired seed schedule while substituting only the V3.2
phase-aware stamina engine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as baseline
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_2_phase_stamina as phase_stamina

DEFAULT_PATHS = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_phase_stamina_v32_audit_aligned.csv"
)
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

    ko_count = r1_ko_count = r_ko = b_ko = actual_winner_ko = 0
    finish_rounds: list[int] = []
    r_stamina_end = b_stamina_end = 0.0
    r_spent = b_spent = 0.0
    r_recovered = b_recovered = 0.0
    r_fatigue_penalty = b_fatigue_penalty = 0.0
    r_effective_power = b_effective_power = 0.0

    for seed in seeds:
        sim = phase_stamina.StaticFSRMCKOTKOV32PhaseStamina(
            red,
            blue,
            collapse=STRONG_COLLAPSE,
            rounds=rounds,
            seed=int(seed),
            red_age=r_age,
            blue_age=b_age,
        )
        result = sim.run()

        r_stamina_end += sim.stamina_state[0].fraction
        b_stamina_end += sim.stamina_state[1].fraction
        r_spent += sim.total_stamina_spent[0]
        b_spent += sim.total_stamina_spent[1]
        r_recovered += sim.total_stamina_recovered[0]
        b_recovered += sim.total_stamina_recovered[1]
        r_fatigue_penalty += sim.fatigue_penalty(0)
        b_fatigue_penalty += sim.fatigue_penalty(1)
        r_effective_power += float(sim._effective_profile(0)["striking_power"])
        b_effective_power += float(sim._effective_profile(1)["striking_power"])

        finish = result.finish
        if finish is None:
            continue
        ko_count += 1
        finish_rounds.append(int(finish.round))
        if int(finish.round) == 1:
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
    predicted = "tie"
    if p_r_ko > p_b_ko:
        predicted = r_id
    elif p_b_ko > p_r_ko:
        predicted = b_id

    return {
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "r_id": r_id,
        "b_id": b_id,
        "winner_id": winner_id,
        "actual_ko_tko": int(bout["actual_ko_tko"]),
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "actual_finish_round": bout["actual_finish_round"],
        "p_any_ko": ko_count / paths,
        "p_r1_ko": r1_ko_count / paths,
        "p_r_ko": p_r_ko,
        "p_b_ko": p_b_ko,
        "p_actual_winner_ko": actual_winner_ko / paths if winner_id in {r_id, b_id} else np.nan,
        "predicted_ko_winner": predicted,
        "ko_winner_direction_hit": (
            int(predicted == winner_id)
            if int(bout["actual_ko_tko"]) == 1 and predicted != "tie" and winner_id in {r_id, b_id}
            else np.nan
        ),
        "ko_winner_direction_tie": (
            int(predicted == "tie")
            if int(bout["actual_ko_tko"]) == 1 and winner_id in {r_id, b_id}
            else np.nan
        ),
        "mean_sim_finish_round": float(np.mean(finish_rounds)) if finish_rounds else np.nan,
        "mean_r_final_stamina_fraction": r_stamina_end / paths,
        "mean_b_final_stamina_fraction": b_stamina_end / paths,
        "mean_r_stamina_spent": r_spent / paths,
        "mean_b_stamina_spent": b_spent / paths,
        "mean_r_stamina_recovered": r_recovered / paths,
        "mean_b_stamina_recovered": b_recovered / paths,
        "mean_r_final_fatigue_penalty": r_fatigue_penalty / paths,
        "mean_b_final_fatigue_penalty": b_fatigue_penalty / paths,
        "mean_r_final_effective_power": r_effective_power / paths,
        "mean_b_final_effective_power": b_effective_power / paths,
    }


def _print_summary(frame: pd.DataFrame, paths: int, rounds: int) -> None:
    print("\n" + "=" * 132)
    print("MATURE 2020+ ALIGNED MC AUDIT — PHASE-STAMINA V3.2 + STRONG KD COLLAPSE + LOCKED AGE")
    print("=" * 132)
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

    actual_rounds = frame.loc[frame["actual_ko_tko"].eq(1), "actual_finish_round"].dropna()
    sim_rounds = frame["mean_sim_finish_round"].dropna()
    print("\nFINISH TIMING")
    print(f"actual KO mean finish round: {actual_rounds.mean():.3f}")
    print(f"mean bout-level simulated finish round: {sim_rounds.mean():.3f}")

    print("\nSTAMINA / EFFECTIVE POWER STATE")
    print(
        "mean final stamina fraction: "
        f"red={frame['mean_r_final_stamina_fraction'].mean():.3f}, "
        f"blue={frame['mean_b_final_stamina_fraction'].mean():.3f}"
    )
    print(
        "mean final fatigue penalty: "
        f"red={frame['mean_r_final_fatigue_penalty'].mean():.3f}, "
        f"blue={frame['mean_b_final_fatigue_penalty'].mean():.3f}"
    )
    print(
        "mean final effective striking power: "
        f"red={frame['mean_r_final_effective_power'].mean():.3f}, "
        f"blue={frame['mean_b_final_effective_power'].mean():.3f}"
    )


def main() -> None:
    args = _parse_args()
    cohort, pairs = cohort32.build_aligned_cohort()
    print(f"aligned FSR-32 mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {len(cohort) * args.paths:,}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    completed = 0
    total = len(cohort) * args.paths

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
        completed += args.paths
        if completed % 1000 == 0 or bout_no == len(cohort):
            print(f"[phase-stamina V3.2 MC] paths {completed:,}/{total:,}; bouts {bout_no:,}/{len(cohort):,}", flush=True)

    result = pd.DataFrame(rows)
    _print_summary(result, args.paths, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {len(result):,} bout rows to {args.output}")
    print("FSR-28, V3.1, and the recovery baseline remain unchanged.")


if __name__ == "__main__":
    main()
