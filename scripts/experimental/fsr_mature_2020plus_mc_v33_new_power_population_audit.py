"""Full mature 2020+ MC audit using the rebuilt rolling FSR-32 striking_power.

This is an evaluation harness only. It does not modify simulator physics.

Locked simulation configuration:
- V3.3 global recovery
- current FSR-32 prefight parquet (therefore rebuilt rolling striking_power)
- strong KD collapse: scale=5.0, curvature=2.0
- KD shock coefficient=90
- three-round horizon
- established aligned mature 2020+ cohort
- deterministic shared seeds

The purpose is to measure how the new fresh-power trait changes KO/TKO and KD
population behavior before any further damage/KD calibration is attempted.
"""
from __future__ import annotations

import argparse
from math import exp

import numpy as np

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
KD_SHOCK_COEFFICIENT = 90.0
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

# Established exact-cohort historical references.
HISTORICAL_ANY_KO_RATE = 0.3144
HISTORICAL_R1_KO_RATE = 0.1406
HISTORICAL_MEAN_KO_ROUND = 1.835
HISTORICAL_MEAN_R1_KD = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000
HISTORICAL_ANY_KD_RATE = 0.3578
HISTORICAL_MEAN_KD = 0.4364


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


class NewPowerV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """V3.3 with the previously selected KD shock coefficient fixed at 90."""

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        logit_p = (
            damage.KD_BASE_LOGIT
            + KD_SHOCK_COEFFICIENT * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + damage.KD_DEPLETION_COEFFICIENT * depletion
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bouts", type=int, default=0, help="0 = full aligned cohort")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    if args.bouts > 0:
        cohort = cohort.head(args.bouts).reset_index(drop=True)

    total_paths = len(cohort) * args.paths
    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    ko_count = 0
    r1_ko_count = 0
    ko_round_sum = 0.0
    kd_total = 0
    paths_with_kd = 0
    r1_kd_total = 0
    paths_with_r1_kd = 0

    completed = 0
    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for seed in seed_matrix[bout_idx]:
            sim = NewPowerV33(
                red,
                blue,
                collapse=STRONG_COLLAPSE,
                rounds=3,
                seed=int(seed),
                red_age=r_age,
                blue_age=b_age,
            )
            result = sim.run()

            r_kd = int(sim.stats[0].knockdowns_scored)
            b_kd = int(sim.stats[1].knockdowns_scored)
            total_path_kd = r_kd + b_kd
            kd_total += total_path_kd
            if total_path_kd > 0:
                paths_with_kd += 1

            # Prefer explicit segment/round KD event bookkeeping when available.
            r1_path_kd = 0
            if hasattr(sim, "damage_events"):
                for event in sim.damage_events:
                    if bool(event.get("knockdown", False)) and int(event.get("round", 0)) == 1:
                        r1_path_kd += 1
            elif hasattr(sim, "knockdown_events"):
                for event in sim.knockdown_events:
                    if int(event.get("round", 0)) == 1:
                        r1_path_kd += 1
            r1_kd_total += r1_path_kd
            if r1_path_kd > 0:
                paths_with_r1_kd += 1

            if result.finish is not None:
                ko_count += 1
                finish_round = int(getattr(result.finish, "round", 0) or 0)
                if finish_round <= 0 and isinstance(result.finish, dict):
                    finish_round = int(result.finish.get("round", 0) or 0)
                if finish_round > 0:
                    ko_round_sum += finish_round
                    if finish_round == 1:
                        r1_ko_count += 1

        completed += args.paths
        if completed % 1000 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}", flush=True)

    any_ko = ko_count / total_paths
    r1_ko = r1_ko_count / total_paths
    mean_ko_round = ko_round_sum / ko_count if ko_count else float("nan")
    mean_kd = kd_total / total_paths
    any_kd = paths_with_kd / total_paths
    mean_r1_kd = r1_kd_total / total_paths
    any_r1_kd = paths_with_r1_kd / total_paths

    print("\n" + "=" * 112)
    print("V3.3 MATURE 2020+ POPULATION AUDIT — REBUILT FSR-32 STRIKING POWER")
    print("=" * 112)
    print(f"bouts: {len(cohort):,}")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {total_paths:,}")
    print(f"seed: {args.seed}")
    print(f"KD shock coefficient: {KD_SHOCK_COEFFICIENT:.0f}")
    print("KD collapse: strong (5.0, 2.0)")
    print("FSR source: current fsr_32_prefight_snapshots.parquet")

    print("\nRESULT                            HISTORICAL      SIMULATED       DELTA")
    print(f"Any KO/TKO rate                  {HISTORICAL_ANY_KO_RATE:10.2%} {any_ko:14.2%} {any_ko-HISTORICAL_ANY_KO_RATE:+11.2%}")
    print(f"R1 KO/TKO rate                   {HISTORICAL_R1_KO_RATE:10.2%} {r1_ko:14.2%} {r1_ko-HISTORICAL_R1_KO_RATE:+11.2%}")
    print(f"Mean KO finish round             {HISTORICAL_MEAN_KO_ROUND:10.3f} {mean_ko_round:14.3f} {mean_ko_round-HISTORICAL_MEAN_KO_ROUND:+11.3f}")
    print(f"Mean R1 KDs/fight                {HISTORICAL_MEAN_R1_KD:10.4f} {mean_r1_kd:14.4f} {mean_r1_kd-HISTORICAL_MEAN_R1_KD:+11.4f}")
    print(f"Fight/path with >=1 R1 KD        {HISTORICAL_ANY_R1_KD_RATE:10.2%} {any_r1_kd:14.2%} {any_r1_kd-HISTORICAL_ANY_R1_KD_RATE:+11.2%}")
    print(f"Mean KDs/fight                   {HISTORICAL_MEAN_KD:10.4f} {mean_kd:14.4f} {mean_kd-HISTORICAL_MEAN_KD:+11.4f}")
    print(f"Fight/path with >=1 KD           {HISTORICAL_ANY_KD_RATE:10.2%} {any_kd:14.2%} {any_kd-HISTORICAL_ANY_KD_RATE:+11.2%}")

    print("\nThis harness changes no simulator physics. Any movement versus prior V3.3 audits is attributable to the rebuilt FSR-32 inputs (principally striking_power) plus the explicitly fixed shock=90 setting.")


if __name__ == "__main__":
    main()
