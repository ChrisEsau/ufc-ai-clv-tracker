"""Full-cohort neutral-power baseline for current V3.3 KO/TKO physics.

Research-only audit. No simulator or FSR artifact is modified.

Purpose
-------
Establish the true current-engine baseline before adding fighter-specific
striking_power back into strike severity.

Neutral power means:
- heavy-tail probability is fixed at the base 6% for every fighter;
- heavy-tail magnitude is not multiplied by striking_power.

Everything else stays current and unchanged:
- rebuilt FSR-32 fighter profiles for all other traits
- V3.3 global recovery
- KD shock coefficient = 90
- strong KD collapse = (5.0, 2.0)
- three-round horizon
- aligned mature 2020+ cohort
- deterministic seeds

R1 KD metrics are measured by replaying each seed with a one-round horizon, so
we do not depend on damage-event round metadata.
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

HISTORICAL_ANY_KO_RATE = 0.3144
HISTORICAL_R1_KO_RATE = 0.1406
HISTORICAL_MEAN_KO_ROUND = 1.835
HISTORICAL_MEAN_R1_KD = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000
HISTORICAL_MEAN_KD = 0.4364
HISTORICAL_ANY_KD_RATE = 0.3578


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


class NeutralPowerV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """Current V3.3 physics with fighter-specific power removed from severity."""

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

    def _tail_probability(self, attacker: int) -> float:
        return float(damage.POWER_TAIL_BASE_PROBABILITY)

    def _draw_strike_damage(self, attacker: int) -> float:
        raw_damage = float(
            self.rng.gamma(
                damage.BASE_SEVERITY_GAMMA_SHAPE,
                damage.BASE_SEVERITY_GAMMA_SCALE,
            )
        )
        if self.rng.random() < self._tail_probability(attacker):
            raw_damage += float(
                self.rng.gamma(
                    damage.TAIL_SEVERITY_GAMMA_SHAPE,
                    damage.TAIL_SEVERITY_GAMMA_SCALE,
                )
            )
        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)


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
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

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
            kwargs = dict(
                collapse=STRONG_COLLAPSE,
                seed=int(seed),
                red_age=r_age,
                blue_age=b_age,
            )

            sim = NeutralPowerV33(red, blue, rounds=3, **kwargs)
            result = sim.run()

            path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
            kd_total += path_kd
            paths_with_kd += int(path_kd > 0)

            if result.finish is not None:
                ko_count += 1
                finish_round = int(getattr(result.finish, "round", 0) or 0)
                ko_round_sum += finish_round
                r1_ko_count += int(finish_round == 1)

            # Exact prefix replay for Round 1.
            sim_r1 = NeutralPowerV33(red, blue, rounds=1, **kwargs)
            sim_r1.run()
            path_r1_kd = (
                int(sim_r1.stats[0].knockdowns_scored)
                + int(sim_r1.stats[1].knockdowns_scored)
            )
            r1_kd_total += path_r1_kd
            paths_with_r1_kd += int(path_r1_kd > 0)

        completed += args.paths
        if completed % 1000 == 0 or bout_idx + 1 == len(cohort):
            print(
                f"paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}",
                flush=True,
            )

    any_ko = ko_count / total_paths
    r1_ko = r1_ko_count / total_paths
    mean_ko_round = ko_round_sum / ko_count if ko_count else float("nan")
    mean_kd = kd_total / total_paths
    any_kd = paths_with_kd / total_paths
    mean_r1_kd = r1_kd_total / total_paths
    any_r1_kd = paths_with_r1_kd / total_paths

    print("\n" + "=" * 112)
    print("V3.3 MATURE 2020+ FULL-COHORT BASELINE — NEUTRALIZED STRIKING POWER")
    print("=" * 112)
    print(f"bouts: {len(cohort):,}")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {total_paths:,}")
    print(f"seed: {args.seed}")
    print(f"base heavy-tail probability: {damage.POWER_TAIL_BASE_PROBABILITY:.2%}")
    print("tail magnitude power multiplier: OFF")
    print(f"KD shock coefficient: {KD_SHOCK_COEFFICIENT:.0f}")
    print("KD collapse: strong (5.0, 2.0)")

    print("\nRESULT                            HISTORICAL      SIMULATED       DELTA")
    print(f"Any KO/TKO rate                  {HISTORICAL_ANY_KO_RATE:10.2%} {any_ko:14.2%} {any_ko-HISTORICAL_ANY_KO_RATE:+11.2%}")
    print(f"R1 KO/TKO rate                   {HISTORICAL_R1_KO_RATE:10.2%} {r1_ko:14.2%} {r1_ko-HISTORICAL_R1_KO_RATE:+11.2%}")
    print(f"Mean KO finish round             {HISTORICAL_MEAN_KO_ROUND:10.3f} {mean_ko_round:14.3f} {mean_ko_round-HISTORICAL_MEAN_KO_ROUND:+11.3f}")
    print(f"Mean R1 KDs/fight                {HISTORICAL_MEAN_R1_KD:10.4f} {mean_r1_kd:14.4f} {mean_r1_kd-HISTORICAL_MEAN_R1_KD:+11.4f}")
    print(f"Fight/path with >=1 R1 KD        {HISTORICAL_ANY_R1_KD_RATE:10.2%} {any_r1_kd:14.2%} {any_r1_kd-HISTORICAL_ANY_R1_KD_RATE:+11.2%}")
    print(f"Mean KDs/fight                   {HISTORICAL_MEAN_KD:10.4f} {mean_kd:14.4f} {mean_kd-HISTORICAL_MEAN_KD:+11.4f}")
    print(f"Fight/path with >=1 KD           {HISTORICAL_ANY_KD_RATE:10.2%} {any_kd:14.2%} {any_kd-HISTORICAL_ANY_KD_RATE:+11.2%}")
    print("\nNeutral means striking_power does not alter strike-severity probability or magnitude.")
    print("Research only: no simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
