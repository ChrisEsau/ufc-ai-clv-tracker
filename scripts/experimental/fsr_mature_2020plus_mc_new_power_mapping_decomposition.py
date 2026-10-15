"""Decompose how rebuilt FSR-32 striking_power enters MC damage.

Research-only audit. No simulator or FSR artifact is modified.

The rebuilt power trait has a much wider range than the old compressed trait.
V3.1 currently maps power into strike damage in two separate places:

1. heavy-tail probability
2. heavy-tail magnitude

This harness isolates those effects on the established mature 2020+ cohort.

Candidates:
- both:       scale=50 tail probability + magnitude scale=55
- frequency:  scale=50 tail probability + flat tail magnitude
- magnitude:  constant 6% tail probability + magnitude scale=55
- neutral:    constant 6% tail probability + flat tail magnitude

All other V3.3 physics stay fixed, including shock=90 and strong KD collapse.
R1 KD metrics are measured with a one-round replay of the same path seed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp

import numpy as np

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
KD_SHOCK_COEFFICIENT = 90.0
TAIL_RATING_SCALE = 50.0
TAIL_MAGNITUDE_SCALE = 55.0
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

HISTORICAL_ANY_KO_RATE = 0.3144
HISTORICAL_R1_KO_RATE = 0.1406
HISTORICAL_MEAN_KO_ROUND = 1.835
HISTORICAL_MEAN_R1_KD = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000
HISTORICAL_MEAN_KD = 0.4364
HISTORICAL_ANY_KD_RATE = 0.3578


@dataclass(frozen=True)
class Candidate:
    name: str
    power_tail_probability: bool
    power_tail_magnitude: bool


CANDIDATES = (
    Candidate("both", True, True),
    Candidate("frequency_only", True, False),
    Candidate("magnitude_only", False, True),
    Candidate("neutral", False, False),
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


class MappingDiagnosticV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    def __init__(self, *args, candidate: Candidate, **kwargs) -> None:
        self.mapping_candidate = candidate
        super().__init__(*args, **kwargs)

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
        if not self.mapping_candidate.power_tail_probability:
            return float(damage.POWER_TAIL_BASE_PROBABILITY)
        power = base._value(self.fighters[attacker], "striking_power")
        return damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / TAIL_RATING_SCALE
        )

    def _draw_strike_damage(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
        raw_damage = float(
            self.rng.gamma(
                damage.BASE_SEVERITY_GAMMA_SHAPE,
                damage.BASE_SEVERITY_GAMMA_SCALE,
            )
        )
        if self.rng.random() < self._tail_probability(attacker):
            tail = float(
                self.rng.gamma(
                    damage.TAIL_SEVERITY_GAMMA_SHAPE,
                    damage.TAIL_SEVERITY_GAMMA_SCALE,
                )
            )
            if self.mapping_candidate.power_tail_magnitude:
                tail *= exp((power - 50.0) / TAIL_MAGNITUDE_SCALE)
            raw_damage += tail
        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

    results = []
    for candidate in CANDIDATES:
        total_paths = len(cohort) * args.paths
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
                    candidate=candidate,
                )

                sim = MappingDiagnosticV33(red, blue, rounds=3, **kwargs)
                result = sim.run()

                path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
                kd_total += path_kd
                paths_with_kd += int(path_kd > 0)

                if result.finish is not None:
                    ko_count += 1
                    finish_round = int(getattr(result.finish, "round", 0) or 0)
                    ko_round_sum += finish_round
                    r1_ko_count += int(finish_round == 1)

                # Same seed, one-round horizon gives exact R1 KD count without
                # depending on damage-event round metadata.
                sim_r1 = MappingDiagnosticV33(red, blue, rounds=1, **kwargs)
                sim_r1.run()
                path_r1_kd = (
                    int(sim_r1.stats[0].knockdowns_scored)
                    + int(sim_r1.stats[1].knockdowns_scored)
                )
                r1_kd_total += path_r1_kd
                paths_with_r1_kd += int(path_r1_kd > 0)

            completed += args.paths
            if completed % 500 == 0 or bout_idx + 1 == len(cohort):
                print(
                    f"[{candidate.name}] paths {completed:,}/{total_paths:,}; "
                    f"bouts {bout_idx + 1:,}/{len(cohort):,}",
                    flush=True,
                )

        results.append(
            dict(
                name=candidate.name,
                any_ko=ko_count / total_paths,
                r1_ko=r1_ko_count / total_paths,
                mean_round=ko_round_sum / ko_count if ko_count else float("nan"),
                mean_r1_kd=r1_kd_total / total_paths,
                any_r1_kd=paths_with_r1_kd / total_paths,
                mean_kd=kd_total / total_paths,
                any_kd=paths_with_kd / total_paths,
            )
        )

    print("\n" + "=" * 116)
    print("NEW FSR-32 POWER MAPPING DECOMPOSITION — 200-BOUT SCREEN")
    print("=" * 116)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; seed: {args.seed}")
    print(f"tail probability scale when active: {TAIL_RATING_SCALE:.0f}")
    print(f"tail magnitude scale when active: {TAIL_MAGNITUDE_SCALE:.0f}")
    print(f"base tail probability when power-frequency disabled: {damage.POWER_TAIL_BASE_PROBABILITY:.2%}")
    print(f"KD shock: {KD_SHOCK_COEFFICIENT:.0f}; collapse: strong (5.0, 2.0)")
    print("\nHISTORICAL TARGETS")
    print(
        f"any KO={HISTORICAL_ANY_KO_RATE:.2%}; R1 KO={HISTORICAL_R1_KO_RATE:.2%}; "
        f"mean KO round={HISTORICAL_MEAN_KO_ROUND:.3f}; mean R1 KD={HISTORICAL_MEAN_R1_KD:.4f}; "
        f"any R1 KD={HISTORICAL_ANY_R1_KD_RATE:.2%}; mean KD={HISTORICAL_MEAN_KD:.4f}; "
        f"any KD={HISTORICAL_ANY_KD_RATE:.2%}"
    )
    print("\nRESULTS")
    print(
        f"{'candidate':>16} {'any_KO':>9} {'R1_KO':>9} {'meanRnd':>9} "
        f"{'meanR1KD':>10} {'anyR1KD':>10} {'meanKD':>9} {'anyKD':>9}"
    )
    for row in results:
        print(
            f"{row['name']:>16} {row['any_ko']:9.2%} {row['r1_ko']:9.2%} "
            f"{row['mean_round']:9.3f} {row['mean_r1_kd']:10.4f} "
            f"{row['any_r1_kd']:10.2%} {row['mean_kd']:9.4f} {row['any_kd']:9.2%}"
        )

    print("\nInterpretation:")
    print("  both           = power changes tail frequency and magnitude")
    print("  frequency_only = power changes tail frequency; tail magnitude is flat")
    print("  magnitude_only = tail frequency fixed at 6%; power changes tail magnitude")
    print("  neutral        = power does not affect strike severity; physics baseline")
    print("Research only: no simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
