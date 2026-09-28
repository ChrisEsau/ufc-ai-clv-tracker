"""Research-only MC calibration sweep for the rebuilt FSR-32 striking_power scale.

The rebuilt power trait is intentionally preserved. This harness changes only the
translation from striking_power rating to heavy-damage-tail probability.

Why this exists
---------------
The old V3.1 mapping used a very steep rating scale (6.5), which was calibrated
against the old compressed striking_power trait. With the rebuilt trait, elite
fighters legitimately reach the high 70s / low 80s, making the old mapping put
nearly every landed strike into the heavy-damage tail.

This sweep keeps all other V3.3 physics fixed and screens wider tail-rating
scales on 200 mature 2020+ bouts. Tail magnitude scale remains 55.0, KD shock
remains 90, and strong KD collapse remains locked.

For R1 KD metrics, each path is also replayed with a one-round horizon using the
same seed. This avoids relying on the older damage-event bookkeeping, which does
not expose round attribution reliably.
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

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
TAIL_RATING_SCALES = (20.0, 30.0, 40.0, 50.0)
TAIL_MAGNITUDE_SCALE = 55.0
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


class TailScaleSweepV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    def __init__(self, *args, tail_rating_scale: float, **kwargs) -> None:
        self.tail_rating_scale = float(tail_rating_scale)
        super().__init__(*args, **kwargs)

    def _tail_probability(self, attacker: int) -> float:
        power = base._value(self.fighters[attacker], "striking_power")
        return damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / self.tail_rating_scale
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
            tail *= exp((power - 50.0) / TAIL_MAGNITUDE_SCALE)
            raw_damage += tail
        return max(0.0, raw_damage * damage.STRIKE_DAMAGE_SCALE)

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
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _tail_probability_for_rating(power: float, scale: float) -> float:
    return damage._sigmoid(
        damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
        + (float(power) - 50.0) / float(scale)
    )


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    print("\n" + "=" * 128)
    print("REBUILT FSR-32 POWER — MC HEAVY-TAIL RATING-SCALE SWEEP")
    print("=" * 128)
    print(f"bouts: {len(cohort):,}; paths/candidate: {len(cohort) * args.paths:,}")
    print(f"tail magnitude scale fixed: {TAIL_MAGNITUDE_SCALE:.1f}")
    print(f"KD shock fixed: {KD_SHOCK_COEFFICIENT:.0f}")
    print("KD collapse fixed: strong (5.0, 2.0)")
    print("\nTAIL PROBABILITY BY POWER RATING")
    print(f"{'scale':>7} {'p40':>8} {'p50':>8} {'p60':>8} {'p70':>8} {'p80':>8}")
    for scale in TAIL_RATING_SCALES:
        probs = [_tail_probability_for_rating(p, scale) for p in (40, 50, 60, 70, 80)]
        print(f"{scale:7.0f}" + "".join(f" {p:8.2%}" for p in probs))

    rows = []
    for scale in TAIL_RATING_SCALES:
        total_paths = len(cohort) * args.paths
        ko_count = 0
        r1_ko_count = 0
        ko_round_sum = 0.0
        kd_total = 0
        any_kd_count = 0
        r1_kd_total = 0
        any_r1_kd_count = 0

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
                    tail_rating_scale=scale,
                )

                sim = TailScaleSweepV33(red, blue, rounds=3, **kwargs)
                result = sim.run()
                path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
                kd_total += path_kd
                any_kd_count += int(path_kd > 0)

                if result.finish is not None:
                    ko_count += 1
                    finish_round = int(result.finish.round)
                    ko_round_sum += finish_round
                    r1_ko_count += int(finish_round == 1)

                # Same seed, one-round replay: exact R1 state/path up to the end
                # of round 1, without depending on event round labels.
                sim_r1 = TailScaleSweepV33(red, blue, rounds=1, **kwargs)
                sim_r1.run()
                path_r1_kd = (
                    int(sim_r1.stats[0].knockdowns_scored)
                    + int(sim_r1.stats[1].knockdowns_scored)
                )
                r1_kd_total += path_r1_kd
                any_r1_kd_count += int(path_r1_kd > 0)

            completed += args.paths
            if completed % 500 == 0 or bout_idx + 1 == len(cohort):
                print(f"[scale={scale:.0f}] paths {completed:,}/{total_paths:,}", flush=True)

        any_ko = ko_count / total_paths
        r1_ko = r1_ko_count / total_paths
        mean_round = ko_round_sum / ko_count if ko_count else float("nan")
        mean_kd = kd_total / total_paths
        any_kd = any_kd_count / total_paths
        mean_r1_kd = r1_kd_total / total_paths
        any_r1_kd = any_r1_kd_count / total_paths
        rows.append((scale, any_ko, r1_ko, mean_round, mean_r1_kd, any_r1_kd, mean_kd, any_kd))

    print("\nHISTORICAL TARGETS")
    print(
        f"any KO={HISTORICAL_ANY_KO_RATE:.2%}; R1 KO={HISTORICAL_R1_KO_RATE:.2%}; "
        f"mean KO round={HISTORICAL_MEAN_KO_ROUND:.3f}; mean R1 KD={HISTORICAL_MEAN_R1_KD:.4f}; "
        f"any R1 KD={HISTORICAL_ANY_R1_KD_RATE:.2%}; mean KD={HISTORICAL_MEAN_KD:.4f}; "
        f"any KD={HISTORICAL_ANY_KD_RATE:.2%}"
    )

    print("\nRESULTS")
    print(
        f"{'scale':>7} {'any_KO':>9} {'R1_KO':>9} {'meanRnd':>9} "
        f"{'meanR1KD':>10} {'anyR1KD':>10} {'meanKD':>9} {'anyKD':>9}"
    )
    for row in rows:
        scale, any_ko, r1_ko, mean_round, mean_r1_kd, any_r1_kd, mean_kd, any_kd = row
        print(
            f"{scale:7.0f} {any_ko:9.2%} {r1_ko:9.2%} {mean_round:9.3f} "
            f"{mean_r1_kd:10.4f} {any_r1_kd:10.2%} {mean_kd:9.4f} {any_kd:9.2%}"
        )

    print("\nResearch only: no simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
