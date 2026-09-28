"""Research-only contact-quality x striking-power magnitude screen.

Purpose
-------
Prototype the agreed strike-severity architecture without modifying the shadow
simulator or FSR artifact:

    action/output -> strike lands -> random contact quality
    -> effective striking power scales magnitude -> damage -> KD/KO physics

Key contract
------------
- striking_power DOES NOT change strike-attempt frequency;
- striking_power DOES NOT change the probability of receiving a special tail;
- every landed significant strike draws one continuous contact-quality value;
- contact quality is global fighter-independent physics;
- effective (fatigued) striking_power scales damage magnitude only;
- action-first fatigue timing is inherited from V3.3 unchanged.

Contact quality
---------------
Q is lognormal with E[Q] = 1 exactly:

    log(Q) ~ Normal(-sigma^2 / 2, sigma)

Changing sigma therefore changes spread/right-tail behavior without changing
mean contact quality.  This is intentionally cleaner than the old binary
6%-tail event.

Damage anchor
-------------
The initial mean landed-strike damage at effective power=50 is 1.18 reservoir
units.  That matches the mean of the prior *neutral-power* severity mixture:

    0.50 * (base mean 2.0 + 0.06 * tail mean 6.0) = 1.18

This keeps the power-50 mean damage budget fixed while we test distribution
shape and power magnitude translation.

The screen is diagnostic, not a calibration lock.  Current KD shock/depletion,
strong collapse, reservoir, stamina, and recovery remain unchanged so we can
observe how the proposed damage architecture interacts with existing physics.
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
BASE_DAMAGE_AT_POWER_50 = 1.18
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
    sigma: float
    power_scale: float

    @property
    def name(self) -> str:
        return f"s{self.sigma:.2f}_p{self.power_scale:.0f}"


# Broad first-pass grid.  Sigma controls contact-quality spread/right tail.
# power_scale controls how strongly rating differences alter damage magnitude:
# multiplier = exp((effective_power - 50) / power_scale).
CANDIDATES = tuple(
    Candidate(sigma=sigma, power_scale=power_scale)
    for sigma in (0.60, 0.80, 1.00, 1.20)
    for power_scale in (55.0, 75.0, 100.0)
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def power_multiplier(power: float, scale: float) -> float:
    return float(exp((float(power) - 50.0) / float(scale)))


class ContactQualityPowerV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """V3.3 with continuous global contact quality and magnitude-only power."""

    def __init__(self, *args, candidate: Candidate, **kwargs) -> None:
        self.contact_candidate = candidate
        super().__init__(*args, **kwargs)

    def _draw_contact_quality(self) -> float:
        sigma = self.contact_candidate.sigma
        # mu=-sigma^2/2 makes arithmetic mean exactly 1.0.
        return float(self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))

    def _draw_strike_damage(self, attacker: int) -> float:
        # self.fighters contains the segment's rolling effective profile in the
        # V3.1+ inheritance chain, so stamina affects power only on subsequent
        # segments under the existing action-first contract.
        effective_power = base._value(self.fighters[attacker], "striking_power")
        q = self._draw_contact_quality()
        magnitude = power_multiplier(effective_power, self.contact_candidate.power_scale)
        return max(0.0, BASE_DAMAGE_AT_POWER_50 * q * magnitude)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        # Hold current KD physics fixed for this architecture screen.
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


def _distribution_summary(candidate: Candidate, *, seed: int, draws: int = 200_000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sigma = candidate.sigma
    q = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=draws)
    return {
        "q50": float(np.quantile(q, 0.50)),
        "q90": float(np.quantile(q, 0.90)),
        "q95": float(np.quantile(q, 0.95)),
        "q99": float(np.quantile(q, 0.99)),
        "q999": float(np.quantile(q, 0.999)),
    }


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

    print("\n" + "=" * 150)
    print("CONTACT QUALITY x STRIKING POWER MAGNITUDE — 200-BOUT ARCHITECTURE SCREEN")
    print("=" * 150)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; seed: {args.seed}")
    print(f"damage anchor at power 50: {BASE_DAMAGE_AT_POWER_50:.3f} reservoir units / landed sig strike")
    print(f"KD shock: {KD_SHOCK_COEFFICIENT:.0f}; depletion: {damage.KD_DEPLETION_COEFFICIENT:.2f}; collapse: strong (5.0,2.0)")
    print("power affects MAGNITUDE ONLY; contact-quality distribution is identical for every fighter")

    print("\nHISTORICAL TARGETS")
    print(
        f"any KO={HISTORICAL_ANY_KO_RATE:.2%}; R1 KO={HISTORICAL_R1_KO_RATE:.2%}; "
        f"mean KO round={HISTORICAL_MEAN_KO_ROUND:.3f}; mean R1 KD={HISTORICAL_MEAN_R1_KD:.4f}; "
        f"any R1 KD={HISTORICAL_ANY_R1_KD_RATE:.2%}; mean KD={HISTORICAL_MEAN_KD:.4f}; "
        f"any KD={HISTORICAL_ANY_KD_RATE:.2%}"
    )

    results: list[dict[str, float | str]] = []
    for candidate_idx, candidate in enumerate(CANDIDATES):
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
                sim = ContactQualityPowerV33(red, blue, rounds=3, **kwargs)
                result = sim.run()
                path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
                kd_total += path_kd
                paths_with_kd += int(path_kd > 0)

                if result.finish is not None:
                    ko_count += 1
                    finish_round = int(getattr(result.finish, "round", 0) or 0)
                    ko_round_sum += finish_round
                    r1_ko_count += int(finish_round == 1)

                # Exact R1 prefix replay using the same path seed.
                sim_r1 = ContactQualityPowerV33(red, blue, rounds=1, **kwargs)
                sim_r1.run()
                path_r1_kd = int(sim_r1.stats[0].knockdowns_scored) + int(sim_r1.stats[1].knockdowns_scored)
                r1_kd_total += path_r1_kd
                paths_with_r1_kd += int(path_r1_kd > 0)

            completed += args.paths
            if completed % 1000 == 0 or bout_idx + 1 == len(cohort):
                print(
                    f"[{candidate.name}] paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}",
                    flush=True,
                )

        dist = _distribution_summary(candidate, seed=args.seed + candidate_idx + 1000)
        results.append(
            {
                "name": candidate.name,
                "sigma": candidate.sigma,
                "p_scale": candidate.power_scale,
                "q50": dist["q50"],
                "q95": dist["q95"],
                "q99": dist["q99"],
                "q999": dist["q999"],
                "m40": power_multiplier(40, candidate.power_scale),
                "m50": power_multiplier(50, candidate.power_scale),
                "m60": power_multiplier(60, candidate.power_scale),
                "m70": power_multiplier(70, candidate.power_scale),
                "m80": power_multiplier(80, candidate.power_scale),
                "any_ko": ko_count / total_paths,
                "r1_ko": r1_ko_count / total_paths,
                "mean_round": ko_round_sum / ko_count if ko_count else float("nan"),
                "mean_r1_kd": r1_kd_total / total_paths,
                "any_r1_kd": paths_with_r1_kd / total_paths,
                "mean_kd": kd_total / total_paths,
                "any_kd": paths_with_kd / total_paths,
            }
        )

    print("\nCONTACT QUALITY + POWER CURVE")
    print(
        f"{'candidate':>12} {'Q50':>7} {'Q95':>7} {'Q99':>7} {'Q99.9':>7} "
        f"{'M40':>7} {'M50':>7} {'M60':>7} {'M70':>7} {'M80':>7}"
    )
    for row in results:
        print(
            f"{row['name']:>12} {row['q50']:7.3f} {row['q95']:7.3f} {row['q99']:7.3f} {row['q999']:7.3f} "
            f"{row['m40']:7.3f} {row['m50']:7.3f} {row['m60']:7.3f} {row['m70']:7.3f} {row['m80']:7.3f}"
        )

    print("\nMC OUTCOMES")
    print(
        f"{'candidate':>12} {'any_KO':>9} {'R1_KO':>9} {'meanRnd':>9} "
        f"{'meanR1KD':>10} {'anyR1KD':>10} {'meanKD':>9} {'anyKD':>9}"
    )
    for row in results:
        print(
            f"{row['name']:>12} {row['any_ko']:9.2%} {row['r1_ko']:9.2%} "
            f"{row['mean_round']:9.3f} {row['mean_r1_kd']:10.4f} "
            f"{row['any_r1_kd']:10.2%} {row['mean_kd']:9.4f} {row['any_kd']:9.2%}"
        )

    print("\nMEAN DAMAGE PER LANDED STRIKE BY FRESH POWER (before fatigue; analytic)")
    print(f"{'candidate':>12} {'P40':>8} {'P50':>8} {'P60':>8} {'P70':>8} {'P80':>8}")
    for row in results:
        print(
            f"{row['name']:>12} "
            f"{BASE_DAMAGE_AT_POWER_50 * row['m40']:8.3f} "
            f"{BASE_DAMAGE_AT_POWER_50 * row['m50']:8.3f} "
            f"{BASE_DAMAGE_AT_POWER_50 * row['m60']:8.3f} "
            f"{BASE_DAMAGE_AT_POWER_50 * row['m70']:8.3f} "
            f"{BASE_DAMAGE_AT_POWER_50 * row['m80']:8.3f}"
        )

    print("\nInterpretation guardrails:")
    print("  sigma changes contact-quality variability/right tail but NOT mean contact quality.")
    print("  power_scale changes damage magnitude ONLY; larger scale = weaker power separation.")
    print("  output/attempt/landing generation is unchanged by striking_power.")
    print("  current KD/KO constants are held fixed; this screen does not select final KD calibration.")
    print("Research only: no simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
