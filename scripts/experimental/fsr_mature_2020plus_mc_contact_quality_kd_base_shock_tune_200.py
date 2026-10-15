"""Research-only KD base-logit x acute-shock tuning screen.

Working strike model is fixed:
- contact quality sigma = 0.80, mean-1 lognormal
- power magnitude scale = 75
- striking_power changes damage magnitude only
- action/landing generation unchanged by power

KD depletion sensitivity is fixed at zero for this screen because the prior
shock x depletion sweep showed depletion consistently worsened later-fight KD
excess.  We now tune the *shape* of the KD curve by pairing a lower baseline KD
hazard with stronger acute-shock sensitivity.

200-bout x 10-path screen. CSV is rewritten after every completed candidate so
partial progress survives a Codespaces restart.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
CONTACT_SIGMA = 0.80
POWER_MAGNITUDE_SCALE = 75.0
BASE_DAMAGE_AT_POWER_50 = 1.18
KD_DEPLETION_COEFFICIENT = 0.0
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)
OUTPUT_PATH = Path("data/experimental/contact_quality_kd_base_shock_tune_200.csv")

HISTORICAL_MEAN_R1_KD = 0.2281
HISTORICAL_ANY_R1_KD_RATE = 0.2000
HISTORICAL_MEAN_KD = 0.4364
HISTORICAL_ANY_KD_RATE = 0.3578
HISTORICAL_ANY_KO_RATE = 0.3144
HISTORICAL_R1_KO_RATE = 0.1406
HISTORICAL_MEAN_KO_ROUND = 1.835


@dataclass(frozen=True)
class Candidate:
    base_logit: float
    shock: float

    @property
    def name(self) -> str:
        return f"base{self.base_logit:.2f}_shock{self.shock:.0f}"


# Center around the current baseline -8.6359 and the promising 90-95 shock
# region, extending to lower baselines paired with larger acute-shock slopes.
CANDIDATES = tuple(
    Candidate(base_logit=base_logit, shock=shock)
    for base_logit in (-8.65, -8.90, -9.15, -9.40)
    for shock in (90.0, 95.0, 100.0, 105.0)
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def power_multiplier(power: float) -> float:
    return float(exp((float(power) - 50.0) / POWER_MAGNITUDE_SCALE))


class ContactQualityKDShapeV33(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    def __init__(self, *args, candidate: Candidate, **kwargs) -> None:
        self.kd_candidate = candidate
        super().__init__(*args, **kwargs)

    def _draw_contact_quality(self) -> float:
        sigma = CONTACT_SIGMA
        return float(self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))

    def _draw_strike_damage(self, attacker: int) -> float:
        effective_power = base._value(self.fighters[attacker], "striking_power")
        q = self._draw_contact_quality()
        return max(0.0, BASE_DAMAGE_AT_POWER_50 * q * power_multiplier(effective_power))

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        logit_p = (
            self.kd_candidate.base_logit
            + self.kd_candidate.shock * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + KD_DEPLETION_COEFFICIENT * (1.0 - state.reservoir_fraction)
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))


def _score(row: dict[str, float]) -> float:
    # Relative squared error over the four KD targets only.
    terms = (
        (row["mean_r1_kd"] - HISTORICAL_MEAN_R1_KD) / HISTORICAL_MEAN_R1_KD,
        (row["any_r1_kd"] - HISTORICAL_ANY_R1_KD_RATE) / HISTORICAL_ANY_R1_KD_RATE,
        (row["mean_kd"] - HISTORICAL_MEAN_KD) / HISTORICAL_MEAN_KD,
        (row["any_kd"] - HISTORICAL_ANY_KD_RATE) / HISTORICAL_ANY_KD_RATE,
    )
    return float(sum(x * x for x in terms))


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
    total_paths = len(cohort) * args.paths
    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    print("\n" + "=" * 132)
    print("CONTACT-QUALITY KD CURVE TUNING — BASE LOGIT x SHOCK — 200-BOUT SCREEN")
    print("=" * 132)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; paths/candidate: {total_paths:,}; seed: {args.seed}")
    print(f"contact sigma: {CONTACT_SIGMA:.2f}; power magnitude scale: {POWER_MAGNITUDE_SCALE:.0f}; depletion: {KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"CSV: {OUTPUT_PATH}")

    rows: list[dict[str, float | str]] = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for candidate in CANDIDATES:
        ko_count = r1_ko_count = 0
        ko_round_sum = 0.0
        kd_total = paths_with_kd = 0
        r1_kd_total = paths_with_r1_kd = 0
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

                sim = ContactQualityKDShapeV33(red, blue, rounds=3, **kwargs)
                result = sim.run()
                path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
                kd_total += path_kd
                paths_with_kd += int(path_kd > 0)

                if result.finish is not None:
                    ko_count += 1
                    finish_round = int(getattr(result.finish, "round", 0) or 0)
                    ko_round_sum += finish_round
                    r1_ko_count += int(finish_round == 1)

                sim_r1 = ContactQualityKDShapeV33(red, blue, rounds=1, **kwargs)
                sim_r1.run()
                path_r1_kd = int(sim_r1.stats[0].knockdowns_scored) + int(sim_r1.stats[1].knockdowns_scored)
                r1_kd_total += path_r1_kd
                paths_with_r1_kd += int(path_r1_kd > 0)

            completed += args.paths
            if completed % 1000 == 0 or bout_idx + 1 == len(cohort):
                print(f"[{candidate.name}] paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}", flush=True)

        row: dict[str, float | str] = {
            "candidate": candidate.name,
            "base_logit": candidate.base_logit,
            "shock": candidate.shock,
            "depletion": KD_DEPLETION_COEFFICIENT,
            "mean_r1_kd": r1_kd_total / total_paths,
            "any_r1_kd": paths_with_r1_kd / total_paths,
            "mean_kd": kd_total / total_paths,
            "any_kd": paths_with_kd / total_paths,
            "any_ko": ko_count / total_paths,
            "r1_ko": r1_ko_count / total_paths,
            "mean_ko_round": ko_round_sum / ko_count if ko_count else float("nan"),
        }
        row["score"] = _score(row)  # type: ignore[arg-type]
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

        print(
            f"  -> R1KD {row['mean_r1_kd']:.4f} / {row['any_r1_kd']:.2%}; "
            f"totalKD {row['mean_kd']:.4f} / {row['any_kd']:.2%}; score {row['score']:.5f}",
            flush=True,
        )

    ranked = pd.DataFrame(rows).sort_values(["score", "candidate"]).reset_index(drop=True)
    ranked.to_csv(OUTPUT_PATH, index=False)

    print("\nHISTORICAL KD REFERENCES")
    print(
        f"mean R1 KD={HISTORICAL_MEAN_R1_KD:.4f}; any R1 KD={HISTORICAL_ANY_R1_KD_RATE:.2%}; "
        f"mean KD={HISTORICAL_MEAN_KD:.4f}; any KD={HISTORICAL_ANY_KD_RATE:.2%}"
    )
    print("\nRANKED RESULTS")
    print(
        f"{'candidate':>22} {'R1KD':>8} {'R1any':>8} {'KDmean':>8} {'KDany':>8} "
        f"{'KO':>8} {'R1KO':>8} {'meanRnd':>8} {'score':>10}"
    )
    for _, row in ranked.iterrows():
        print(
            f"{row['candidate']:>22} {row['mean_r1_kd']:8.4f} {row['any_r1_kd']:8.2%} "
            f"{row['mean_kd']:8.4f} {row['any_kd']:8.2%} {row['any_ko']:8.2%} "
            f"{row['r1_ko']:8.2%} {row['mean_ko_round']:8.3f} {row['score']:10.5f}"
        )

    print(f"\nSaved CSV: {OUTPUT_PATH}")
    print("Research only: no simulator, FSR artifact, or collapse constant modified.")


if __name__ == "__main__":
    main()
