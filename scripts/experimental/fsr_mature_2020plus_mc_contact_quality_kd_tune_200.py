"""Research-only KD tuning screen with contact-quality strike severity active.

Agreed strike architecture held fixed:
- contact quality: mean-1 lognormal, sigma=0.80
- striking_power changes damage magnitude only
- power magnitude scale=75
- action/output/landing generation unchanged by striking_power
- V3.3 stamina/recovery and action-first fatigue timing unchanged
- strong KD collapse remains fixed during this KD-only tuning screen

This sweep changes ONLY two KD equation terms:
- acute shock coefficient
- accumulated depletion coefficient

The goal is to keep fresh/R1 KD generation near the established historical
reference while reducing excessive later-fight KD accumulation.

The CSV is rewritten after every completed candidate so Codespaces restarts do
not erase completed calibration rows.
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
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental.fsr_mature_2020plus_mc_contact_quality_power_screen import (
    Candidate as ContactCandidate,
    ContactQualityPowerV33,
)

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
OUTPUT_PATH = Path("data/experimental/contact_quality_kd_tune_200.csv")

CONTACT_SIGMA = 0.80
POWER_MAGNITUDE_SCALE = 75.0
CONTACT = ContactCandidate(sigma=CONTACT_SIGMA, power_scale=POWER_MAGNITUDE_SCALE)
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

# Established mature-2020+ historical KD references. These remain the ranking
# targets for this first 200-bout screen; final locks require full-cohort audit.
HIST_MEAN_R1_KD = 0.2281
HIST_ANY_R1_KD = 0.2000
HIST_MEAN_KD = 0.4364
HIST_ANY_KD = 0.3578
HIST_ANY_KO = 0.3144
HIST_R1_KO = 0.1406
HIST_MEAN_KO_ROUND = 1.835


@dataclass(frozen=True)
class KDCandidate:
    shock: float
    depletion: float

    @property
    def name(self) -> str:
        return f"shock{self.shock:.0f}_dep{self.depletion:.2f}"


# Broad enough to locate the basin without spending a full-cohort run.
CANDIDATES = tuple(
    KDCandidate(shock=shock, depletion=depletion)
    for shock in (85.0, 90.0, 95.0, 100.0)
    for depletion in (0.00, 0.25, 0.50, 0.75, 1.00)
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


class ContactQualityKDTuneV33(ContactQualityPowerV33):
    def __init__(self, *args, kd_candidate: KDCandidate, **kwargs) -> None:
        self.kd_candidate = kd_candidate
        super().__init__(*args, **kwargs)

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        depletion = 1.0 - state.reservoir_fraction
        logit_p = (
            damage.KD_BASE_LOGIT
            + self.kd_candidate.shock * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + self.kd_candidate.depletion * depletion
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _score(row: dict[str, float]) -> float:
    # Equal relative-error weight across the four KD targets. KO metrics are
    # reported diagnostically but intentionally excluded from KD calibration.
    pieces = (
        (row["mean_r1_kd"] - HIST_MEAN_R1_KD) / HIST_MEAN_R1_KD,
        (row["any_r1_kd"] - HIST_ANY_R1_KD) / HIST_ANY_R1_KD,
        (row["mean_kd"] - HIST_MEAN_KD) / HIST_MEAN_KD,
        (row["any_kd"] - HIST_ANY_KD) / HIST_ANY_KD,
    )
    return float(sum(x * x for x in pieces))


def _persist(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["kd_score", "shock", "depletion"], kind="stable")
    frame.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), args.paths),
        dtype=np.int64,
    )

    rows: list[dict[str, float | str]] = []
    print("\n" + "=" * 132)
    print("CONTACT-QUALITY KD TUNING — 200-BOUT SCREEN")
    print("=" * 132)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; paths/candidate: {total_paths:,}; seed: {args.seed}")
    print(f"contact sigma: {CONTACT_SIGMA:.2f}; power magnitude scale: {POWER_MAGNITUDE_SCALE:.0f}")
    print("power affects damage MAGNITUDE ONLY; output/landing frequency unchanged")
    print("KD collapse fixed: strong (5.0,2.0); KD base logit/resistance/recent-KD terms fixed")
    print(f"CSV: {args.output}")

    for candidate in CANDIDATES:
        kd_total = 0
        any_kd_count = 0
        r1_kd_total = 0
        any_r1_kd_count = 0
        ko_count = 0
        r1_ko_count = 0
        ko_round_sum = 0.0
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
                    candidate=CONTACT,
                    kd_candidate=candidate,
                )

                sim = ContactQualityKDTuneV33(red, blue, rounds=3, **kwargs)
                result = sim.run()
                path_kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
                kd_total += path_kd
                any_kd_count += int(path_kd > 0)

                if result.finish is not None:
                    ko_count += 1
                    finish_round = int(getattr(result.finish, "round", 0) or 0)
                    ko_round_sum += finish_round
                    r1_ko_count += int(finish_round == 1)

                # Exact R1 replay with same seed and same candidate.
                sim_r1 = ContactQualityKDTuneV33(red, blue, rounds=1, **kwargs)
                sim_r1.run()
                path_r1_kd = int(sim_r1.stats[0].knockdowns_scored) + int(sim_r1.stats[1].knockdowns_scored)
                r1_kd_total += path_r1_kd
                any_r1_kd_count += int(path_r1_kd > 0)

            completed += args.paths
            if completed % 1000 == 0 or bout_idx + 1 == len(cohort):
                print(
                    f"[{candidate.name}] paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}",
                    flush=True,
                )

        row: dict[str, float | str] = {
            "candidate": candidate.name,
            "shock": candidate.shock,
            "depletion": candidate.depletion,
            "contact_sigma": CONTACT_SIGMA,
            "power_magnitude_scale": POWER_MAGNITUDE_SCALE,
            "bouts": len(cohort),
            "paths_per_bout": args.paths,
            "total_paths": total_paths,
            "mean_r1_kd": r1_kd_total / total_paths,
            "any_r1_kd": any_r1_kd_count / total_paths,
            "mean_kd": kd_total / total_paths,
            "any_kd": any_kd_count / total_paths,
            "any_ko": ko_count / total_paths,
            "r1_ko": r1_ko_count / total_paths,
            "mean_ko_round": ko_round_sum / ko_count if ko_count else float("nan"),
        }
        row["kd_score"] = _score(row)  # type: ignore[arg-type]
        rows.append(row)
        _persist(rows, args.output)
        print(
            f"  -> R1KD {row['mean_r1_kd']:.4f} / {row['any_r1_kd']:.2%}; "
            f"totalKD {row['mean_kd']:.4f} / {row['any_kd']:.2%}; "
            f"score {row['kd_score']:.5f}",
            flush=True,
        )

    ranked = pd.DataFrame(rows).sort_values("kd_score", kind="stable").reset_index(drop=True)
    print("\nHISTORICAL KD REFERENCES")
    print(
        f"mean R1 KD={HIST_MEAN_R1_KD:.4f}; any R1 KD={HIST_ANY_R1_KD:.2%}; "
        f"mean KD={HIST_MEAN_KD:.4f}; any KD={HIST_ANY_KD:.2%}"
    )
    print("\nRANKED RESULTS")
    print(
        f"{'candidate':>17} {'R1KD':>8} {'R1any':>8} {'KDmean':>8} {'KDany':>8} "
        f"{'KO':>8} {'R1KO':>8} {'meanRnd':>8} {'score':>10}"
    )
    for _, row in ranked.iterrows():
        print(
            f"{row['candidate']:>17} {row['mean_r1_kd']:8.4f} {row['any_r1_kd']:8.2%} "
            f"{row['mean_kd']:8.4f} {row['any_kd']:8.2%} {row['any_ko']:8.2%} "
            f"{row['r1_ko']:8.2%} {row['mean_ko_round']:8.3f} {row['kd_score']:10.5f}"
        )

    print(f"\nSaved CSV: {args.output}")
    print("Research only: no simulator, FSR artifact, or collapse constant modified.")


if __name__ == "__main__":
    main()
