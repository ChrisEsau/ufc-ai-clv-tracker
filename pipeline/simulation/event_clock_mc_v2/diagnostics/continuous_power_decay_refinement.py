"""Fast research-only refinement of men's continuous consequence-side power decay.

This does not change FSR, strike generation, event timing, frozen V1 source, or
fighter-specific stamina. It searches a small neighborhood around the successful
continuous screen using

    power_offset(t) = intercept - t / denominator

with t in elapsed fight seconds. Historical targets are the full men's audit
cohort; simulation is deterministically thinned for speed and uses common seeds
across arms.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel as FrozenShadow
from pipeline.simulation.event_clock_mc_v2.diagnostics import lethality_round_bucket as bucket
from pipeline.simulation.event_clock_mc_v2.diagnostics import shared_power_decay_grid as shared
from pipeline.simulation.event_clock_mc_v2.diagnostics.weight_class_audit import select_cohort
from pipeline.simulation.event_mc_v1.components.profiles import MatchupProfiles, Side

# Current successful screen plus nearby lower-fresh / steeper-late candidates.
ARMS = (
    ("current_37p5_d12", 37.5, 12.0),
    ("i35_d12p5", 35.0, 12.5),
    ("i35_d12", 35.0, 12.0),
    ("i35_d11p5", 35.0, 11.5),
    ("i32p5_d12", 32.5, 12.0),
    ("i32p5_d11p5", 32.5, 11.5),
)

_CURRENT = ARMS[0]


class RefinementPowerShadow(FrozenShadow):
    def _offset(self, state) -> float:
        _, intercept, denominator = _CURRENT
        t = float(state.fight_time_seconds)
        return float(intercept - t / denominator)

    def resolve_landed_strike(self, *, state, attacker, prior_defender_kds, rng):
        offset = self._offset(state)

        def shifted(side: Side):
            p = self.profiles.fighter(side)
            return replace(p, striking_power=float(p.striking_power) + offset)

        model = FrozenShadow(
            profiles=MatchupProfiles(red=shifted(Side.RED), blue=shifted(Side.BLUE)),
            calibration=self.calibration,
        )
        return model.resolve_landed_strike(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
            rng=rng,
        )


def _thin(cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or len(cohort) <= n:
        return cohort.reset_index(drop=True)
    idx = np.linspace(0, len(cohort) - 1, num=n, dtype=int)
    return cohort.iloc[np.unique(idx)].reset_index(drop=True)


def _simulate_arm(target_n: int, sim_n_per_division: int, paths: int, seed: int, arm) -> pd.DataFrame:
    global _CURRENT
    _CURRENT = arm
    name, intercept, denominator = arm
    frames = []
    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort, _ = select_cohort(division, target_n)
        cohort = _thin(cohort, sim_n_per_division)
        print(
            f"ARM {name} i={intercept:.1f} d={denominator:.1f} | "
            f"{division} | fights={len(cohort)} paths={paths}"
        )
        frames.append(
            bucket.simulate_round_rows(
                cohort,
                division,
                paths,
                seed + i * 100_000_000,
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--sim-n-per-division", type=int, default=15)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/continuous_power_decay_refinement"),
    )
    args = parser.parse_args()

    bucket.EventClockShadowKOKDModel = RefinementPowerShadow
    _, hist = shared._historical_men(args.target_n)

    summaries = []
    rankings = []
    for arm in ARMS:
        name, intercept, denominator = arm
        rows = _simulate_arm(args.target_n, args.sim_n_per_division, args.paths, args.seed, arm)
        summary = shared._aggregate(rows, "simulated", 0.0, 0.0, name)
        score = shared._score(summary, hist)
        summary["arm"] = name
        summary["intercept"] = intercept
        summary["denominator"] = denominator
        summary["score"] = score
        summaries.append(summary)
        rankings.append({
            "arm": name,
            "intercept": intercept,
            "denominator": denominator,
            "r1_mid_offset": intercept - 150.0 / denominator,
            "r2_mid_offset": intercept - 450.0 / denominator,
            "r3_mid_offset": intercept - 750.0 / denominator,
            "score": score,
        })
        print(f"SCORE {name}: {score:.6f}")

    result = pd.concat(summaries, ignore_index=True)
    ranking = pd.DataFrame(rankings).sort_values("score").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(args.out_dir / "historical_men_targets.csv", index=False)
    result.to_csv(args.out_dir / "refinement_round_metrics.csv", index=False)
    ranking.to_csv(args.out_dir / "refinement_ranking.csv", index=False)

    print("\nHISTORICAL MEN TARGETS")
    print(hist.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nREFINEMENT RANKING")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nROUND METRICS")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
