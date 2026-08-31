"""Fast research-only screen for continuous consequence-side power decay.

Compares three men's Event Clock arms while leaving FSR, strike budgets, event
scheduling, wrestling, submissions, judging, and frozen V1 source untouched:

1. baseline: current shadow consequence model
2. step_ref: empirically successful +25 / 0 / -25 rating-point offsets in R1/R2/R3
3. continuous: elapsed-time linear offset whose time-average in R1/R2/R3 is
   approximately +25 / 0 / -25:

       offset(t) = 37.5 - t / 12

   capped to [-37.5, +37.5] over the first 15 minutes.

This is a screening experiment, not a proposed production mechanic. Historical
targets use the full men's audit cohort. Simulation is deterministically thinned
within each division for speed, with common seeds across arms.
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

_MODE = "baseline"


class ContinuousPowerShadow(FrozenShadow):
    """Frozen KO/KD mechanics with a research-only common power offset."""

    def _offset(self, state) -> float:
        t = float(state.fight_time_seconds)
        if _MODE == "baseline":
            return 0.0
        if _MODE == "step_ref":
            round_index = min(max(int(t // 300.0), 0), 2)
            return 25.0 - 25.0 * float(round_index)
        if _MODE == "continuous":
            return float(np.clip(37.5 - t / 12.0, -37.5, 37.5))
        raise RuntimeError(f"Unknown research arm: {_MODE}")

    def resolve_landed_strike(self, *, state, attacker, prior_defender_kds, rng):
        offset = self._offset(state)

        def shifted(side: Side):
            profile = self.profiles.fighter(side)
            return replace(profile, striking_power=float(profile.striking_power) + offset)

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


def _simulate_arm(target_n: int, sim_n_per_division: int, paths: int, seed: int, mode: str) -> pd.DataFrame:
    global _MODE
    _MODE = mode
    frames = []
    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort, _ = select_cohort(division, target_n)
        cohort = _thin(cohort, sim_n_per_division)
        print(f"ARM {mode} | {division} | fights={len(cohort)} paths={paths}")
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
    parser.add_argument("--sim-n-per-division", type=int, default=20)
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/continuous_power_decay_screen"),
    )
    args = parser.parse_args()

    bucket.EventClockShadowKOKDModel = ContinuousPowerShadow
    _, hist_summary = shared._historical_men(args.target_n)

    summaries = []
    rankings = []
    for mode in ("baseline", "step_ref", "continuous"):
        rows = _simulate_arm(
            args.target_n,
            args.sim_n_per_division,
            args.paths,
            args.seed,
            mode,
        )
        summary = shared._aggregate(rows, "simulated", 0.0, 0.0, mode)
        score = shared._score(summary, hist_summary)
        summary["arm"] = mode
        summary["score"] = score
        summaries.append(summary)
        rankings.append({"arm": mode, "score": score})
        print(f"SCORE {mode}: {score:.6f}")

    result = pd.concat(summaries, ignore_index=True)
    ranking = pd.DataFrame(rankings).sort_values("score").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist_summary.to_csv(args.out_dir / "historical_men_targets.csv", index=False)
    result.to_csv(args.out_dir / "screen_round_metrics.csv", index=False)
    ranking.to_csv(args.out_dir / "screen_ranking.csv", index=False)

    print("\nHISTORICAL MEN TARGETS")
    print(hist_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nSCREEN RANKING")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nROUND METRICS")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
