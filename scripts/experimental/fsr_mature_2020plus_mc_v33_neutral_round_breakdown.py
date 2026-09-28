"""Round-by-round diagnostic for the full-cohort neutral-power V3.3 baseline.

Research only. No simulator or FSR artifact is modified.

Uses the exact current neutral-power configuration:
- fighter-specific striking_power disabled in strike severity
- fixed 6% heavy-tail probability
- flat tail magnitude
- KD shock coefficient 90
- strong collapse (5.0, 2.0)
- V3.3 global recovery
- aligned mature 2020+ cohort

For each path seed we run 1-, 2-, and 3-round prefixes. Cumulative KD counts
from identical seeds are differenced to recover exact per-round KD counts without
relying on damage-event metadata. KO finish round is read from the 3-round run.
The 2- and 3-round runs also expose the actual reservoir state immediately
before and after between-round recovery through a small audit-only hook.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_v33_neutral_power_full_cohort as neutral

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810


class NeutralRoundAudit(neutral.NeutralPowerV33):
    """Neutral baseline with audit-only snapshots around round recovery."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.round_recovery_snapshots: list[dict[str, float | int]] = []

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        before = [float(s.reservoir_fraction) for s in self.damage_state]
        stamina_before = [float(s.fraction) for s in self.stamina_state]
        super()._apply_between_round_recovery(completed_round)
        after = [float(s.reservoir_fraction) for s in self.damage_state]
        stamina_after = [float(s.fraction) for s in self.stamina_state]
        for fighter in (0, 1):
            self.round_recovery_snapshots.append(
                {
                    "after_round": int(completed_round),
                    "fighter": int(fighter),
                    "reservoir_before_recovery": before[fighter],
                    "reservoir_after_recovery": after[fighter],
                    "stamina_before_recovery": stamina_before[fighter],
                    "stamina_after_recovery": stamina_after[fighter],
                }
            )


def _path_kds(sim: NeutralRoundAudit) -> int:
    return int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)


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

    kd_count = defaultdict(int)
    paths_with_kd = defaultdict(int)
    ko_count = defaultdict(int)
    reached_round = defaultdict(int)
    reservoir_enter = defaultdict(list)
    reservoir_end_before_recovery = defaultdict(list)
    reservoir_after_recovery = defaultdict(list)
    stamina_enter = defaultdict(list)
    stamina_end_before_recovery = defaultdict(list)
    stamina_after_recovery = defaultdict(list)

    completed = 0
    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for seed in seed_matrix[bout_idx]:
            kwargs = dict(
                collapse=neutral.STRONG_COLLAPSE,
                seed=int(seed),
                red_age=r_age,
                blue_age=b_age,
            )

            sims = []
            results = []
            cumulative_kd = []
            for rounds in (1, 2, 3):
                sim = NeutralRoundAudit(red, blue, rounds=rounds, **kwargs)
                result = sim.run()
                sims.append(sim)
                results.append(result)
                cumulative_kd.append(_path_kds(sim))

            per_round_kd = (
                cumulative_kd[0],
                max(0, cumulative_kd[1] - cumulative_kd[0]),
                max(0, cumulative_kd[2] - cumulative_kd[1]),
            )
            for rnd, count in enumerate(per_round_kd, start=1):
                kd_count[rnd] += count
                paths_with_kd[rnd] += int(count > 0)

            finish = results[2].finish
            finish_round = int(getattr(finish, "round", 0) or 0) if finish is not None else 0
            if finish_round in (1, 2, 3):
                ko_count[finish_round] += 1

            # Every path reaches R1. A path reaches later rounds only if the
            # previous round did not finish in the 3-round run.
            reached_round[1] += 1
            if finish_round == 0 or finish_round >= 2:
                reached_round[2] += 1
            if finish_round == 0 or finish_round >= 3:
                reached_round[3] += 1

            # Fresh state entering R1 is always full by construction.
            reservoir_enter[1].extend([1.0, 1.0])
            stamina_enter[1].extend([1.0, 1.0])

            # Use the 3-round run's actual recovery hook for states after R1/R2.
            snaps = sims[2].round_recovery_snapshots
            by_round = defaultdict(list)
            for snap in snaps:
                by_round[int(snap["after_round"])].append(snap)

            for completed_round in (1, 2):
                round_snaps = by_round.get(completed_round, [])
                if len(round_snaps) == 2:
                    for snap in round_snaps:
                        reservoir_end_before_recovery[completed_round].append(
                            float(snap["reservoir_before_recovery"])
                        )
                        reservoir_after_recovery[completed_round].append(
                            float(snap["reservoir_after_recovery"])
                        )
                        stamina_end_before_recovery[completed_round].append(
                            float(snap["stamina_before_recovery"])
                        )
                        stamina_after_recovery[completed_round].append(
                            float(snap["stamina_after_recovery"])
                        )
                    next_round = completed_round + 1
                    reservoir_enter[next_round].extend(
                        float(s["reservoir_after_recovery"]) for s in round_snaps
                    )
                    stamina_enter[next_round].extend(
                        float(s["stamina_after_recovery"]) for s in round_snaps
                    )

        completed += args.paths
        if completed % 1000 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}; bouts {bout_idx + 1:,}/{len(cohort):,}", flush=True)

    print("\n" + "=" * 126)
    print("V3.3 NEUTRAL-POWER FULL-COHORT — ROUND-BY-ROUND DIAGNOSTIC")
    print("=" * 126)
    print(f"bouts: {len(cohort):,}; paths/bout: {args.paths}; total paths: {total_paths:,}; seed: {args.seed}")
    print("power severity: neutral; base tail=6%; shock=90; collapse=strong (5.0,2.0)")

    print("\nROUND OUTCOMES")
    print(f"{'round':>5} {'KO/all':>10} {'KO|reached':>12} {'KD/all':>10} {'anyKD/all':>12} {'reached':>10}")
    for rnd in (1, 2, 3):
        reached = reached_round[rnd]
        print(
            f"{rnd:5d} "
            f"{ko_count[rnd] / total_paths:10.2%} "
            f"{(ko_count[rnd] / reached if reached else 0.0):12.2%} "
            f"{kd_count[rnd] / total_paths:10.4f} "
            f"{paths_with_kd[rnd] / total_paths:12.2%} "
            f"{reached:10,d}"
        )

    print("\nSTATE BY ROUND")
    print(f"{'round':>5} {'mean reservoir entering':>24} {'mean stamina entering':>22}")
    for rnd in (1, 2, 3):
        rvals = reservoir_enter[rnd]
        svals = stamina_enter[rnd]
        rmean = float(np.mean(rvals)) if rvals else float('nan')
        smean = float(np.mean(svals)) if svals else float('nan')
        print(f"{rnd:5d} {rmean:24.4f} {smean:22.4f}")

    print("\nBETWEEN-ROUND RECOVERY")
    print(
        f"{'after':>5} {'reservoir pre':>15} {'reservoir post':>16} "
        f"{'stamina pre':>13} {'stamina post':>14}"
    )
    for rnd in (1, 2):
        rb = reservoir_end_before_recovery[rnd]
        ra = reservoir_after_recovery[rnd]
        sb = stamina_end_before_recovery[rnd]
        sa = stamina_after_recovery[rnd]
        print(
            f"{rnd:5d} "
            f"{(float(np.mean(rb)) if rb else float('nan')):15.4f} "
            f"{(float(np.mean(ra)) if ra else float('nan')):16.4f} "
            f"{(float(np.mean(sb)) if sb else float('nan')):13.4f} "
            f"{(float(np.mean(sa)) if sa else float('nan')):14.4f}"
        )

    print("\nHistorical references currently established for this exact cohort:")
    print("  R1 KO/TKO: 14.06%")
    print("  mean R1 KD/fight: 0.2281")
    print("  any R1 KD: 20.00%")
    print("  total KO/TKO: 31.44%")
    print("  total mean KD/fight: 0.4364")
    print("  any KD: 35.78%")
    print("Research only: no simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
