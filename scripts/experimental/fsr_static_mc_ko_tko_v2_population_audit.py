"""Population audit for shock-driven KO/TKO V2 candidate curves.

This script runs full static Monte Carlo paths with the locked Damage V1 / KD=80
mechanics and several deliberately provisional KO/TKO curves. It is a research
comparison tool, not a calibration lock.

The audit is designed to catch the exact failure mode seen in rejected KO V1:
ordinary landed strikes accumulating into too many high-reservoir, non-KD
finishes. For each candidate it reports overall finish rate plus the shock,
reservoir, KD, and recent-KD state at the selected finishing strike.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_population_audit.parquet"
)
DEFAULT_MATCHUPS = 300
DEFAULT_PATHS_PER_MATCHUP = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809

# These are intentionally broad research candidates. They are NOT proposed
# locks. Candidate movement changes both the steepness of the acute-shock term
# and the amount of secondary accumulation/KD susceptibility.
CANDIDATES = [
    ko.KOParameters(
        name="A_soft_shock",
        base_logit=-12.0,
        shock_coefficient=32.0,
        shock_curvature=6.0,
        depletion_coefficient=3.5,
        current_kd_logit_bonus=2.0,
        recent_kd_logit_bonus=1.0,
    ),
    ko.KOParameters(
        name="B_balanced",
        base_logit=-12.0,
        shock_coefficient=40.0,
        shock_curvature=8.0,
        depletion_coefficient=3.0,
        current_kd_logit_bonus=2.25,
        recent_kd_logit_bonus=1.0,
    ),
    ko.KOParameters(
        name="C_shock_heavy",
        base_logit=-12.5,
        shock_coefficient=48.0,
        shock_curvature=10.0,
        depletion_coefficient=2.5,
        current_kd_logit_bonus=2.5,
        recent_kd_logit_bonus=1.0,
    ),
    ko.KOParameters(
        name="D_accumulation",
        base_logit=-12.5,
        shock_coefficient=38.0,
        shock_curvature=8.0,
        depletion_coefficient=4.5,
        current_kd_logit_bonus=2.25,
        recent_kd_logit_bonus=1.25,
    ),
    ko.KOParameters(
        name="E_kd_followup",
        base_logit=-12.5,
        shock_coefficient=42.0,
        shock_curvature=8.0,
        depletion_coefficient=3.0,
        current_kd_logit_bonus=3.0,
        recent_kd_logit_bonus=1.75,
    ),
]


def _choose_matchups(
    profiles: pd.DataFrame,
    matchup_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _ in range(matchup_count):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def _path_schedule(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    rng = np.random.default_rng(seed)
    matchups = _choose_matchups(profiles, matchup_count, rng)
    schedule: list[tuple[int, int, int, int]] = []
    for matchup_index, (red_i, blue_i) in enumerate(matchups, start=1):
        for path_index in range(paths_per_matchup):
            path_seed = int(rng.integers(0, 2**31 - 1))
            schedule.append((matchup_index, path_index, red_i, blue_i, path_seed))
    return schedule


def _run_candidate(
    profiles: pd.DataFrame,
    schedule: list[tuple[int, int, int, int, int]],
    params: ko.KOParameters,
    rounds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(schedule)

    for counter, (matchup_index, path_index, red_i, blue_i, path_seed) in enumerate(schedule, start=1):
        red = profiles.iloc[red_i]
        blue = profiles.iloc[blue_i]
        sim = ko.StaticFSRMCKOTKOV2(
            red,
            blue,
            ko_params=params,
            rounds=rounds,
            seed=path_seed,
        )
        path = sim.run()
        finish = path.finish

        row: dict[str, Any] = {
            "candidate": params.name,
            "matchup_index": matchup_index,
            "path_index": path_index,
            "path_seed": path_seed,
            "red_id": str(red["fighter_id"]),
            "blue_id": str(blue["fighter_id"]),
            "finish": int(finish is not None),
            "rounds": rounds,
            "red_sig_landed": sim.stats[0].sig_landed,
            "blue_sig_landed": sim.stats[1].sig_landed,
            "red_kd_scored": sim.stats[0].knockdowns_scored,
            "blue_kd_scored": sim.stats[1].knockdowns_scored,
        }

        if finish is not None:
            row.update(
                {
                    "winner": finish.winner,
                    "loser": finish.loser,
                    "finish_round": finish.round,
                    "finish_segment": finish.segment,
                    "finish_probability": finish.probability,
                    "finish_strike_damage": finish.strike_damage,
                    "finish_shock": finish.shock_fraction,
                    "loser_reservoir_before": finish.reservoir_fraction_before,
                    "loser_reservoir_after": finish.reservoir_fraction_after,
                    "finish_on_kd": int(finish.knockdown_on_strike),
                    "recent_kd_before_finish": int(finish.recent_kd_before),
                }
            )
        else:
            row.update(
                {
                    "winner": np.nan,
                    "loser": np.nan,
                    "finish_round": np.nan,
                    "finish_segment": np.nan,
                    "finish_probability": np.nan,
                    "finish_strike_damage": np.nan,
                    "finish_shock": np.nan,
                    "loser_reservoir_before": np.nan,
                    "loser_reservoir_after": np.nan,
                    "finish_on_kd": 0,
                    "recent_kd_before_finish": 0,
                }
            )
        rows.append(row)

        if counter % 1000 == 0 or counter == total:
            print(
                f"[KO V2 {params.name}] paths {counter:,}/{total:,}; "
                f"finishes={sum(r['finish'] for r in rows):,}",
                flush=True,
            )

    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, g in frame.groupby("candidate", sort=False):
        finishes = g[g["finish"] == 1]
        row: dict[str, Any] = {
            "candidate": candidate,
            "paths": len(g),
            "ko_tko_finish_rate": g["finish"].mean(),
            "mean_finish_round": finishes["finish_round"].mean(),
            "median_finish_shock": finishes["finish_shock"].median(),
            "p90_finish_shock": finishes["finish_shock"].quantile(0.90),
            "median_loser_reservoir_after": finishes["loser_reservoir_after"].median(),
            "finish_share_reservoir_gt_50pct": (
                (finishes["loser_reservoir_after"] > 0.50).mean() if len(finishes) else np.nan
            ),
            "finish_share_reservoir_gt_25pct": (
                (finishes["loser_reservoir_after"] > 0.25).mean() if len(finishes) else np.nan
            ),
            "finish_share_on_current_kd": (
                finishes["finish_on_kd"].mean() if len(finishes) else np.nan
            ),
            "finish_share_recent_kd": (
                finishes["recent_kd_before_finish"].mean() if len(finishes) else np.nan
            ),
            "finish_share_any_kd_context": (
                (
                    (finishes["finish_on_kd"] == 1)
                    | (finishes["recent_kd_before_finish"] == 1)
                ).mean()
                if len(finishes)
                else np.nan
            ),
            "finish_share_shock_ge_6pct": (
                (finishes["finish_shock"] >= 0.06).mean() if len(finishes) else np.nan
            ),
            "finish_share_shock_ge_8pct": (
                (finishes["finish_shock"] >= 0.08).mean() if len(finishes) else np.nan
            ),
            "finish_share_shock_ge_10pct": (
                (finishes["finish_shock"] >= 0.10).mean() if len(finishes) else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit shock-driven KO/TKO V2 full paths")
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=damage.FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[KO V2 audit] loading profiles from {args.fsr_path}", flush=True)
    profiles = damage.load_profiles(args.fsr_path)
    print(f"[KO V2 audit] latest fighter profiles: {len(profiles):,}", flush=True)

    schedule = _path_schedule(
        profiles,
        args.matchups,
        args.paths_per_matchup,
        args.seed,
    )
    print(
        f"[KO V2 audit] shared schedule: {len(schedule):,} paths/candidate; "
        f"rounds={args.rounds}; KD shock coefficient={damage.KD_SHOCK_COEFFICIENT:g}",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    for params in CANDIDATES:
        print("\n" + "=" * 120)
        print(f"RUNNING KO V2 CANDIDATE: {params.name}")
        print(params)
        print("=" * 120)
        frames.append(_run_candidate(profiles, schedule, params, args.rounds))

    combined = pd.concat(frames, ignore_index=True)
    summary = _summary(combined)

    print("\n" + "=" * 160)
    print("SHOCK-DRIVEN KO/TKO V2 — FULL-PATH POPULATION SUMMARY")
    print("=" * 160)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nRESEARCH BOUNDARY")
    print("- No KO candidate is locked by this script.")
    print("- Reject curves that recreate many high-reservoir/non-KD ordinary-strike finishes.")
    print("- Compare aggregate finish rate, finish timing, shock concentration, accumulation, and KD context together.")
    print("- KD=80 remains locked independently of this KO sweep.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"\n[KO V2 audit] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
