"""Audit R1 phase occupancy and strike attempts/lands for one historical bout.

Research-only diagnostic. Reuses the exact current locked shadow KO configuration
and parses the simulator event log; no simulator physics are modified.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from scripts.experimental import fsr_mature_2020plus_full_cohort_ko_validation_r3_d60_s0 as full
from scripts.experimental import run_single_historical_ko_bout as single
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
PHASES = ("DISTANCE", "CLINCH", "GROUND")


def _parse_striking(text: str, name: str) -> tuple[int, int]:
    pattern = re.compile(re.escape(name) + r"\s+(\d+)/(\d+)\s+sig")
    match = pattern.search(str(text))
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    p = argparse.ArgumentParser(description="Audit R1 phase occupancy and striking for one historical bout")
    p.add_argument("--bout-id", required=True)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    full._configure_locked_candidate()
    bout, pair = single._select_bout(str(args.bout_id))
    red, blue = pair
    red_age = float(bout["r_age"]) if pd.notna(bout.get("r_age")) else None
    blue_age = float(bout["b_age"]) if pd.notna(bout.get("b_age")) else None
    names = [base._display_name(red), base._display_name(blue)]

    phase_segments = {phase: 0 for phase in PHASES}
    striking = {
        (side, phase): {"landed": 0, "attempts": 0}
        for side in (0, 1)
        for phase in PHASES
    }
    total_events = 0
    finish_paths = 0

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)

    for i, seed in enumerate(seeds, start=1):
        sim, path, _, finish_round = full._run_prefix(
            red,
            blue,
            rounds=1,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        if path.finish is not None and finish_round == 1:
            finish_paths += 1

        for event in path.events:
            if int(event.get("round", 0)) != 1:
                continue
            phase = str(event.get("phase_start", ""))
            if phase not in PHASES:
                continue
            phase_segments[phase] += 1
            total_events += 1
            text = str(event.get("striking", ""))
            for side, name in enumerate(names):
                landed, attempts = _parse_striking(text, name)
                striking[(side, phase)]["landed"] += landed
                striking[(side, phase)]["attempts"] += attempts

        if i % 250 == 0 or i == args.paths:
            print(f"paths {i:,}/{args.paths:,}", flush=True)

    print("\n" + "=" * 108)
    print("R1 PHASE / STRIKING AUDIT — CURRENT LOCKED SHADOW CONFIGURATION")
    print("=" * 108)
    print(f"bout_id: {args.bout_id}")
    print(f"historical: {names[0]} (RED) vs {names[1]} (BLUE)")
    print(f"paths: {args.paths:,}")
    print(f"R1 KO/TKO paths: {finish_paths:,} ({finish_paths / args.paths:.2%})")
    print(f"mean simulated R1 duration observed: {total_events * 10.0 / args.paths:.1f}s")

    print("\nR1 PHASE OCCUPANCY")
    print(" phase       segments      share    mean sec/path")
    for phase in PHASES:
        segments = phase_segments[phase]
        share = segments / total_events if total_events else np.nan
        mean_sec = segments * 10.0 / args.paths
        print(f" {phase:8s} {segments:12,} {share:10.2%} {mean_sec:16.1f}")

    print("\nR1 STRIKING BY FIGHTER / PHASE — MEAN PER PATH")
    print(" side  fighter                       phase       landed   attempts  accuracy")
    for side, label in ((0, "RED"), (1, "BLUE")):
        for phase in PHASES:
            landed = striking[(side, phase)]["landed"]
            attempts = striking[(side, phase)]["attempts"]
            mean_landed = landed / args.paths
            mean_attempts = attempts / args.paths
            accuracy = landed / attempts if attempts else np.nan
            print(
                f" {label:4s}  {names[side]:28.28s} {phase:8s} "
                f"{mean_landed:8.3f} {mean_attempts:10.3f} {accuracy:9.2%}"
            )

    print("\nR1 TOTALS — MEAN PER PATH")
    for side, label in ((0, "RED"), (1, "BLUE")):
        landed = sum(striking[(side, phase)]["landed"] for phase in PHASES)
        attempts = sum(striking[(side, phase)]["attempts"] for phase in PHASES)
        accuracy = landed / attempts if attempts else np.nan
        print(
            f"{label:4s} {names[side]}: {landed / args.paths:.3f}/{attempts / args.paths:.3f} "
            f"sig ({accuracy:.2%})"
        )
    combined_landed = sum(v["landed"] for v in striking.values())
    combined_attempts = sum(v["attempts"] for v in striking.values())
    print(
        f"COMBINED: {combined_landed / args.paths:.3f}/{combined_attempts / args.paths:.3f} sig"
    )
    print("Research-only audit; no production or simulator physics modified.")


if __name__ == "__main__":
    main()
