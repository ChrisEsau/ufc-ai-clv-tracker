"""Run the standalone round-level UFC simulator in shadow mode.

Example:

    python -m pipeline.simulation.run_round_simulator \
        --input configs/simulation/example_matchup.json \
        --simulations 25000 \
        --seed 7

This command does not modify production prediction, betting, feature, or master
artifacts. It writes one JSON summary under the existing model-lab data area.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common.paths import MODEL_LAB_DIR, ensure_data_dirs
from pipeline.simulation.contracts import SimulatorConfig
from pipeline.simulation.engine import run_simulation
from pipeline.simulation.io import load_matchup, write_summary


DEFAULT_INPUT_PATH = Path("configs/simulation/example_matchup.json")
DEFAULT_OUTPUT_PATH = MODEL_LAB_DIR / "simulation" / "latest_round_simulation_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run round-level UFC Monte Carlo simulation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_data_dirs()

    matchup = load_matchup(args.input)
    config = SimulatorConfig(simulations=args.simulations, seed=args.seed)
    summary, _ = run_simulation(matchup, config)
    output_path = write_summary(summary, args.output)

    probabilities = summary.probabilities
    print("=" * 80)
    print("UFC ROUND SIMULATOR V0 — SHADOW OUTPUT")
    print("=" * 80)
    print(f"Fight: {summary.red_fighter_name} vs {summary.blue_fighter_name}")
    print(f"Simulations: {summary.simulations:,}")
    print(f"Red win: {probabilities['red_win']:.3%}")
    print(f"Blue win: {probabilities['blue_win']:.3%}")
    print(f"Goes distance: {probabilities['goes_distance']:.3%}")
    print(f"Inside distance: {probabilities['inside_distance']:.3%}")
    print(f"Expected fight time: {summary.expectations['fight_time_seconds']:.1f} sec")
    print(f"Wrote summary: {output_path}")
    print("WARNING: V0 mechanics are not calibrated for wagering or production promotion.")


if __name__ == "__main__":
    main()
