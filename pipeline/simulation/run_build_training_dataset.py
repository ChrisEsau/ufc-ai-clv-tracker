"""Build the shadow simulator's leakage-safe fighter-round training table.

Run from the repository root:

    python -m pipeline.simulation.run_build_training_dataset

The command reads authoritative historical artifacts and writes only generated
model-lab/audit outputs. It does not change master, RFS, prediction, or betting
artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    ROUND_STATS_PATH,
    ensure_data_dirs,
)
from pipeline.simulation.artifacts import (
    SIMULATION_TRAINING_AUDIT_PATH,
    SIMULATION_TRAINING_DATASET_PATH,
    ensure_simulation_dirs,
)
from pipeline.simulation.historical_training import (
    build_historical_simulation_training_dataset,
)
from pipeline.simulation.parameter_models import validate_training_targets


DEFAULT_STATE_PATHS: Mapping[str, Path] = {
    "trajectory": ROUND_FIGHTER_STATE_HISTORY_PATH,
    "suppression": ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    "wrestling": ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    "defense": ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
}


class SimulationTrainingRunnerError(RuntimeError):
    """Raised when source artifacts cannot be loaded or outputs cannot be saved."""


def _read_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SimulationTrainingRunnerError(f"{label} not found: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - backend/environment dependent
        raise SimulationTrainingRunnerError(f"Could not read {label} at {path}: {exc}") from exc


def _load_state_sources(
    include_rfs: bool,
    require_all_rfs: bool,
) -> dict[str, pd.DataFrame]:
    if not include_rfs:
        return {}

    sources: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for source_name, path in DEFAULT_STATE_PATHS.items():
        if not path.exists():
            missing.append(f"{source_name}={path}")
            continue
        sources[source_name] = _read_required_parquet(path, f"{source_name} RFS history")

    if require_all_rfs and missing:
        raise SimulationTrainingRunnerError(
            "Required RFS history artifacts are missing: " + ", ".join(missing)
        )

    if missing:
        print("Optional RFS sources not found; continuing without them:")
        for item in missing:
            print(f"  - {item}")

    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe fighter-round simulator parameter training data"
    )
    parser.add_argument("--round-stats", type=Path, default=ROUND_STATS_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--output", type=Path, default=SIMULATION_TRAINING_DATASET_PATH)
    parser.add_argument("--audit-output", type=Path, default=SIMULATION_TRAINING_AUDIT_PATH)
    parser.add_argument(
        "--without-rfs",
        action="store_true",
        help="Build targets and prior-round context without joining RFS histories.",
    )
    parser.add_argument(
        "--require-all-rfs",
        action="store_true",
        help="Fail when any registered RFS history artifact is unavailable.",
    )
    return parser


def _audit_value(audit: pd.DataFrame, check: str) -> object | None:
    match = audit.loc[audit["check"].eq(check), "value"]
    if match.empty:
        return None
    return match.iloc[0]


def main() -> None:
    args = build_parser().parse_args()
    ensure_data_dirs()
    ensure_simulation_dirs()

    rounds = _read_required_parquet(args.round_stats, "round stats")
    master = _read_required_parquet(args.master, "master fights")
    state_sources = _load_state_sources(
        include_rfs=not args.without_rfs,
        require_all_rfs=args.require_all_rfs,
    )

    result = build_historical_simulation_training_dataset(
        round_stats_df=rounds,
        master_df=master,
        state_sources=state_sources,
    )
    validate_training_targets(result.dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    result.dataset.to_parquet(args.output, index=False)
    result.audit.to_parquet(args.audit_output, index=False)

    dataset = result.dataset
    excluded_fights = _audit_value(
        result.audit,
        "historical_excluded_nonstandard_round_fights",
    )
    excluded_rows = _audit_value(
        result.audit,
        "historical_excluded_nonstandard_fighter_round_rows",
    )

    print("=" * 80)
    print("UFC SIMULATOR FIGHTER-ROUND TRAINING DATASET")
    print("=" * 80)
    print(f"Rows: {len(dataset):,}")
    print(f"Fights: {dataset['fight_id'].nunique():,}")
    print(f"Fighters: {dataset['fighter_id'].nunique():,}")
    print(f"Columns: {len(dataset.columns):,}")
    print(f"RFS sources joined: {', '.join(state_sources) if state_sources else 'none'}")
    if excluded_fights is not None:
        print(f"Excluded nonstandard/missing scheduled-round fights: {int(excluded_fights):,}")
    if excluded_rows is not None:
        print(f"Excluded fighter-round rows from those fights: {int(excluded_rows):,}")
    print(f"Training dataset: {args.output}")
    print(f"Validation audit: {args.audit_output}")
    print("Shadow-only artifact. No production model or betting contract was changed.")


if __name__ == "__main__":
    main()
