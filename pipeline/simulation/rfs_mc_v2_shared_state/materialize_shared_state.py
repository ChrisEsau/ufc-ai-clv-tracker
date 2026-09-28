"""Materialize validated shared RFS state for Monte Carlo V2.

This is a shadow-only cache layer. It does not overwrite production
Round Fighter State artifacts.

Run from repo root:

    python -m pipeline.simulation.rfs_mc_v2_shared_state.materialize_shared_state
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.round_stats.build_round_fighter_state import (
    build_round_fighter_state,
)


OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state"
)

HISTORY_PATH = OUTPUT_DIR / "historical_fighter_state.parquet"
LATEST_PATH = OUTPUT_DIR / "latest_fighter_state.parquet"

EXPECTED_HISTORY_ROWS = 13_390
EXPECTED_LATEST_ROWS = 2_186

CURRENT_FIGHT_PREFIXES = (
    "rfs_traj_fight_",
    "rfs_open_fight_",
    "rfs_phase_base_fight_",
    "rfs_phase_interact_fight_",
    "rfs_dynamic_response_fight_",
    "rfs_finish_state_fight_",
)


class SharedStateMaterializationError(RuntimeError):
    """Raised when the shadow RFS cache fails validation."""


def validate_shared_state(
    history: pd.DataFrame,
    latest: pd.DataFrame,
) -> None:
    """Validate critical shared-state invariants before writing."""

    if len(history) != EXPECTED_HISTORY_ROWS:
        raise SharedStateMaterializationError(
            f"unexpected history rows: {len(history)} "
            f"(expected {EXPECTED_HISTORY_ROWS})"
        )

    if len(latest) != EXPECTED_LATEST_ROWS:
        raise SharedStateMaterializationError(
            f"unexpected latest rows: {len(latest)} "
            f"(expected {EXPECTED_LATEST_ROWS})"
        )

    history_duplicates = int(
        history.duplicated(
            subset=["fight_id", "fighter_id"]
        ).sum()
    )

    if history_duplicates:
        raise SharedStateMaterializationError(
            f"history contains {history_duplicates} duplicate "
            "fight/fighter rows"
        )

    latest_duplicates = int(
        latest.duplicated(
            subset=["fighter_id"]
        ).sum()
    )

    if latest_duplicates:
        raise SharedStateMaterializationError(
            f"latest contains {latest_duplicates} duplicate fighters"
        )

    leaked_latest_columns = [
        column
        for column in latest.columns
        if column.startswith(CURRENT_FIGHT_PREFIXES)
    ]

    if leaked_latest_columns:
        raise SharedStateMaterializationError(
            "latest state contains current-fight observations: "
            f"{leaked_latest_columns[:10]}"
        )

    family_prefixes = (
        "rfs_phase_base_",
        "rfs_phase_interact_",
        "rfs_dynamic_response_",
        "rfs_finish_state_",
    )

    for prefix in family_prefixes:
        if not any(
            column.startswith(prefix)
            for column in history.columns
        ):
            raise SharedStateMaterializationError(
                f"history is missing family prefix {prefix}"
            )

        if not any(
            column.startswith(prefix)
            for column in latest.columns
        ):
            raise SharedStateMaterializationError(
                f"latest is missing family prefix {prefix}"
            )


def materialize_shared_state() -> tuple[Path, Path]:
    """Build, validate, and write shadow shared-state cache artifacts."""

    print("=" * 78)
    print("MATERIALIZE RFS MC V2 SHARED STATE")
    print("=" * 78)

    result = build_round_fighter_state()

    history = result.history_df
    latest = result.latest_df

    print(
        f"Built history: {history.shape[0]:,} rows x "
        f"{history.shape[1]:,} columns"
    )
    print(
        f"Built latest : {latest.shape[0]:,} rows x "
        f"{latest.shape[1]:,} columns"
    )

    validate_shared_state(
        history,
        latest,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_parquet(
        HISTORY_PATH,
        index=False,
    )
    latest.to_parquet(
        LATEST_PATH,
        index=False,
    )

    print()
    print(f"History cache: {HISTORY_PATH}")
    print(f"Latest cache : {LATEST_PATH}")
    print("SHADOW SHARED STATE MATERIALIZED")

    return HISTORY_PATH, LATEST_PATH


def main() -> None:
    """CLI entry point."""

    materialize_shared_state()


if __name__ == "__main__":
    main()
