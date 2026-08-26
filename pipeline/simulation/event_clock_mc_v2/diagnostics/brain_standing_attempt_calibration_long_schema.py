"""Long-schema adapter for the Brain standing-attempt calibration diagnostic.

Research only. This wrapper changes only how historical UFCStats distance
attempts are aggregated from data/fight_details/ufc_round_stats.parquet.
The underlying calibration simulation and candidate scale sweep are unchanged.
"""
from __future__ import annotations

import pandas as pd

from pipeline.simulation.event_clock_mc_v2.diagnostics import brain_standing_attempt_calibration as base


def actual_distance_attempts(round_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-format UFCStats distance attempts into red/blue bout totals."""
    required = {"fight_id", "corner", "distance_attempted"}
    missing = sorted(required.difference(round_stats.columns))
    if missing:
        raise KeyError(f"round stats missing required long-schema columns: {missing}")

    work = round_stats[["fight_id", "corner", "distance_attempted"]].copy()
    work["fight_id"] = work["fight_id"].astype(str)
    work["corner"] = work["corner"].astype(str).str.strip().str.lower()
    work["distance_attempted"] = pd.to_numeric(
        work["distance_attempted"], errors="coerce"
    ).fillna(0.0)

    def side(value: str) -> str:
        if value in {"red", "r"}:
            return "red"
        if value in {"blue", "b"}:
            return "blue"
        raise ValueError(f"unexpected UFCStats corner value: {value!r}")

    work["side"] = work["corner"].map(side)
    grouped = (
        work.groupby(["fight_id", "side"], as_index=False)["distance_attempted"]
        .sum()
        .pivot(index="fight_id", columns="side", values="distance_attempted")
        .fillna(0.0)
        .reset_index()
    )
    for needed in ("red", "blue"):
        if needed not in grouped.columns:
            grouped[needed] = 0.0
    return grouped[["fight_id", "red", "blue"]].rename(
        columns={
            "red": "red_distance_attempts",
            "blue": "blue_distance_attempts",
        }
    )


def main() -> None:
    base.actual_distance_attempts = actual_distance_attempts
    base.main()


if __name__ == "__main__":
    main()
