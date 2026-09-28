"""Compatibility runner for the pressure -> phase-volume predictive audit.

Root-cause fix only: the original audit trimmed leakage-safe FSR-28 rows down to
fighter id + pressure columns before passing them into StaticFSRMCDamageV1. The
simulator correctly requires its full base feature contract plus the three damage
traits. This runner preserves the complete FSR row while keeping the original
audit logic unchanged.

No simulator constants or architecture are modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_pressure_phase_volume_predictive_test as audit


def _load_full_fsr_rows(path: Path, bout_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    bout_key = "fight_id" if "fight_id" in frame.columns else "bout_id"

    required = (
        {bout_key, "fighter_id"}
        | set(audit.PRESSURE_COLUMNS.values())
        | set(damage.base.REQUIRED_COLUMNS)
        | set(damage.REQUIRED_DAMAGE_COLUMNS)
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR artifact missing required audit/simulator columns: {missing}")

    frame[bout_key] = frame[bout_key].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame = frame[frame[bout_key].isin(bout_ids)].copy()

    if bout_key != "bout_id":
        frame = frame.rename(columns={bout_key: "bout_id"})

    if frame.duplicated(["bout_id", "fighter_id"]).any():
        raise ValueError("FSR cohort has duplicate bout/fighter rows.")

    counts = frame.groupby("bout_id")["fighter_id"].nunique()
    bad = counts[counts != 2]
    if not bad.empty:
        raise ValueError(
            f"Expected two leakage-safe FSR fighters per bout; bad bouts={len(bad)}"
        )

    # Critical compatibility fix: return the complete leakage-safe FSR profile.
    return frame.copy()


def main() -> None:
    audit._load_fsr_rows = _load_full_fsr_rows
    audit.main()


if __name__ == "__main__":
    main()
