"""Research-only Allen vs Shahbazyan last-5 FSR shadow.

Uses the repository's existing recency builder with WINDOW=5 for V3-native
fight-relevant traits, then runs the same 500-path calibrated standing-attempt
shadow (scale 0.25) with mechanics, judging, pressure logic, and seeds frozen.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency
from pipeline.simulation.event_clock_mc_v2.diagnostics import allen_shahbazyan_calibrated_attempt_shadow as base_shadow

BACKUP_PATH = Path("data/fsr_v3/fsr_v3_prefight_snapshots.canonical_backup.parquet")


def main() -> None:
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)

    recency.WINDOW = 5
    last5 = recency.build_variant(canonical, "last3")

    shutil.copy2(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, BACKUP_PATH)
    try:
        last5.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        print("FSR_VARIANT=last5_all_v3_native")
        print("FSR_WINDOW=5")
        base_shadow.main()
    finally:
        shutil.move(BACKUP_PATH, FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
