"""Build the shadow 18-trait pre-fight FSR database.

The locked 13-skill replay is reused without modification. Five new dynamic
ratings are replayed independently, then merged at fighter-fight grain.

Shadow/research only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_dynamic_families_v1 as dynamic
from scripts.experimental import fsr_locked_families_v1 as locked
from scripts.experimental.run_fsr_v1_5_2026_replay import (
    build_locked_prefight_snapshots,
)


OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_18_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_18_prefight_snapshots.parquet"


def build_fsr_18_database(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
) -> pd.DataFrame:
    locked_snapshots = build_locked_prefight_snapshots(rfs)
    dynamic_snapshots = dynamic.build_prefight_snapshots(rfs, rounds)

    keys = ["fight_id", "fighter_id"]
    if locked_snapshots.duplicated(keys).any():
        raise RuntimeError("locked snapshots violate fighter-fight grain")
    if dynamic_snapshots.duplicated(keys).any():
        raise RuntimeError("dynamic snapshots violate fighter-fight grain")

    locked_key_set = set(map(tuple, locked_snapshots[keys].astype(str).to_numpy()))
    dynamic_key_set = set(map(tuple, dynamic_snapshots[keys].astype(str).to_numpy()))
    if locked_key_set != dynamic_key_set:
        missing_dynamic = len(locked_key_set - dynamic_key_set)
        missing_locked = len(dynamic_key_set - locked_key_set)
        raise RuntimeError(
            "locked/dynamic snapshot key mismatch: "
            f"missing_dynamic={missing_dynamic}, missing_locked={missing_locked}"
        )

    dynamic_keep = [
        *keys,
        *dynamic.SKILLS,
        *[f"{skill}_updates" for skill in dynamic.SKILLS],
    ]
    merged = locked_snapshots.merge(
        dynamic_snapshots[dynamic_keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    rating_columns = [*locked.SKILLS, *dynamic.SKILLS]
    if len(rating_columns) != 18:
        raise RuntimeError(f"FSR ontology must contain 18 traits, found {len(rating_columns)}")

    numeric = merged[rating_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise RuntimeError("FSR-18 contains missing ratings")
    if ((numeric < locked.MIN_RATING) | (numeric > locked.MAX_RATING)).any().any():
        raise RuntimeError("FSR-18 contains out-of-range ratings")

    return merged.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def main() -> None:
    if not locked.RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {locked.RFS_PATH}")
    if not dynamic.ROUND_PATH.exists():
        raise RuntimeError(f"round stats not found: {dynamic.ROUND_PATH}")

    rfs = pd.read_parquet(locked.RFS_PATH)
    rounds = pd.read_parquet(dynamic.ROUND_PATH)
    database = build_fsr_18_database(rfs, rounds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(database):,} FSR-18 pre-fight rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
