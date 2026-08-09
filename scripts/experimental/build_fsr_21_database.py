"""Build the shadow 21-trait pre-fight FSR database.

Reuses the existing FSR-18 database components unchanged, then merges the
three-candidate ground-striking family at exact fighter-fight grain.

Shadow/research only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_dynamic_families_v1 as dynamic
from scripts.experimental import fsr_ground_striking_v1 as ground
from scripts.experimental import fsr_locked_families_v1 as locked
from scripts.experimental.run_fsr_v1_5_2026_replay import (
    build_locked_prefight_snapshots,
)


OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_21_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_21_prefight_snapshots.parquet"


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[["fight_id", "fighter_id"]].astype(str).to_numpy()))


def build_fsr_21_database(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
) -> pd.DataFrame:
    locked_snapshots = build_locked_prefight_snapshots(rfs)
    dynamic_snapshots = dynamic.build_prefight_snapshots(rfs, rounds)
    ground_snapshots = ground.build_prefight_snapshots(rfs)

    keys = ["fight_id", "fighter_id"]
    for label, frame in (
        ("locked", locked_snapshots),
        ("dynamic", dynamic_snapshots),
        ("ground", ground_snapshots),
    ):
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{label} snapshots violate fighter-fight grain")

    locked_keys = _key_set(locked_snapshots)
    dynamic_keys = _key_set(dynamic_snapshots)
    ground_keys = _key_set(ground_snapshots)

    if locked_keys != dynamic_keys or locked_keys != ground_keys:
        raise RuntimeError(
            "FSR-21 snapshot key mismatch: "
            f"locked={len(locked_keys)}, dynamic={len(dynamic_keys)}, ground={len(ground_keys)}"
        )

    dynamic_keep = [
        *keys,
        *dynamic.SKILLS,
        *[f"{skill}_updates" for skill in dynamic.SKILLS],
    ]
    ground_keep = [
        *keys,
        *ground.SKILLS,
        *[f"{skill}_updates" for skill in ground.SKILLS],
    ]

    merged = locked_snapshots.merge(
        dynamic_snapshots[dynamic_keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    ).merge(
        ground_snapshots[ground_keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    rating_columns = [*locked.SKILLS, *dynamic.SKILLS, *ground.SKILLS]
    if len(rating_columns) != 21:
        raise RuntimeError(
            f"FSR shadow ontology must contain 21 traits, found {len(rating_columns)}"
        )

    numeric = merged[rating_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise RuntimeError("FSR-21 contains missing ratings")
    if ((numeric < locked.MIN_RATING) | (numeric > locked.MAX_RATING)).any().any():
        raise RuntimeError("FSR-21 contains out-of-range ratings")

    return merged.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def main() -> None:
    if not locked.RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {locked.RFS_PATH}")
    if not dynamic.ROUND_PATH.exists():
        raise RuntimeError(f"round stats not found: {dynamic.ROUND_PATH}")

    rfs = pd.read_parquet(locked.RFS_PATH)
    rounds = pd.read_parquet(dynamic.ROUND_PATH)
    database = build_fsr_21_database(rfs, rounds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(database):,} FSR-21 pre-fight rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
