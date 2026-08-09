"""Build the shadow 26-trait pre-fight FSR database incrementally.

Loads FSR-25 unchanged, appends only the newly replayed
`distance_striking_pressure` trait, and promotes the existing locked distance
ratings to canonical `distance_striking_*` names via compatibility aliases.

The legacy columns `distance_precision` and `distance_defense` are retained
unchanged so downstream code is not broken during the shadow transition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_distance_striking_pressure_v1 as distance


FSR25_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_25_shadow/"
    "fsr_25_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_26_prefight_snapshots.parquet"

LEGACY_PRECISION = "distance_precision"
LEGACY_DEFENSE = "distance_defense"
CANONICAL_PRECISION = "distance_striking_precision"
CANONICAL_DEFENSE = "distance_striking_defense"


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[["fight_id", "fighter_id"]].astype(str).to_numpy()))


def build_fsr_26_database(
    fsr25: pd.DataFrame,
    rfs: pd.DataFrame,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]

    if fsr25.duplicated(keys).any():
        raise RuntimeError("FSR-25 violates fighter-fight grain")

    for column in (LEGACY_PRECISION, LEGACY_DEFENSE):
        if column not in fsr25.columns:
            raise RuntimeError(f"FSR-25 missing locked distance rating: {column}")

    pressure = distance.build_prefight_snapshots(rfs, progress=progress)
    if pressure.duplicated(keys).any():
        raise RuntimeError("distance-pressure snapshots violate fighter-fight grain")

    if _key_set(fsr25) != _key_set(pressure):
        raise RuntimeError(
            "FSR-26 snapshot key mismatch: "
            f"fsr25={len(_key_set(fsr25))}, distance={len(_key_set(pressure))}"
        )

    keep = [
        *keys,
        distance.SKILL,
        f"{distance.SKILL}_updates",
    ]
    merged = fsr25.merge(
        pressure[keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    # Canonical names for the two already-locked distance ratings. Keep the
    # legacy columns unchanged as compatibility aliases during shadow use.
    merged[CANONICAL_PRECISION] = merged[LEGACY_PRECISION]
    merged[CANONICAL_DEFENSE] = merged[LEGACY_DEFENSE]

    # Carry update counts under canonical names when available.
    legacy_precision_updates = f"{LEGACY_PRECISION}_updates"
    legacy_defense_updates = f"{LEGACY_DEFENSE}_updates"
    if legacy_precision_updates in merged.columns:
        merged[f"{CANONICAL_PRECISION}_updates"] = merged[legacy_precision_updates]
    if legacy_defense_updates in merged.columns:
        merged[f"{CANONICAL_DEFENSE}_updates"] = merged[legacy_defense_updates]

    rating_columns = [
        distance.SKILL,
        CANONICAL_PRECISION,
        CANONICAL_DEFENSE,
    ]
    numeric = merged[rating_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise RuntimeError("FSR-26 contains missing distance-striking ratings")
    if ((numeric < distance.MIN_RATING) | (numeric > distance.MAX_RATING)).any().any():
        raise RuntimeError("FSR-26 contains out-of-range distance-striking ratings")

    # Alias contract: canonical precision/defense must be exact copies.
    if not merged[CANONICAL_PRECISION].equals(merged[LEGACY_PRECISION]):
        raise RuntimeError("distance precision compatibility alias changed values")
    if not merged[CANONICAL_DEFENSE].equals(merged[LEGACY_DEFENSE]):
        raise RuntimeError("distance defense compatibility alias changed values")

    sort_cols = [c for c in ("date", "fight_id", "fighter_id") if c in merged.columns]
    return merged.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    print(f"[FSR-26] loading FSR-25 from {FSR25_PATH}", flush=True)
    if not FSR25_PATH.exists():
        raise RuntimeError(f"FSR-25 database not found: {FSR25_PATH}")
    fsr25 = pd.read_parquet(FSR25_PATH)
    print(f"[FSR-26] loaded {len(fsr25):,} FSR-25 rows", flush=True)

    print(f"[FSR-26] loading RFS history from {RFS_PATH}", flush=True)
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[FSR-26] loaded {len(rfs):,} RFS rows", flush=True)

    print("[FSR-26] replaying distance-striking pressure", flush=True)
    database = build_fsr_26_database(fsr25, rfs, progress=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[FSR-26] writing {len(database):,} rows to {OUTPUT_PATH}", flush=True)
    database.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(database):,} FSR-26 pre-fight rows to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
