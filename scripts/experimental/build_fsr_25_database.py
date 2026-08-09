"""Build the shadow 25-trait pre-fight FSR database incrementally.

Loads FSR-22 unchanged, replays only the three clinch-striking ratings, and
appends them at exact fighter-fight grain. Shadow/research only.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from scripts.experimental import fsr_clinch_striking_v1 as clinch

FSR22_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_22_shadow/"
    "fsr_22_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_25_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_25_prefight_snapshots.parquet"


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[["fight_id", "fighter_id"]].astype(str).to_numpy()))


def build_fsr_25_database(fsr22: pd.DataFrame, rfs: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    if fsr22.duplicated(keys).any():
        raise RuntimeError("FSR-22 violates fighter-fight grain")

    clinch_snapshots = clinch.build_prefight_snapshots(rfs)
    if clinch_snapshots.duplicated(keys).any():
        raise RuntimeError("clinch snapshots violate fighter-fight grain")

    if _key_set(fsr22) != _key_set(clinch_snapshots):
        raise RuntimeError(
            "FSR-25 snapshot key mismatch: "
            f"fsr22={len(_key_set(fsr22))}, clinch={len(_key_set(clinch_snapshots))}"
        )

    keep = [
        *keys,
        *clinch.SKILLS,
        *[f"{skill}_updates" for skill in clinch.SKILLS],
    ]
    merged = fsr22.merge(
        clinch_snapshots[keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    numeric = merged[list(clinch.SKILLS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise RuntimeError("FSR-25 contains missing clinch ratings")
    if ((numeric < clinch.MIN_RATING) | (numeric > clinch.MAX_RATING)).any().any():
        raise RuntimeError("FSR-25 contains out-of-range clinch ratings")

    sort_cols = [c for c in ("date", "fight_id", "fighter_id") if c in merged.columns]
    return merged.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    if not FSR22_PATH.exists():
        raise RuntimeError(f"FSR-22 database not found: {FSR22_PATH}")
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")

    fsr22 = pd.read_parquet(FSR22_PATH)
    rfs = pd.read_parquet(RFS_PATH)
    database = build_fsr_25_database(fsr22, rfs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(database):,} FSR-25 pre-fight rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
