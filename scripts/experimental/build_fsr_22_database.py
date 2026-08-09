"""Build the shadow 22-trait pre-fight FSR database incrementally.

Loads the existing FSR-21 database unchanged, replays only the new
``reversal_ability`` rating from RFS history, and appends it at exact
fighter-fight grain.

Shadow/research only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_reversal_v1 as reversal


FSR21_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_21_shadow/"
    "fsr_21_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_22_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_22_prefight_snapshots.parquet"


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[["fight_id", "fighter_id"]].astype(str).to_numpy()))


def build_fsr_22_database(
    fsr21: pd.DataFrame,
    rfs: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]

    if fsr21.duplicated(keys).any():
        raise RuntimeError("FSR-21 violates fighter-fight grain")

    reversal_snapshots = reversal.build_prefight_snapshots(rfs, fsr21)
    if reversal_snapshots.duplicated(keys).any():
        raise RuntimeError("reversal snapshots violate fighter-fight grain")

    if _key_set(fsr21) != _key_set(reversal_snapshots):
        raise RuntimeError(
            "FSR-22 snapshot key mismatch: "
            f"fsr21={len(_key_set(fsr21))}, reversal={len(_key_set(reversal_snapshots))}"
        )

    keep = [
        *keys,
        reversal.SKILL,
        f"{reversal.SKILL}_updates",
    ]

    merged = fsr21.merge(
        reversal_snapshots[keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    rating = pd.to_numeric(merged[reversal.SKILL], errors="coerce")
    if rating.isna().any():
        raise RuntimeError("FSR-22 contains missing reversal ratings")
    if ((rating < reversal.MIN_RATING) | (rating > reversal.MAX_RATING)).any():
        raise RuntimeError("FSR-22 contains out-of-range reversal ratings")

    # Confirm the append did not mutate any existing FSR-21 values.
    original_columns = list(fsr21.columns)
    check = fsr21.merge(
        merged[keys + original_columns[2:]],
        on=keys,
        suffixes=("_21", "_22"),
        validate="one_to_one",
    )
    for column in original_columns:
        if column in keys:
            continue
        left = f"{column}_21"
        right = f"{column}_22"
        if left in check.columns and right in check.columns:
            if not check[left].equals(check[right]):
                raise RuntimeError(f"FSR-22 changed existing FSR-21 column: {column}")

    sort_columns = [c for c in ("date", "fight_id", "fighter_id") if c in merged.columns]
    return merged.sort_values(sort_columns).reset_index(drop=True)


def main() -> None:
    if not FSR21_PATH.exists():
        raise RuntimeError(f"FSR-21 database not found: {FSR21_PATH}")
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")

    fsr21 = pd.read_parquet(FSR21_PATH)
    rfs = pd.read_parquet(RFS_PATH)
    database = build_fsr_22_database(fsr21, rfs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(database):,} FSR-22 pre-fight rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
