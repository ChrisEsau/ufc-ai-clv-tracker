"""Build the shadow FSR-28 pre-fight database.

FSR-28 preserves the existing FSR-26 artifact unchanged and appends the two
Damage Reservoir V1 defensive traits:

- knockdown_resistance
- damage_durability

Legacy ``chin_resistance`` and ``damage_resistance`` remain present for
compatibility and research comparison.  No production schema is modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir


FSR_26_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_28_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_28_prefight_snapshots.parquet"


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[["fight_id", "fighter_id"]].astype(str).to_numpy()))


def build_fsr_28_database(fsr_26: pd.DataFrame, rfs: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]

    base = fsr_26.copy()
    for key in keys:
        base[key] = base[key].astype(str)

    if base.duplicated(keys).any():
        raise RuntimeError("FSR-26 snapshots violate fighter-fight grain")

    reservoir_snapshots = reservoir.build_prefight_snapshots(rfs)
    if reservoir_snapshots.duplicated(keys).any():
        raise RuntimeError("reservoir snapshots violate fighter-fight grain")

    base_keys = _key_set(base)
    reservoir_keys = _key_set(reservoir_snapshots)
    if base_keys != reservoir_keys:
        raise RuntimeError(
            "FSR-28 snapshot key mismatch: "
            f"fsr26={len(base_keys)}, reservoir={len(reservoir_keys)}"
        )

    keep = [
        *keys,
        *reservoir.SKILLS,
        *[f"{skill}_updates" for skill in reservoir.SKILLS],
        "knockdown_resistance_evidence_score",
        "damage_durability_evidence_score",
    ]
    merged = base.merge(
        reservoir_snapshots[keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    # FSR-26 should contain 26 rating columns. We append exactly two new ratings.
    rating_columns = [
        col
        for col in base.columns
        if not col.endswith("_updates")
        and col not in {"fight_id", "fighter_id", "opponent_id", "date"}
        and pd.api.types.is_numeric_dtype(base[col])
    ]
    # Do not infer the ontology count from arbitrary metadata columns; the hard
    # requirement here is simply that both new traits exist and are bounded.
    new_ratings = merged[list(reservoir.SKILLS)].apply(pd.to_numeric, errors="coerce")
    if new_ratings.isna().any().any():
        raise RuntimeError("FSR-28 contains missing reservoir ratings")
    if ((new_ratings < reservoir.MIN_RATING) | (new_ratings > reservoir.MAX_RATING)).any().any():
        raise RuntimeError("FSR-28 contains out-of-range reservoir ratings")

    if len(merged) != len(base):
        raise RuntimeError(
            f"FSR-28 row mismatch: expected {len(base):,}, got {len(merged):,}"
        )

    sort_cols = [c for c in ("date", "fight_id", "fighter_id") if c in merged.columns]
    return merged.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    if not FSR_26_PATH.exists():
        raise RuntimeError(f"FSR-26 artifact not found: {FSR_26_PATH}")
    if not reservoir.RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {reservoir.RFS_PATH}")

    print(f"[FSR-28] loading FSR-26: {FSR_26_PATH}", flush=True)
    fsr_26 = pd.read_parquet(FSR_26_PATH)
    print(f"[FSR-28] FSR-26 rows: {len(fsr_26):,}", flush=True)

    print(f"[FSR-28] loading RFS history: {reservoir.RFS_PATH}", flush=True)
    rfs = pd.read_parquet(reservoir.RFS_PATH)
    print(f"[FSR-28] RFS rows: {len(rfs):,}", flush=True)

    database = build_fsr_28_database(fsr_26, rfs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(database):,} FSR-28 pre-fight rows to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
