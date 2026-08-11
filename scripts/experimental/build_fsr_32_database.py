"""Build the shadow FSR-32 simulator-facing stamina contract.

FSR-32 starts from FSR-28 and appends three explicit fighter stamina parameters:

- stamina_capacity
- stamina_depletion_resistance
- stamina_performance_resilience

The two rating-like stamina parameters are exact aliases of existing leakage-safe
FSR cardio traits. The capacity parameter is explicitly stored in the FSR
artifact rather than being invented inside the simulator.

Recovery is intentionally NOT fighter-specific in this shadow contract.
Population audit showed that the historical ``recovery_ability`` proxy is
functionally non-informative for fighter differentiation (81.5% of rows exactly
50; standard deviation about 0.30) and it is not a direct physiological measure
of one-minute corner recovery. Both ``recovery_ability`` and the former
``stamina_recovery_ability`` alias are therefore removed from the FSR-32 output.
Recovery physics belongs in the simulator until a genuinely informative fighter
trait exists.

Shadow/research only. No production schema is modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FSR_28_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_28_shadow/"
    "fsr_28_prefight_snapshots.parquet"
)
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_32_prefight_snapshots.parquet"

STAMINA_CAPACITY = "stamina_capacity"
STAMINA_DEPLETION_RESISTANCE = "stamina_depletion_resistance"
STAMINA_PERFORMANCE_RESILIENCE = "stamina_performance_resilience"

DEFAULT_STAMINA_CAPACITY = 100.0

STAMINA_SOURCE_COLUMNS = {
    STAMINA_DEPLETION_RESISTANCE: "fatigue_accumulation_resistance",
    STAMINA_PERFORMANCE_RESILIENCE: "fatigue_performance_resilience",
}

STAMINA_COLUMNS = (
    STAMINA_CAPACITY,
    STAMINA_DEPLETION_RESISTANCE,
    STAMINA_PERFORMANCE_RESILIENCE,
)

REMOVED_RECOVERY_COLUMNS = (
    "recovery_ability",
    "stamina_recovery_ability",
)


def build_fsr_32_database(fsr_28: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    base = fsr_28.copy()

    missing = [
        column
        for column in [*keys, *STAMINA_SOURCE_COLUMNS.values()]
        if column not in base.columns
    ]
    if missing:
        raise RuntimeError(f"FSR-28 missing required stamina-source columns: {missing}")
    if base.duplicated(keys).any():
        raise RuntimeError("FSR-28 violates fighter-fight grain")

    # The recovery proxy is intentionally removed from the simulator-facing FSR.
    base = base.drop(columns=list(REMOVED_RECOVERY_COLUMNS), errors="ignore")

    base[STAMINA_CAPACITY] = float(DEFAULT_STAMINA_CAPACITY)
    for target, source in STAMINA_SOURCE_COLUMNS.items():
        base[target] = pd.to_numeric(base[source], errors="coerce")

    numeric = base[list(STAMINA_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("FSR-32 contains missing or non-finite stamina parameters")
    if (numeric[STAMINA_CAPACITY] <= 0.0).any():
        raise RuntimeError("FSR-32 stamina_capacity must be positive")

    for column in (
        STAMINA_DEPLETION_RESISTANCE,
        STAMINA_PERFORMANCE_RESILIENCE,
    ):
        if ((numeric[column] < 10.0) | (numeric[column] > 90.0)).any():
            raise RuntimeError(f"FSR-32 {column} is outside the established 10-90 FSR range")

    for target, source in STAMINA_SOURCE_COLUMNS.items():
        if not np.allclose(
            pd.to_numeric(base[target], errors="coerce"),
            pd.to_numeric(base[source], errors="coerce"),
            equal_nan=False,
        ):
            raise RuntimeError(f"FSR-32 alias mismatch: {target} != {source}")

    if any(column in base.columns for column in REMOVED_RECOVERY_COLUMNS):
        raise RuntimeError("FSR-32 recovery columns were not removed")

    sort_cols = [c for c in ("date", "fight_id", "fighter_id") if c in base.columns]
    return base.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    if not FSR_28_PATH.exists():
        raise RuntimeError(f"FSR-28 artifact not found: {FSR_28_PATH}")

    print(f"[FSR-32] loading FSR-28: {FSR_28_PATH}", flush=True)
    fsr_28 = pd.read_parquet(FSR_28_PATH)
    print(f"[FSR-32] FSR-28 rows: {len(fsr_28):,}", flush=True)

    database = build_fsr_32_database(fsr_28)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)

    print(f"Wrote {len(database):,} FSR-32 pre-fight rows to {OUTPUT_PATH}", flush=True)
    print("FSR-32 stamina contract:", flush=True)
    for column in STAMINA_COLUMNS:
        print(f"  - {column}", flush=True)
    print("Removed recovery columns:", flush=True)
    for column in REMOVED_RECOVERY_COLUMNS:
        print(f"  - {column}", flush=True)
    print("Between-round recovery is simulator-global physics.", flush=True)


if __name__ == "__main__":
    main()
