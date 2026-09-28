"""Real-data parity audit for the preserved FSR-32 physical trait contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v2.physical import PHYSICAL_COLUMNS, build_physical_snapshots


FSR32_PREFIGHT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)


def main() -> None:
    if not FSR32_PREFIGHT_PATH.exists():
        raise FileNotFoundError(
            "FSR-32 parity artifact not found: "
            f"{FSR32_PREFIGHT_PATH}"
        )

    snapshots = build_physical_snapshots()
    old = pd.read_parquet(FSR32_PREFIGHT_PATH)
    new = snapshots.prefight

    keys = ["fight_id", "fighter_id"]
    old["fight_id"] = old["fight_id"].astype(str)
    old["fighter_id"] = old["fighter_id"].astype(str)
    new = new.copy()
    new["fight_id"] = new["fight_id"].astype(str)
    new["fighter_id"] = new["fighter_id"].astype(str)

    missing_old_columns = [c for c in PHYSICAL_COLUMNS if c not in old.columns]
    if missing_old_columns:
        raise RuntimeError(
            f"FSR-32 artifact missing physical columns: {missing_old_columns}"
        )

    old_keys = set(map(tuple, old[keys].to_numpy()))
    new_keys = set(map(tuple, new[keys].to_numpy()))

    print("=" * 110)
    print("FSR V2 PHYSICAL TRAIT PARITY VS FSR-32")
    print("=" * 110)
    print(f"old prefight rows : {len(old):,}")
    print(f"new prefight rows : {len(new):,}")
    print(f"old keys          : {len(old_keys):,}")
    print(f"new keys          : {len(new_keys):,}")
    print(f"missing in new    : {len(old_keys - new_keys):,}")
    print(f"extra in new      : {len(new_keys - old_keys):,}")

    if old_keys != new_keys:
        raise RuntimeError("FSR V2 physical prefight key set does not match FSR-32")

    joined = old[[*keys, *PHYSICAL_COLUMNS]].merge(
        new[[*keys, *PHYSICAL_COLUMNS]],
        on=keys,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    for column in PHYSICAL_COLUMNS:
        old_values = pd.to_numeric(joined[f"{column}_old"], errors="raise").to_numpy(float)
        new_values = pd.to_numeric(joined[f"{column}_new"], errors="raise").to_numpy(float)
        diff = new_values - old_values
        rows.append(
            {
                "field": column,
                "rows": len(diff),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "gt_1e-12": int((np.abs(diff) > 1e-12).sum()),
            }
        )

    audit = pd.DataFrame(rows)
    print("\n" + audit.to_string(index=False, float_format=lambda x: f"{x:.12f}"))

    learned = [column for column in PHYSICAL_COLUMNS if column != "stamina_capacity"]
    latest = snapshots.latest
    numeric = latest[list(PHYSICAL_COLUMNS)].apply(pd.to_numeric, errors="coerce")

    print("\n" + "=" * 110)
    print("LATEST PHYSICAL STATE")
    print("=" * 110)
    print(f"fighters : {latest['fighter_id'].nunique():,}")
    print(f"rows     : {len(latest):,}")
    print(f"nan      : {int(numeric.isna().sum().sum()):,}")
    print(f"inf      : {int(np.isinf(numeric.to_numpy(float)).sum()):,}")
    print(f"capacity : {sorted(numeric['stamina_capacity'].unique().tolist())}")
    print(
        "learned ranges:\n"
        + pd.DataFrame(
            {
                "min": numeric[learned].min(),
                "max": numeric[learned].max(),
            }
        ).to_string(float_format=lambda x: f"{x:.6f}")
    )

    if audit["max_abs_diff"].max() > 1e-12:
        raise RuntimeError("FSR V2 physical prefight values are not exact FSR-32 parity")
    if numeric.isna().any().any() or np.isinf(numeric.to_numpy(float)).any():
        raise RuntimeError("latest physical FSR contains NaN/inf")
    if not np.allclose(numeric["stamina_capacity"].to_numpy(float), 100.0):
        raise RuntimeError("latest stamina_capacity is not fixed at 100")

    print("\nPASS: historical physical traits exactly match FSR-32.")
    print("PASS: latest physical state includes post-last-fight evidence.")
    print("PASS: no neutral 50 substitution is required by canonical FSR V2 publication.")


if __name__ == "__main__":
    main()
