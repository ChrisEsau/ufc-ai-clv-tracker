"""Population audit for the leakage-safe FSR-28 reservoir traits.

This audit verifies that the production-shaped shadow transforms preserve the
research signal discovered in Studies 7-9 before any simulator consumes them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir


FSR_28_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_28_shadow/"
    "fsr_28_prefight_snapshots.parquet"
)
RFS_PATH = reservoir.RFS_PATH


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"x": x[mask], "y": y[mask]}).corr(method="spearman").iloc[0, 1])


def _quintiles(frame: pd.DataFrame, signal: str, outcome: str, label: str) -> pd.DataFrame:
    work = frame.dropna(subset=[signal, outcome]).copy()
    work["bucket"] = pd.cut(
        work[signal].rank(method="first", pct=True),
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in work.groupby("bucket", observed=True, sort=False):
        rows.append(
            {
                "trait": label,
                "bucket": str(bucket),
                "fighter_fights": int(len(group)),
                "mean_rating": float(group[signal].mean()),
                "mean_updates": float(group[f"{signal}_updates"].mean()),
                "outcome_probability": float(group[outcome].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not FSR_28_PATH.exists():
        raise RuntimeError(f"FSR-28 artifact not found: {FSR_28_PATH}")
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")

    fsr = pd.read_parquet(FSR_28_PATH)
    rfs = pd.read_parquet(RFS_PATH)

    keys = ["fight_id", "fighter_id"]
    for frame in (fsr, rfs):
        for key in keys:
            frame[key] = frame[key].astype(str)

    required = {
        *keys,
        "knockdown_resistance",
        "knockdown_resistance_updates",
        "damage_durability",
        "damage_durability_updates",
    }
    missing = sorted(required - set(fsr.columns))
    if missing:
        raise RuntimeError(f"FSR-28 missing columns: {missing}")

    eval_cols = [
        *keys,
        reservoir.KD_COL,
        reservoir.SIG_ABS_COL,
        reservoir.KO_LOSS_COL,
    ]
    missing = sorted(set(eval_cols) - set(rfs.columns))
    if missing:
        raise RuntimeError(f"RFS missing audit columns: {missing}")

    work = fsr.merge(rfs[eval_cols], on=keys, how="inner", validate="one_to_one")
    for col in (reservoir.KD_COL, reservoir.SIG_ABS_COL, reservoir.KO_LOSS_COL):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["kd_indicator"] = (work[reservoir.KD_COL].fillna(0.0) > 0).astype(int)
    work["kd_per_sig_absorbed"] = np.divide(
        work[reservoir.KD_COL].fillna(0.0),
        work[reservoir.SIG_ABS_COL],
        out=np.full(len(work), np.nan),
        where=pd.to_numeric(work[reservoir.SIG_ABS_COL], errors="coerce").fillna(0.0).to_numpy() > 0,
    )

    print("=" * 112)
    print("FSR-28 RESERVOIR TRAIT POPULATION AUDIT")
    print("=" * 112)
    print(f"rows: {len(work):,}")
    print(f"unique fighter-fight keys: {work[keys].drop_duplicates().shape[0]:,}")

    for trait in reservoir.SKILLS:
        values = pd.to_numeric(work[trait], errors="coerce")
        print(
            f"{trait}: min={values.min():.3f} p10={values.quantile(.10):.3f} "
            f"median={values.median():.3f} p90={values.quantile(.90):.3f} max={values.max():.3f}"
        )

    eligible_kd = work[work["knockdown_resistance_updates"] >= 2].copy()
    eligible_dur = work[work["damage_durability_updates"] >= 2].copy()

    print("\nKNOCKDOWN RESISTANCE — >=2 PRIOR FIGHTS")
    print(f"eligible rows: {len(eligible_kd):,}")
    print(
        "Spearman vs current KD indicator: "
        f"{_spearman(eligible_kd['knockdown_resistance'], eligible_kd['kd_indicator']):.5f}"
    )
    print(
        "Spearman vs current KD/sig absorbed: "
        f"{_spearman(eligible_kd['knockdown_resistance'], eligible_kd['kd_per_sig_absorbed']):.5f}"
    )
    print(
        _quintiles(
            eligible_kd,
            "knockdown_resistance",
            "kd_indicator",
            "knockdown_resistance",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nDAMAGE DURABILITY — >=2 PRIOR FIGHTS")
    print(f"eligible rows: {len(eligible_dur):,}")
    print(
        "Spearman vs current KO/TKO loss: "
        f"{_spearman(eligible_dur['damage_durability'], eligible_dur[reservoir.KO_LOSS_COL]):.5f}"
    )
    print(
        _quintiles(
            eligible_dur,
            "damage_durability",
            reservoir.KO_LOSS_COL,
            "damage_durability",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print(
        "\nPASS DIRECTION: knockdown_resistance should trend negatively against KD outcomes; "
        "damage_durability should trend negatively against KO/TKO loss. Do not wire these "
        "traits into the simulator if the leakage-safe transform destroys those directions."
    )


if __name__ == "__main__":
    main()
