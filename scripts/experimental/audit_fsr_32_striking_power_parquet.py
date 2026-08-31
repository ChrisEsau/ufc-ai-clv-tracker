"""Direct audit of the written FSR-32 parquet striking_power column.

Reads the actual fsr_32_prefight_snapshots.parquet artifact and reports:
- row count, schema/grain checks
- historical and latest-fighter striking_power distributions
- top/bottom/near-50 latest fighters
- selected reference fighters
- monotonicity after first demonstrated-power state (>50): no later decline allowed

Research only; does not modify artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
POWER = "striking_power"

REFERENCE_NAMES = [
    "Francis Ngannou",
    "Derrick Lewis",
    "Sergei Pavlovich",
    "Tom Aspinall",
    "Jiri Prochazka",
    "Alexander Volkanovski",
    "Sean O'Malley",
    "Manel Kape",
    "Raquel Pennington",
    "Erin Blanchfield",
]


def latest_per_fighter(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ("date", "fight_id") if c in df.columns]
    if sort_cols:
        ordered = df.sort_values(["fighter_id", *sort_cols])
    else:
        ordered = df.copy()
    return ordered.groupby("fighter_id", as_index=False).tail(1).reset_index(drop=True)


def print_distribution(label: str, s: pd.Series) -> None:
    s = pd.to_numeric(s, errors="coerce")
    print(f"\n{label}")
    print(f"  n: {s.notna().sum():,}")
    print(f"  missing: {s.isna().sum():,}")
    print(f"  mean: {s.mean():.3f}")
    print(f"  median: {s.median():.3f}")
    print(f"  std: {s.std(ddof=0):.3f}")
    print(f"  min/max: {s.min():.3f} / {s.max():.3f}")
    print(f"  below 50: {(s < 50).sum():,}")
    print(f"  exactly 50: {np.isclose(s, 50.0, equal_nan=False).sum():,}")
    print(f"  above 50: {(s > 50).sum():,}")
    for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        print(f"  p{p:>3}: {np.nanpercentile(s, p):8.3f}")


def main() -> None:
    if not PATH.exists():
        raise FileNotFoundError(PATH)

    df = pd.read_parquet(PATH)
    print("=" * 120)
    print("FSR-32 WRITTEN PARQUET — STRIKING POWER AUDIT")
    print("=" * 120)
    print(f"path: {PATH}")
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns):,}")

    required = {"fight_id", "fighter_id", POWER}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    dup = df.duplicated(["fight_id", "fighter_id"]).sum()
    print(f"duplicate fighter-fight rows: {dup:,}")
    if dup:
        print(df.loc[df.duplicated(["fight_id", "fighter_id"], keep=False), ["fight_id", "fighter_id"]].head(20).to_string(index=False))

    power = pd.to_numeric(df[POWER], errors="coerce")
    print(f"non-finite striking_power: {(~np.isfinite(power.fillna(np.nan))).sum():,}")
    print_distribution("HISTORICAL PREFIGHT SNAPSHOTS", power)

    latest = latest_per_fighter(df)
    print_distribution("LATEST SNAPSHOT PER FIGHTER", latest[POWER])

    name_col = "fighter_name" if "fighter_name" in latest.columns else None
    cols = [c for c in (name_col, "fighter_id", "date", "fight_id", POWER) if c]

    print("\nLATEST — TOP 30")
    print(latest.sort_values(POWER, ascending=False).head(30)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nLATEST — BOTTOM 30")
    print(latest.sort_values(POWER, ascending=True).head(30)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    near = latest.copy()
    near["distance_to_50"] = (pd.to_numeric(near[POWER], errors="coerce") - 50.0).abs()
    print("\nLATEST — 30 CLOSEST TO 50")
    print(near.nsmallest(30, "distance_to_50")[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if name_col:
        print("\nREFERENCE FIGHTERS — FULL SNAPSHOT HISTORY")
        ref = df[df[name_col].isin(REFERENCE_NAMES)].copy()
        sort_cols = [c for c in (name_col, "date", "fight_id") if c in ref.columns]
        if sort_cols:
            ref = ref.sort_values(sort_cols)
        print(ref[[c for c in (name_col, "date", "fight_id", POWER) if c in ref.columns]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Contract check: once a fighter has a prefight snapshot >50, future stored
    # power may hold or increase, but must never decline because later quiet
    # fights do not erase demonstrated fresh power.
    order_cols = [c for c in ("date", "fight_id") if c in df.columns]
    ordered = df.sort_values(["fighter_id", *order_cols]).copy() if order_cols else df.copy()
    violations = []
    for fighter_id, g in ordered.groupby("fighter_id", sort=False):
        vals = pd.to_numeric(g[POWER], errors="coerce").to_numpy(dtype=float)
        pos = np.flatnonzero(vals > 50.0 + 1e-9)
        if len(pos) == 0:
            continue
        tail = vals[pos[0]:]
        drops = np.flatnonzero(np.diff(tail) < -1e-9)
        if len(drops):
            first_drop = int(drops[0])
            a = g.iloc[pos[0] + first_drop]
            b = g.iloc[pos[0] + first_drop + 1]
            violations.append({
                "fighter_id": fighter_id,
                "fighter_name": a.get(name_col, "") if name_col else "",
                "from_power": float(a[POWER]),
                "to_power": float(b[POWER]),
                "from_date": a.get("date", ""),
                "to_date": b.get("date", ""),
            })

    print("\nNON-DEGRADING DEMONSTRATED-POWER CHECK")
    print(f"fighters with a post->50 decline: {len(violations):,}")
    if violations:
        print(pd.DataFrame(violations).head(50).to_string(index=False))

    print("\nAudit complete. No artifact modified.")


if __name__ == "__main__":
    main()
