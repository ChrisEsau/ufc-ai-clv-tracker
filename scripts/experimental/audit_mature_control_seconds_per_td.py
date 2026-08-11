"""Audit historical control seconds per landed takedown in the mature 2020+ cohort.

Research-only. Uses the exact aligned mature cohort used by the current KO validation
and reads fight-level RFS observations. This is a calibration proxy, not a literal
ground-sequence duration: UFCStats control time can include clinch control as well as
ground control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_locked_families_v1 as locked

CONTROL_PER_TD = "rfs_phase_base_fight_control_seconds_per_td_landed"
CONTROL_SECONDS = "rfs_phase_interact_fight_control_seconds"
TD_LANDED = "rfs_finish_state_fight_takedowns_landed"


def _summary(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "n": int(series.notna().sum()),
        "mean": float(series.mean()),
        "p10": float(q.loc[0.10]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "p95": float(q.loc[0.95]),
        "max": float(series.max()),
    }


def _print_summary(label: str, series: pd.Series) -> None:
    s = _summary(series)
    print(f"\n{label}")
    print(f"n:      {s['n']:,}")
    print(f"mean:   {s['mean']:.1f}s")
    print(f"p10:    {s['p10']:.1f}s")
    print(f"p25:    {s['p25']:.1f}s")
    print(f"median: {s['median']:.1f}s")
    print(f"p75:    {s['p75']:.1f}s")
    print(f"p90:    {s['p90']:.1f}s")
    print(f"p95:    {s['p95']:.1f}s")
    print(f"max:    {s['max']:.1f}s")


def main() -> None:
    cohort, _ = cohort32.build_aligned_cohort()
    bout_ids = set(cohort["bout_id"].astype(str))

    rfs = pd.read_parquet(locked.RFS_PATH).copy()
    rfs["fight_id"] = rfs["fight_id"].astype(str)
    rfs = rfs[rfs["fight_id"].isin(bout_ids)].copy()

    required = [CONTROL_PER_TD, CONTROL_SECONDS, TD_LANDED]
    missing = [c for c in required if c not in rfs.columns]
    if missing:
        raise RuntimeError(f"RFS missing required audit columns: {missing}")

    for c in required:
        rfs[c] = pd.to_numeric(rfs[c], errors="coerce")

    eligible = rfs[(rfs[TD_LANDED] > 0) & rfs[CONTROL_PER_TD].notna()].copy()
    values = eligible[CONTROL_PER_TD].replace([np.inf, -np.inf], np.nan).dropna()

    print("=" * 100)
    print("MATURE 2020+ HISTORICAL CONTROL SECONDS PER LANDED TAKEDOWN")
    print("=" * 100)
    print(f"aligned bouts: {len(bout_ids):,}")
    print(f"fighter-fight rows with >=1 landed TD: {len(eligible):,}")
    print("proxy = total UFCStats control seconds / takedowns landed")
    print("NOTE: control time can include clinch control, so this is not an exact ground-sequence duration.")

    _print_summary("ALL ELIGIBLE FIGHTER-FIGHTS", values)

    for td_count in (1, 2, 3):
        subset = eligible.loc[eligible[TD_LANDED] == td_count, CONTROL_PER_TD].dropna()
        if len(subset):
            _print_summary(f"EXACTLY {td_count} LANDED TD{'S' if td_count != 1 else ''}", subset)

    subset4 = eligible.loc[eligible[TD_LANDED] >= 4, CONTROL_PER_TD].dropna()
    if len(subset4):
        _print_summary("4+ LANDED TDS", subset4)

    print("\nREFERENCE: CURRENT MC GROUND SEQUENCE AUDIT")
    print("Dober-Frevola mean 102.8s | median 80.0s | p75 160.0s | p90 220.0s")
    print("Neutral MC ground-exit prior: 20% per 30s -> ~7.17% per 10s")
    print("Research-only audit; no simulator or FSR values modified.")


if __name__ == "__main__":
    main()
