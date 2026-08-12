"""Compatibility launcher for the FSR trajectory directional audit.

The master artifact may expose its fight date as ``date`` rather than
``event_date``.  The older age helper resolves the date column but its corner-age
calculator still expects an ``event_date`` column.  This launcher normalizes the
resolved date into that compatibility alias, patches only the age-loading helper,
and then runs the existing directional audit unchanged.

Research only.  No FSR, age modifier, simulator, or source data is modified.
"""
from __future__ import annotations

import pandas as pd

from scripts.experimental import backtest_fsr_individual_trajectory_by_age29 as age29
from scripts.experimental import audit_fsr_trajectory_directional_prediction as audit


def _load_ages_fixed() -> pd.DataFrame:
    print(f"[trajectory-age29] loading master ages: {age29.MASTER_PATH}", flush=True)
    master = pd.read_parquet(age29.MASTER_PATH).copy()

    date_col = age29.age_study.modern._resolve_date_column(master)
    master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
    master = master.dropna(subset=[date_col]).copy()

    # Compatibility contract for _resolve_corner_age(), which currently
    # hardcodes master['event_date'] even when the source artifact uses 'date'.
    if "event_date" not in master.columns:
        master["event_date"] = master[date_col]
    else:
        master["event_date"] = pd.to_datetime(master["event_date"], errors="coerce")

    master["fight_id"] = master["fight_id"].astype(str)
    master["r_id"] = master["r_id"].astype(str)
    master["b_id"] = master["b_id"].astype(str)
    master["r_age_calc"] = age29.age_study._resolve_corner_age(master, "r")
    master["b_age_calc"] = age29.age_study._resolve_corner_age(master, "b")

    red = master[["fight_id", "r_id", "r_age_calc"]].rename(
        columns={"r_id": "fighter_id", "r_age_calc": "target_age"}
    )
    blue = master[["fight_id", "b_id", "b_age_calc"]].rename(
        columns={"b_id": "fighter_id", "b_age_calc": "target_age"}
    )
    ages = pd.concat([red, blue], ignore_index=True)
    ages["fighter_id"] = ages["fighter_id"].astype(str)
    ages["fight_id"] = ages["fight_id"].astype(str)
    ages["target_age"] = pd.to_numeric(ages["target_age"], errors="coerce")
    ages = ages.dropna(subset=["target_age"]).drop_duplicates(
        ["fight_id", "fighter_id"], keep="last"
    )
    print(
        f"[trajectory-age29] age rows={len(ages):,} | "
        f"fighter-fights={ages[['fight_id','fighter_id']].drop_duplicates().shape[0]:,}",
        flush=True,
    )
    return ages


def main() -> None:
    age29._load_ages = _load_ages_fixed
    # audit imported the same module object, so this patch is visible there too.
    audit.main()


if __name__ == "__main__":
    main()
