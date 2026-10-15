"""Non-leaky descriptive historical KD anchor from completed master rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MASTER_PATH = Path("data/master/ufc_master.parquet")


def historical_kd_anchor(path: Path = MASTER_PATH):
    frame = pd.read_parquet(path)
    required = {"fight_id", "r_kd", "b_kd"}
    missing = required - set(frame.columns)
    if missing:
        return {"available": False, "blocker": f"missing columns: {sorted(missing)}"}
    by_fight = frame.assign(
        r_kd=pd.to_numeric(frame.r_kd, errors="coerce").fillna(0),
        b_kd=pd.to_numeric(frame.b_kd, errors="coerce").fillna(0),
    ).groupby("fight_id", as_index=False)[["r_kd", "b_kd"]].sum()
    totals = by_fight.r_kd + by_fight.b_kd
    return {
        "available": True,
        "fights": len(by_fight),
        "knockdowns_per_fight": float(totals.mean()),
        "zero_kd_fight_rate": float((totals == 0).mean()),
        "at_least_one_kd_fight_rate": float((totals >= 1).mean()),
        "multi_kd_fight_rate": float((totals >= 2).mean()),
    }


if __name__ == "__main__":
    print(json.dumps(historical_kd_anchor(), indent=2, sort_keys=True))
