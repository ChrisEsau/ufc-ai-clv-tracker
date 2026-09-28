"""Descriptive completed-fight decision-rate anchor."""

import json

import pandas as pd

from pipeline.common.paths import MASTER_PATH


def historical_decision_anchor(path=MASTER_PATH):
    fights = pd.read_parquet(path).drop_duplicates("fight_id")
    fights = fights[fights["winner"].notna()]
    decision = fights["method"].astype(str).str.upper().str.startswith("DECISION")
    return {"completed_fights": len(fights), "decision_rate": float(decision.mean()), "decision_finishes": int(decision.sum())}


if __name__ == "__main__":
    print(json.dumps(historical_decision_anchor(), indent=2, sort_keys=True))
