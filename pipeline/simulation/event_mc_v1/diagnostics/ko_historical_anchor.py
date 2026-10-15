"""Descriptive historical KO/TKO anchor from the authoritative master."""
import json
import pandas as pd
from .kd_historical_anchor import MASTER_PATH

def historical_ko_anchor(path=MASTER_PATH):
    frame = pd.read_parquet(path)
    method = next((c for c in ("method", "win_by") if c in frame), None)
    if method is None: return {"available": False, "blocker": "method column unavailable"}
    fights = frame.drop_duplicates("fight_id")
    ko = fights[method].astype(str).str.upper().str.contains("KO|TKO", regex=True)
    return {"available": True, "fights": len(fights), "ko_tko_rate": float(ko.mean())}

if __name__ == "__main__": print(json.dumps(historical_ko_anchor(), indent=2, sort_keys=True))
