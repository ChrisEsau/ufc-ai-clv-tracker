"""Run the 42-fight ALL-BUT vs market benchmark using data/market/market_outcomes.parquet."""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import compare_all_but_vs_prefight_market_42 as bench

MARKET = Path("data/market/market_outcomes.parquet")


def _load_market_outcomes() -> pd.DataFrame:
    if not MARKET.exists():
        raise FileNotFoundError(MARKET)

    m = pd.read_parquet(MARKET).copy()
    required = {"fight_id", "market_key", "side", "american_odds", "implied_probability"}
    missing = sorted(required - set(m.columns))
    if missing:
        raise RuntimeError(f"market_outcomes missing columns: {missing}; available={list(m.columns)}")

    m = m.loc[m["market_key"].astype(str).str.lower().eq("moneyline")].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["side"] = m["side"].astype(str).str.lower()
    m = m.loc[m["side"].isin(["red", "blue"])].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["american_odds"] = pd.to_numeric(m["american_odds"], errors="coerce")

    # market_outcomes can contain multiple books/snapshots. Prefer the latest
    # available snapshot deterministically, then collapse to one price per side.
    sort_cols = [c for c in ["commence_time", "snapshot_timestamp", "bookmaker", "provider_selection_id"] if c in m.columns]
    if sort_cols:
        m = m.sort_values(sort_cols)
    m = m.drop_duplicates(["fight_id", "side"], keep="last")

    keep = ["fight_id", "side", "american_odds", "implied_probability"]
    p = m[keep].pivot(index="fight_id", columns="side")
    p.columns = [f"market_{metric}_{side}" for metric, side in p.columns]
    p = p.reset_index().rename(columns={"fight_id": "bout_id"})

    for side in ("red", "blue"):
        for metric in ("implied_probability", "american_odds"):
            col = f"market_{metric}_{side}"
            if col not in p.columns:
                p[col] = np.nan

    raw_sum = p["market_implied_probability_red"] + p["market_implied_probability_blue"]
    p["market_overround"] = raw_sum - 1.0
    p["market_novig_p_red"] = p["market_implied_probability_red"] / raw_sum
    p["market_novig_p_blue"] = p["market_implied_probability_blue"] / raw_sum
    return p


if __name__ == "__main__":
    bench.MARKET = MARKET
    bench._load_market = _load_market_outcomes
    bench.main()
