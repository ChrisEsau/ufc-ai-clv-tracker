"""Run the 42-fight ALL-BUT vs market benchmark using data/market/market_outcomes.parquet."""
from pathlib import Path

from scripts.experimental import compare_all_but_vs_prefight_market_42 as bench


if __name__ == "__main__":
    bench.MARKET = Path("data/market/market_outcomes.parquet")
    try:
        bench.main()
    except RuntimeError as exc:
        # If this canonical market parquet uses a different schema, make the
        # next repair trivial by printing the exact available columns.
        if bench.MARKET.exists():
            import pandas as pd

            cols = list(pd.read_parquet(bench.MARKET).columns)
            print(f"market source: {bench.MARKET}")
            print(f"market columns: {cols}")
        raise exc
