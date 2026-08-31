from pathlib import Path
import pandas as pd

ROOT = Path("data/research/prop_mispricing")
PRED = ROOT / "xgboost_method_market_offset__2025_2026_walkforward_predictions.csv"
BETS = ROOT / "xgboost_method_market_offset__2025_2026_roi_bet_ledger.csv"
OUT = ROOT / "xgboost_method_market_offset__2025_2026_favorite_breakdown.csv"


def roi(g):
    n = len(g)
    p = float(g["profit_units"].sum()) if n else 0.0
    return {
        "bets": n,
        "wins": int(g["won"].sum()) if n else 0,
        "hit_rate": float(g["won"].mean()) if n else None,
        "profit_units": p,
        "roi": p / n if n else None,
        "avg_model_prob": float(g["model_prob"].mean()) if n else None,
        "avg_decimal_odds": float(g["decimal_odds"].mean()) if n else None,
        "avg_logit_residual": float(g["signed_logit_residual"].mean()) if n else None,
    }


def main():
    p = pd.read_csv(PRED)
    b = pd.read_csv(BETS)
    p["red_win_fair"] = p[["market_red_ko", "market_red_sub", "market_red_dec"]].sum(axis=1)
    p["blue_win_fair"] = p[["market_blue_ko", "market_blue_sub", "market_blue_dec"]].sum(axis=1)
    p["fair_favorite_side"] = p.apply(lambda r: "red" if r.red_win_fair >= r.blue_win_fair else "blue", axis=1)
    x = b.merge(p[["fight_id", "red_win_fair", "blue_win_fair", "fair_favorite_side"]], on="fight_id", how="left", validate="one_to_one")
    x["bet_side"] = x["class_slug"].str.split("_").str[0]
    x["method_family"] = x["class_name"].str.replace("RED_", "", regex=False).str.replace("BLUE_", "", regex=False)
    x["favorite_status"] = x.apply(lambda r: "FAVORITE" if r.bet_side == r.fair_favorite_side else "UNDERDOG", axis=1)

    rows = []
    for fav, g in x.groupby("favorite_status"):
        rows.append({"group": fav, "method_family": "ALL", **roi(g)})
    for (fav, method), g in x.groupby(["favorite_status", "method_family"]):
        rows.append({"group": fav, "method_family": method, **roi(g)})
    for method, g in x.groupby("method_family"):
        rows.append({"group": "ALL", "method_family": method, **roi(g)})

    out = pd.DataFrame(rows).sort_values(["group", "method_family"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("\nCOUNT CHECK", len(x))


if __name__ == "__main__":
    main()
