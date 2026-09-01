from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research/prop_mispricing"
OOF = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"
HOLDOUT = OUT / "xgboost_method_hierarchical_v5_holdout_2025_2026_predictions.csv"
MARKET = ROOT / "data/market/historical_market_outcomes.parquet"
LEDGER = OUT / "hierarchical_v5_top_method_2021_2025_ledger.csv"
SUMMARY_CSV = OUT / "hierarchical_v5_top_method_2021_2025_summary.csv"
SUMMARY_JSON = OUT / "hierarchical_v5_top_method_2021_2025_summary.json"

METHODS = ["ko", "sub", "dec"]
CLASS_IDX = {
    "red_ko": 0,
    "red_sub": 1,
    "red_dec": 2,
    "blue_ko": 3,
    "blue_sub": 4,
    "blue_dec": 5,
}
MARKET_KEY = {
    "ko": "win_by_ko_tko_dq",
    "sub": "win_by_submission",
    "dec": "win_by_decision",
}


def summarize(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {
            "period": label,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": np.nan,
            "stake_units": 0.0,
            "profit_units": 0.0,
            "roi": np.nan,
            "avg_decimal_odds": np.nan,
        }
    stake = float(frame["stake_units"].sum())
    profit = float(frame["profit_units"].sum())
    return {
        "period": label,
        "bets": int(len(frame)),
        "wins": int(frame["won"].sum()),
        "losses": int((1 - frame["won"]).sum()),
        "hit_rate": float(frame["won"].mean()),
        "stake_units": stake,
        "profit_units": profit,
        "roi": float(profit / stake) if stake else np.nan,
        "avg_decimal_odds": float(frame["decimal_odds"].mean()),
    }


def load_predictions() -> pd.DataFrame:
    oof = pd.read_csv(OOF)
    oof["date"] = pd.to_datetime(oof["date"])
    oof = oof[(oof["date"] >= "2021-01-01") & (oof["date"] <= "2024-12-31")].copy()
    oof["evaluation_set"] = "2021-2024_OOF"

    holdout = pd.read_csv(HOLDOUT)
    holdout["date"] = pd.to_datetime(holdout["date"])
    holdout = holdout[(holdout["date"] >= "2025-01-01") & (holdout["date"] <= "2025-12-31")].copy()
    holdout["evaluation_set"] = "2025_holdout"

    common = [
        "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
        "v5_model_p_red",
        "hier_red_ko", "hier_red_sub", "hier_red_dec",
        "hier_blue_ko", "hier_blue_sub", "hier_blue_dec",
        "evaluation_set",
    ]
    out = pd.concat([oof[common], holdout[common]], ignore_index=True)
    out["fight_id"] = out["fight_id"].astype(str)
    return out.sort_values(["date", "fight_id"]).reset_index(drop=True)


def build_price_map() -> dict:
    m = pd.read_parquet(MARKET).copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m = m[(m["bookmaker"] == "legacy_consensus") & m["outcome_side"].astype(str).isin(["red", "blue"])].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m = m[np.isfinite(m["implied_probability"]) & (m["implied_probability"] > 0) & (m["implied_probability"] < 1)].copy()

    price_map = {}
    for side in ["red", "blue"]:
        for method in METHODS:
            z = m[(m["outcome_side"].astype(str) == side) & (m["market_key"] == MARKET_KEY[method])].copy()
            counts = z.groupby("fight_id").size()
            unique_fights = counts[counts == 1].index
            z = z[z["fight_id"].isin(unique_fights)]
            for r in z[["fight_id", "implied_probability", "american_odds"]].itertuples(index=False):
                price_map[(str(r.fight_id), f"{side}_{method}")] = (
                    float(r.implied_probability),
                    float(r.american_odds) if pd.notna(r.american_odds) else np.nan,
                )
    return price_map


def main() -> None:
    pred = load_predictions()
    price_map = build_price_map()

    rows = []
    skipped_no_price = {"2021-2024_OOF": 0, "2025_holdout": 0}
    available_fights = {"2021-2024_OOF": 0, "2025_holdout": 0}

    for r in pred.itertuples(index=False):
        side = "red" if float(r.v5_model_p_red) >= 0.5 else "blue"
        selected_method = max(METHODS, key=lambda method: float(getattr(r, f"hier_{side}_{method}")))
        slug = f"{side}_{selected_method}"
        price = price_map.get((str(r.fight_id), slug))
        if price is None:
            skipped_no_price[r.evaluation_set] += 1
            continue

        raw_imp, american_odds = price
        decimal_odds = 1.0 / raw_imp
        class_idx = CLASS_IDX[slug]
        won = int(int(r.target) == class_idx)
        profit_units = (decimal_odds - 1.0) if won else -1.0
        available_fights[r.evaluation_set] += 1

        rows.append({
            "fight_id": str(r.fight_id),
            "date": pd.Timestamp(r.date).date().isoformat(),
            "year": int(pd.Timestamp(r.date).year),
            "evaluation_set": r.evaluation_set,
            "event_name": r.event_name,
            "red_fighter": r.red_fighter,
            "blue_fighter": r.blue_fighter,
            "v5_model_p_red": float(r.v5_model_p_red),
            "predicted_winner_side": side,
            "predicted_winner": r.red_fighter if side == "red" else r.blue_fighter,
            "selected_method": selected_method.upper(),
            "selected_method_model_probability": float(getattr(r, f"hier_{side}_{selected_method}")),
            "selected_slug": slug,
            "raw_implied_probability": raw_imp,
            "american_odds": american_odds,
            "decimal_odds": decimal_odds,
            "target_class": int(r.target),
            "won": won,
            "stake_units": 1.0,
            "profit_units": profit_units,
        })

    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger = ledger.sort_values(["date", "fight_id"]).reset_index(drop=True)
    ledger.to_csv(LEDGER, index=False)

    summary_rows = []
    oof_ledger = ledger[ledger["evaluation_set"] == "2021-2024_OOF"]
    val_ledger = ledger[ledger["evaluation_set"] == "2025_holdout"]
    summary_rows.append(summarize(oof_ledger, "2021-2024 OOF"))
    for year in [2021, 2022, 2023, 2024]:
        summary_rows.append(summarize(oof_ledger[oof_ledger["year"] == year], str(year)))
    summary_rows.append(summarize(val_ledger, "2025 holdout"))
    summary_rows.append(summarize(ledger, "2021-2025 combined"))

    for evaluation_set, label in [("2021-2024_OOF", "OOF"), ("2025_holdout", "2025")]:
        g = ledger[ledger["evaluation_set"] == evaluation_set]
        for method in ["KO", "SUB", "DEC"]:
            summary_rows.append(summarize(g[g["selected_method"] == method], f"{label} - {method}"))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    payload = {
        "experiment": "hierarchical_v5_ml_winner_top_method_roi_v1",
        "rule": {
            "winner": "fighter with higher frozen V5 moneyline model probability",
            "method": "single highest hierarchical V5 method probability among KO/SUB/DEC for that predicted winner",
            "bets_per_priced_fight": 1,
            "stake_units": 1.0,
            "price_source": "legacy_consensus historical exact-method implied probability",
            "payout": "decimal odds = 1 / raw implied probability",
            "missing_selected_method_price": "skip fight; do not substitute second-best method",
            "roi_used_for_model_selection": False,
        },
        "prediction_sources": {
            "development": OOF.name,
            "validation": HOLDOUT.name,
            "validation_filter": "2025 only; 2026 rows are not scored",
        },
        "coverage": {
            "prediction_rows": {k: int(v) for k, v in pred.groupby("evaluation_set").size().to_dict().items()},
            "priced_bets": available_fights,
            "skipped_no_selected_method_price": skipped_no_price,
        },
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2))

    print(summary.to_string(index=False))
    print(json.dumps(payload["coverage"], indent=2))


if __name__ == "__main__":
    main()
