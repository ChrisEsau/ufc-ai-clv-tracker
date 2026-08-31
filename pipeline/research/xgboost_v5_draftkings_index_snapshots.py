#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OUT = Path("data/research/prop_mispricing")
PRED = OUT / "v5_draftkings_index_snapshot_predictions.csv"
LATEST = OUT / "v5_draftkings_index_latest_by_fight.csv"
SUMMARY = OUT / "v5_draftkings_index_summary.json"

FROZEN_MARKET = Path("/tmp/v5_frozen/historical_market_outcomes.parquet")
FROZEN_FEATURES = Path("/tmp/v5_frozen/moneyline_feature_view.parquet")
CURRENT_FEATURES = Path("data/features/moneyline_feature_view.parquet")
INDEX = Path("data/market/draftkings_raw_index.parquet")
HISTORY = Path("data/market/market_intelligence_history.parquet")
MASTER = Path("data/master/ufc_master.parquet")
V5_SUMMARY = OUT / "xgboost_v5_exact_reproduction_summary.json"

CFG = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 2,
}
ROUNDS = 300


def clip_p(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)


def logit(x):
    p = clip_p(x)
    return np.log(p / (1 - p))


def norm_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def load_selected_features():
    s = json.loads(V5_SUMMARY.read_text())
    selected = list(s["selected_features"])
    if len(selected) != 50:
        raise RuntimeError(f"Expected 50 frozen signed features, got {len(selected)}")
    if any(not c.endswith("_diff") for c in selected):
        raise RuntimeError("Frozen V5 feature list contains a non-diff feature")
    return selected


def build_frozen_training(selected):
    market = pd.read_parquet(FROZEN_MARKET).copy()
    market = market[
        (market["bookmaker"] == "legacy_consensus")
        & (market["market_key"] == "moneyline")
        & (market["result_status"] == "graded")
        & market["won"].notna()
    ].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["implied_probability"] = pd.to_numeric(market["implied_probability"], errors="coerce")
    market = market.dropna(subset=["date", "implied_probability"]).copy()
    good = market.groupby("fight_id").size()
    good = good[good.eq(2)].index
    market = market[market["fight_id"].isin(good)].copy()
    market["market_overround"] = market.groupby("fight_id")["implied_probability"].transform("sum")
    market["fair_market_p"] = market["implied_probability"] / market["market_overround"]
    red = market[market["outcome_side"].astype(str).eq("red")].copy()

    fv = pd.read_parquet(FROZEN_FEATURES).copy()
    missing = [c for c in selected if c not in fv.columns]
    if missing:
        raise RuntimeError(f"Frozen training feature columns missing: {missing}")

    df = red.merge(fv[["fight_id"] + selected], on="fight_id", how="inner", validate="one_to_one")
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)
    Xraw = df[selected + ["market_overround"]].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in selected + ["market_overround"] if Xraw[c].notna().any()]
    if len(valid) != 51:
        raise RuntimeError(f"Expected 51 valid V5 features, got {len(valid)}")
    med = Xraw[valid].median(numeric_only=True)
    X = Xraw[valid].fillna(med).fillna(0.0)
    y = red.set_index("fight_id").loc[df["fight_id"], "won"].astype(int).to_numpy()
    base = logit(df["fair_market_p"].to_numpy())
    dtrain = xgb.DMatrix(X, label=y, base_margin=base, feature_names=valid)
    booster = xgb.train(CFG, dtrain, num_boost_round=ROUNDS, verbose_eval=False)
    return booster, med, valid, len(df), str(df["date"].max().date())


def build_snapshot_market():
    idx = pd.read_parquet(INDEX).copy()
    idx["snapshot_timestamp"] = pd.to_datetime(idx["snapshot_timestamp"], errors="coerce", utc=True)
    idx = idx[idx["snapshot_timestamp"].notna()].copy()
    run_meta = (idx.groupby("snapshot_run_id", as_index=False)
                .agg(index_snapshot_timestamp=("snapshot_timestamp", "max"),
                     indexed_payloads=("raw_payload_path", "nunique"),
                     indexed_events=("provider_event_id", "nunique")))

    hist = pd.read_parquet(HISTORY).copy()
    hist = hist[(hist["bookmaker"] == "DraftKings") & (hist["market_key"] == "moneyline")].copy()
    hist = hist[hist["source_run_id"].isin(set(run_meta["snapshot_run_id"]))].copy()
    if hist.empty:
        raise RuntimeError("No DraftKings moneyline rows in market_intelligence_history match draftkings_raw_index snapshot_run_ids")

    hist = hist.merge(run_meta, left_on="source_run_id", right_on="snapshot_run_id", how="inner", validate="many_to_one")
    hist["american_odds"] = pd.to_numeric(hist["american_odds"], errors="coerce")
    hist["implied_probability"] = pd.to_numeric(hist["implied_probability"], errors="coerce")
    hist = hist.dropna(subset=["fight_id", "fighter_name", "implied_probability", "american_odds"]).copy()

    # Collapse duplicate refresh copies of the same indexed snapshot/selection.
    hist = hist.sort_values("refresh_timestamp").drop_duplicates(
        ["source_run_id", "fight_id", "comparison_key"], keep="last"
    )

    master = pd.read_parquet(MASTER).copy()
    keep = [c for c in ["fight_id", "date", "event_name", "r_name", "b_name", "winner"] if c in master.columns]
    m = master[keep].drop_duplicates("fight_id").copy()
    if "date" in m:
        m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    hist = hist.merge(m, on="fight_id", how="left", suffixes=("", "_master"), validate="many_to_one")

    def orient(row):
        f = norm_name(row["fighter_name"])
        r = norm_name(row.get("r_name", ""))
        b = norm_name(row.get("b_name", ""))
        if f == r:
            return "red"
        if f == b:
            return "blue"
        return None

    hist["orientation_side"] = hist.apply(orient, axis=1)
    hist = hist[hist["orientation_side"].isin(["red", "blue"])].copy()

    group_cols = ["source_run_id", "fight_id"]
    counts = hist.groupby(group_cols)["orientation_side"].agg(lambda x: set(x))
    good = counts[counts.map(lambda s: s == {"red", "blue"})].index
    good_set = set(good)
    hist = hist[hist.apply(lambda r: (r["source_run_id"], r["fight_id"]) in good_set, axis=1)].copy()
    hist = hist.drop_duplicates(group_cols + ["orientation_side"], keep="last")

    hist["market_overround"] = hist.groupby(group_cols)["implied_probability"].transform("sum")
    hist["fair_market_p"] = hist["implied_probability"] / hist["market_overround"]
    return hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = load_selected_features()
    booster, med, valid, train_n, train_end = build_frozen_training(selected)
    market = build_snapshot_market()

    current = pd.read_parquet(CURRENT_FEATURES).copy()
    missing = [c for c in selected if c not in current.columns]
    if missing:
        raise RuntimeError(f"Current feature view missing frozen V5 columns: {missing}")
    current = current[["fight_id"] + selected].drop_duplicates("fight_id")

    red = market[market["orientation_side"].eq("red")].copy()
    red = red.merge(current, on="fight_id", how="inner", validate="many_to_one")
    if red.empty:
        raise RuntimeError("No indexed DraftKings moneyline snapshots matched current V5 feature rows")

    Xraw = red[selected].replace([np.inf, -np.inf], np.nan).copy()
    Xraw["market_overround"] = red["market_overround"].to_numpy()
    X = Xraw[valid].fillna(med).fillna(0.0)
    base = logit(red["fair_market_p"].to_numpy())
    dtest = xgb.DMatrix(X, base_margin=base, feature_names=valid)
    p_red = booster.predict(dtest)
    red["v5_model_p_red"] = p_red
    red["market_logit_red"] = base
    red["model_logit_red"] = logit(p_red)
    red["tree_correction_logit_red"] = red["model_logit_red"] - red["market_logit_red"]

    scored = market.merge(
        red[["source_run_id", "fight_id", "v5_model_p_red", "market_logit_red", "model_logit_red", "tree_correction_logit_red"]],
        on=["source_run_id", "fight_id"], how="inner", validate="many_to_one"
    )
    scored["v5_model_p"] = np.where(scored["orientation_side"].eq("red"), scored["v5_model_p_red"], 1.0 - scored["v5_model_p_red"])
    scored["edge"] = scored["v5_model_p"] - scored["fair_market_p"]
    scored["abs_edge"] = scored["edge"].abs()

    cols = [
        "index_snapshot_timestamp", "source_run_id", "fight_id", "date", "event_name", "fighter_name", "orientation_side",
        "american_odds", "implied_probability", "market_overround", "fair_market_p", "v5_model_p", "edge", "abs_edge",
        "tree_correction_logit_red", "market_logit_red", "model_logit_red", "indexed_payloads", "indexed_events"
    ]
    cols = [c for c in cols if c in scored.columns]
    out = scored[cols].sort_values(["index_snapshot_timestamp", "fight_id", "orientation_side"]).reset_index(drop=True)
    out.to_csv(PRED, index=False)

    latest = (out.sort_values("index_snapshot_timestamp")
              .drop_duplicates(["fight_id", "fighter_name"], keep="last")
              .sort_values(["date", "fight_id", "orientation_side"]).reset_index(drop=True))
    latest.to_csv(LATEST, index=False)

    summary = {
        "model": "frozen_v5_top_50_pre2021_gain",
        "training_source_commit": "7df1b61126be1f4e036b256d1c774c531b8a281f",
        "training_rows": train_n,
        "training_end_date": train_end,
        "feature_count": len(valid),
        "indexed_snapshot_runs": int(out["source_run_id"].nunique()),
        "indexed_fights_scored": int(out["fight_id"].nunique()),
        "snapshot_side_rows": int(len(out)),
        "snapshot_fight_instances": int(out[["source_run_id", "fight_id"]].drop_duplicates().shape[0]),
        "snapshot_min": str(out["index_snapshot_timestamp"].min()),
        "snapshot_max": str(out["index_snapshot_timestamp"].max()),
        "mean_abs_edge": float(out["abs_edge"].mean()),
        "max_abs_edge": float(out["abs_edge"].max()),
        "edges_ge_0_05": int((out["edge"] >= 0.05).sum()),
        "edges_ge_0_075": int((out["edge"] >= 0.075).sum()),
        "edges_ge_0_10": int((out["edge"] >= 0.10).sum()),
        "latest_rows": int(len(latest)),
        "notes": "DraftKings raw index is authoritative for included snapshot_run_ids; normalized moneyline rows are joined from market_intelligence_history. No V5 architecture or feature selection changes.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nTOP POSITIVE EDGES")
    print(out.sort_values("edge", ascending=False).head(25).to_string(index=False))


if __name__ == "__main__":
    main()
