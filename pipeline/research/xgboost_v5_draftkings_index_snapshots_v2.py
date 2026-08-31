#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import unicodedata
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
V5_SUMMARY = OUT / "xgboost_v5_exact_reproduction_summary.json"

CFG = {
    "max_depth": 1, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
    "min_child_weight": 10, "lambda": 8.0, "alpha": 1.0,
    "objective": "binary:logistic", "eval_metric": "logloss", "seed": 42, "nthread": 2,
}
ROUNDS = 300


def clip_p(x):
    return np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)


def logit(x):
    p = clip_p(x)
    return np.log(p / (1 - p))


def norm_text(x):
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_name(x):
    return norm_text(x)


def token_name(x):
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii").lower()
    toks = re.findall(r"[a-z0-9]+", s)
    return "".join(sorted(toks))


def pair_key(a, b):
    return "||".join(sorted([norm_name(a), norm_name(b)]))


def token_pair_key(a, b):
    return "||".join(sorted([token_name(a), token_name(b)]))


def selected_features():
    obj = json.loads(V5_SUMMARY.read_text())
    fs = list(obj["selected_features"])
    if len(fs) != 50 or any(not x.endswith("_diff") for x in fs):
        raise RuntimeError(f"Frozen V5 feature-list drift: {len(fs)}")
    return fs


def train_v5(fs):
    m = pd.read_parquet(FROZEN_MARKET).copy()
    m = m[(m.bookmaker == "legacy_consensus") & (m.market_key == "moneyline") &
          (m.result_status == "graded") & m.won.notna()].copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m = m.dropna(subset=["date", "implied_probability"])
    n = m.groupby("fight_id").size()
    m = m[m.fight_id.isin(n[n.eq(2)].index)].copy()
    m["market_overround"] = m.groupby("fight_id")["implied_probability"].transform("sum")
    m["fair_market_p"] = m["implied_probability"] / m["market_overround"]
    red = m[m.outcome_side.astype(str).eq("red")].copy()

    fv = pd.read_parquet(FROZEN_FEATURES)
    miss = [c for c in fs if c not in fv.columns]
    if miss:
        raise RuntimeError(f"Frozen V5 features missing: {miss}")
    df = red.merge(fv[["fight_id"] + fs], on="fight_id", how="inner", validate="one_to_one")
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)
    cols = fs + ["market_overround"]
    X0 = df[cols].replace([np.inf, -np.inf], np.nan)
    med = X0.median(numeric_only=True)
    X = X0.fillna(med).fillna(0.0)
    y = df["won"].astype(int).to_numpy()
    base = logit(df["fair_market_p"].to_numpy())
    d = xgb.DMatrix(X, label=y, base_margin=base, feature_names=cols)
    booster = xgb.train(CFG, d, num_boost_round=ROUNDS, verbose_eval=False)
    return booster, med, cols, len(df), str(df.date.max().date())


def build_feature_lookup(fs):
    fv = pd.read_parquet(CURRENT_FEATURES).copy()
    required = ["fight_id", "date", "event_name", "r_name", "b_name"] + fs
    miss = [c for c in required if c not in fv.columns]
    if miss:
        raise RuntimeError(f"Current feature view missing required columns: {miss}")
    fv = fv[required].drop_duplicates("fight_id").copy()
    fv["date"] = pd.to_datetime(fv["date"], errors="coerce").dt.normalize()
    fv["event_norm"] = fv["event_name"].map(norm_text)
    fv["pair_key"] = [pair_key(a, b) for a, b in zip(fv.r_name, fv.b_name)]
    fv["token_pair_key"] = [token_pair_key(a, b) for a, b in zip(fv.r_name, fv.b_name)]
    return fv


def indexed_moneylines(fs):
    idx = pd.read_parquet(INDEX).copy()
    idx["snapshot_timestamp"] = pd.to_datetime(idx["snapshot_timestamp"], errors="coerce", utc=True)
    idx = idx[idx.snapshot_timestamp.notna()].copy()
    meta = idx.groupby("snapshot_run_id", as_index=False).agg(
        index_snapshot_timestamp=("snapshot_timestamp", "max"),
        indexed_payloads=("raw_payload_path", "nunique"),
        indexed_events=("provider_event_id", "nunique"),
    )
    run_ids = set(meta.snapshot_run_id.astype(str))

    h = pd.read_parquet(HISTORY).copy()
    h = h[(h.bookmaker == "DraftKings") & (h.market_key == "moneyline")].copy()
    h["source_run_id"] = h["source_run_id"].astype(str)
    h = h[h.source_run_id.isin(run_ids)].copy()
    if h.empty:
        raise RuntimeError("No DraftKings moneyline history rows correspond to raw-index snapshot_run_ids")
    h = h.merge(meta, left_on="source_run_id", right_on="snapshot_run_id", how="inner", validate="many_to_one")
    h["run_id"] = h["snapshot_run_id"].astype(str)
    h["indexed_pair_id"] = h["fight_id"].astype(str)
    h["fighter_name"] = h["outcome_display"].astype("string")
    h["american_odds"] = pd.to_numeric(h["american_odds"], errors="coerce")
    h["implied_probability"] = pd.to_numeric(h["implied_probability"], errors="coerce")
    h = h.dropna(subset=["indexed_pair_id", "fight_display", "fighter_name", "american_odds", "implied_probability"]).copy()

    # Build the fighter pair from the two moneyline outcomes rather than trusting the synthetic fight id.
    pairs = (h.groupby(["indexed_pair_id", "event_name"], as_index=False)
             .agg(pair_fighters=("fighter_name", lambda s: list(dict.fromkeys(str(x) for x in s if pd.notna(x))))))
    pairs = pairs[pairs.pair_fighters.map(len).eq(2)].copy()
    pairs["pair_key"] = pairs.pair_fighters.map(lambda x: pair_key(x[0], x[1]))
    pairs["token_pair_key"] = pairs.pair_fighters.map(lambda x: token_pair_key(x[0], x[1]))
    pairs["event_norm"] = pairs.event_name.map(norm_text)

    fl = build_feature_lookup(fs)
    exact = pairs.merge(
        fl[["fight_id", "date", "event_name", "event_norm", "r_name", "b_name", "pair_key"]],
        on=["event_norm", "pair_key"], how="left", suffixes=("_market", "_feature")
    )
    exact_good = exact[exact.fight_id.notna()].copy()
    exact_map = exact_good.sort_values("date").drop_duplicates("indexed_pair_id", keep="last")
    mapped_ids = set(exact_map.indexed_pair_id)

    # Token-order-insensitive fallback handles provider naming reversals such as Cong Wang / Wang Cong.
    left = pairs[~pairs.indexed_pair_id.isin(mapped_ids)].copy()
    tok = left.merge(
        fl[["fight_id", "date", "event_name", "event_norm", "r_name", "b_name", "token_pair_key"]],
        on=["event_norm", "token_pair_key"], how="left", suffixes=("_market", "_feature")
    )
    tok_good = tok[tok.fight_id.notna()].copy()
    tok_map = tok_good.sort_values("date").drop_duplicates("indexed_pair_id", keep="last")

    map_cols = ["indexed_pair_id", "fight_id", "date", "r_name", "b_name"]
    mapping = pd.concat([exact_map[map_cols], tok_map[map_cols]], ignore_index=True).drop_duplicates("indexed_pair_id")
    if mapping.empty:
        raise RuntimeError("No indexed DraftKings fighter pairs matched canonical V5 feature fights by event + fighter pair")

    unmatched_pairs = pairs[~pairs.indexed_pair_id.isin(set(mapping.indexed_pair_id))]
    print(f"PAIR_MAPPING matched={len(mapping)} total={len(pairs)} unmatched={len(unmatched_pairs)}")
    if not unmatched_pairs.empty:
        print("UNMATCHED_PAIR_SAMPLE")
        print(unmatched_pairs[["event_name", "pair_fighters"]].head(25).to_string(index=False))

    h = h.merge(mapping, on="indexed_pair_id", how="inner", validate="many_to_one")
    h["fighter_norm"] = h.fighter_name.map(norm_name)
    h["fighter_token"] = h.fighter_name.map(token_name)
    h["red_norm"] = h.r_name.map(norm_name)
    h["blue_norm"] = h.b_name.map(norm_name)
    h["red_token"] = h.r_name.map(token_name)
    h["blue_token"] = h.b_name.map(token_name)
    h["orientation_side"] = np.where(
        h.fighter_norm.eq(h.red_norm) | h.fighter_token.eq(h.red_token), "red",
        np.where(h.fighter_norm.eq(h.blue_norm) | h.fighter_token.eq(h.blue_token), "blue", None),
    )
    h = h[h.orientation_side.isin(["red", "blue"])].copy()

    h = h.sort_values("refresh_timestamp").drop_duplicates(
        ["run_id", "fight_id", "orientation_side"], keep="last"
    )
    counts = h.groupby(["run_id", "fight_id"])["orientation_side"].nunique()
    good = counts[counts.eq(2)].reset_index()[["run_id", "fight_id"]]
    h = h.merge(good, on=["run_id", "fight_id"], how="inner", validate="many_to_one")
    if h.empty:
        raise RuntimeError("No complete two-sided indexed moneyline snapshots remained after canonical pair mapping")

    h["market_overround"] = h.groupby(["run_id", "fight_id"])["implied_probability"].transform("sum")
    h["fair_market_p"] = h["implied_probability"] / h["market_overround"]
    return h, len(pairs), len(mapping), len(unmatched_pairs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = selected_features()
    booster, med, cols, train_n, train_end = train_v5(fs)
    mk, pair_total, pair_mapped, pair_unmatched = indexed_moneylines(fs)

    fv = build_feature_lookup(fs)
    feature_cols = ["fight_id"] + fs
    red = mk[mk.orientation_side.eq("red")].merge(fv[feature_cols], on="fight_id", how="inner", validate="many_to_one")
    if red.empty:
        raise RuntimeError("No complete indexed moneyline snapshot has a current V5 feature row")
    X0 = red[fs].replace([np.inf, -np.inf], np.nan).copy()
    X0["market_overround"] = red["market_overround"].to_numpy()
    X = X0[cols].fillna(med).fillna(0.0)
    base = logit(red.fair_market_p.to_numpy())
    d = xgb.DMatrix(X, base_margin=base, feature_names=cols)
    pred_red = booster.predict(d)
    red["v5_model_p_red"] = pred_red
    red["market_logit_red"] = base
    red["model_logit_red"] = logit(pred_red)
    red["tree_correction_logit_red"] = red.model_logit_red - red.market_logit_red

    keys = ["run_id", "fight_id"]
    scored = mk.merge(red[keys + ["v5_model_p_red", "market_logit_red", "model_logit_red", "tree_correction_logit_red"]],
                      on=keys, how="inner", validate="many_to_one")
    scored["v5_model_p"] = np.where(scored.orientation_side.eq("red"), scored.v5_model_p_red, 1 - scored.v5_model_p_red)
    scored["edge"] = scored.v5_model_p - scored.fair_market_p
    scored["abs_edge"] = scored.edge.abs()

    out_cols = [
        "index_snapshot_timestamp", "run_id", "indexed_pair_id", "fight_id", "date", "event_name", "fighter_name", "orientation_side",
        "american_odds", "implied_probability", "market_overround", "fair_market_p", "v5_model_p", "edge", "abs_edge",
        "tree_correction_logit_red", "market_logit_red", "model_logit_red", "indexed_payloads", "indexed_events"
    ]
    out = scored[out_cols].sort_values(["index_snapshot_timestamp", "fight_id", "orientation_side"]).reset_index(drop=True)
    out.to_csv(PRED, index=False)
    latest = (out.sort_values("index_snapshot_timestamp")
              .drop_duplicates(["fight_id", "fighter_name"], keep="last")
              .sort_values(["date", "fight_id", "orientation_side"]).reset_index(drop=True))
    latest.to_csv(LATEST, index=False)

    s = {
        "model": "frozen_v5_top_50_pre2021_gain",
        "training_source_commit": "7df1b61126be1f4e036b256d1c774c531b8a281f",
        "training_rows": int(train_n), "training_end_date": train_end, "feature_count": len(cols),
        "indexed_snapshot_runs_available": int(pd.read_parquet(INDEX)["snapshot_run_id"].nunique()),
        "indexed_pairs_total": int(pair_total), "indexed_pairs_mapped": int(pair_mapped), "indexed_pairs_unmatched": int(pair_unmatched),
        "indexed_snapshot_runs_scored": int(out.run_id.nunique()),
        "indexed_fights_scored": int(out.fight_id.nunique()),
        "snapshot_fight_instances": int(out[["run_id", "fight_id"]].drop_duplicates().shape[0]),
        "snapshot_side_rows": int(len(out)),
        "fight_date_min": str(pd.to_datetime(out.date).min().date()),
        "fight_date_max": str(pd.to_datetime(out.date).max().date()),
        "snapshot_min": str(out.index_snapshot_timestamp.min()),
        "snapshot_max": str(out.index_snapshot_timestamp.max()),
        "mean_abs_edge": float(out.abs_edge.mean()), "max_abs_edge": float(out.abs_edge.max()),
        "positive_edges_ge_0_05": int((out.edge >= .05).sum()),
        "positive_edges_ge_0_075": int((out.edge >= .075).sum()),
        "positive_edges_ge_0_10": int((out.edge >= .10).sum()),
        "latest_side_rows": int(len(latest)),
        "notes": "Raw index defines eligible DraftKings snapshot runs. Synthetic market fight IDs are mapped to canonical V5 fights by normalized event + exact fighter pair, with token-order-insensitive fallback. Frozen V5 architecture and selected features unchanged."
    }
    SUMMARY.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))
    print("\nTOP POSITIVE EDGES")
    print(out.sort_values("edge", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
