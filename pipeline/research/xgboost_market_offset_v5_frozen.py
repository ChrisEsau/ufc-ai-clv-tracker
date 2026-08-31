from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

V5_SNAPSHOT_SHA = "7df1b61126be1f4e036b256d1c774c531b8a281f"
EXPECTED_OOF_LOG_LOSS = 0.600822510744624
FOLDS = [
    ("2021", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
]
PARAMS = {
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
DENY_TOKENS = [
    "winner", "result", "target", "label", "finish_round", "match_time_sec",
    "profit", "odds", "implied", "market", "actual", "post_",
]


def _clip_p(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def _logit(p):
    p = _clip_p(p)
    return np.log(p / (1 - p))


def _sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30)
    return 1 / (1 + np.exp(-z))


def binary_log_loss(y, p):
    y = np.asarray(y, int)
    p = _clip_p(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def build_v5_frame(market_path: str | Path, feature_path: str | Path):
    market = pd.read_parquet(market_path).copy()
    market = market[(market["bookmaker"] == "legacy_consensus") & (market["result_status"] == "graded") & market["won"].notna()].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market["won"] = market["won"].astype(bool).astype(int)
    market["implied_probability"] = pd.to_numeric(market["implied_probability"], errors="coerce")
    market["profit_per_100"] = pd.to_numeric(market["profit_per_100"], errors="coerce")
    market = market.dropna(subset=["date", "implied_probability", "profit_per_100"]).copy()

    ml = market[market["market_key"] == "moneyline"].copy()
    good = ml.groupby("fight_id").size()
    good = good[good == 2].index
    ml = ml[ml["fight_id"].isin(good)].copy()
    ml["market_overround"] = ml.groupby("fight_id")["implied_probability"].transform("sum")
    ml["fair_market_p"] = ml["implied_probability"] / ml["market_overround"]
    red = ml[ml["outcome_side"].astype(str).eq("red")].copy()

    fv = pd.read_parquet(feature_path).copy()
    def clean_numeric(c):
        if c not in fv.columns or not pd.api.types.is_numeric_dtype(fv[c]):
            return False
        return not any(x in c.lower() for x in DENY_TOKENS)
    signed = sorted([
        c for c in fv.columns
        if clean_numeric(c)
        and c.lower().endswith("_diff")
        and "abs_diff" not in c.lower()
        and "_abs_" not in c.lower()
    ])
    df = red.merge(fv[["fight_id"] + signed], on="fight_id", how="inner").sort_values(["date", "fight_id"]).copy()
    xraw = df[signed + ["market_overround"]].replace([np.inf, -np.inf], np.nan)
    return df, xraw, signed


def frozen_feature_order(df, xraw, signed):
    rank_train = df["date"] <= "2020-12-31"
    rank_cols = [c for c in signed + ["market_overround"] if xraw.loc[rank_train, c].notna().any()]
    rank_med = xraw.loc[rank_train, rank_cols].median(numeric_only=True)
    xrank = xraw.loc[rank_train, rank_cols].fillna(rank_med).fillna(0.0)
    yrank = df.loc[rank_train, "won"].astype(int).to_numpy()
    mrank = _logit(df.loc[rank_train, "fair_market_p"])
    drank = xgb.DMatrix(xrank, label=yrank, base_margin=mrank, feature_names=rank_cols)
    model = xgb.train(PARAMS, drank, num_boost_round=ROUNDS, verbose_eval=False)
    gain = model.get_score(importance_type="gain")
    ranked_signed = sorted(signed, key=lambda c: gain.get(c, 0.0), reverse=True)
    # Frozen V5 selected top-50 pre-2021 gain + market_overround.
    return ranked_signed[:50] + ["market_overround"]


def generate_oof(market_path: str | Path, feature_path: str | Path):
    df, xraw, signed = build_v5_frame(market_path, feature_path)
    feature_cols = frozen_feature_order(df, xraw, signed)
    parts = []
    for fold, train_end, val_start, val_end in FOLDS:
        tr = df["date"] <= train_end
        va = (df["date"] >= val_start) & (df["date"] <= val_end)
        valid = [c for c in feature_cols if xraw.loc[tr, c].notna().any()]
        med = xraw.loc[tr, valid].median(numeric_only=True)
        xtr = xraw.loc[tr, valid].fillna(med).fillna(0.0)
        xva = xraw.loc[va, valid].fillna(med).fillna(0.0)
        ytr = df.loc[tr, "won"].astype(int).to_numpy()
        yva = df.loc[va, "won"].astype(int).to_numpy()
        mtr = _logit(df.loc[tr, "fair_market_p"])
        mva = _logit(df.loc[va, "fair_market_p"])
        dtr = xgb.DMatrix(xtr, label=ytr, base_margin=mtr, feature_names=valid)
        dva = xgb.DMatrix(xva, label=yva, base_margin=mva, feature_names=valid)
        model = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
        full_margin = model.predict(dva, output_margin=True)
        p = _sigmoid(mva + (full_margin - mva))
        parts.append(pd.DataFrame({
            "fight_id": df.loc[va, "fight_id"].astype(str).to_numpy(),
            "date": df.loc[va, "date"].to_numpy(),
            "fold": fold,
            "won_red": yva,
            "market_p_red": _sigmoid(mva),
            "model_p_red": p,
        }))
    out = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    ll = binary_log_loss(out["won_red"], out["model_p_red"])
    return out, feature_cols, ll
