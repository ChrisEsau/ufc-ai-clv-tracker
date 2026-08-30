#!/usr/bin/env python3
"""Frozen V4 market-offset XGBoost 2026 moneyline walk-forward.

Research-only, stateful runner. Each invocation scores exactly the next unprocessed
2026 UFC card in strict chronological order and appends immutable card outputs.
Architecture and feature selection are copied from the authoritative V4 capacity
workflow; 2026 outcomes are evaluation-only and never used to tune configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from pipeline.features.run_build_rolling_features import build_full_rolling_features

OUT = Path("data/research/prop_mispricing")
PRED_PATH = OUT / "v4_2026_moneyline_walkforward_predictions.csv"
BETS_PATH = OUT / "v4_2026_moneyline_walkforward_bets.csv"
CARD_PATH = OUT / "v4_2026_moneyline_walkforward_card_summary.csv"
JSON_PATH = OUT / "v4_2026_moneyline_walkforward_summary.json"

THRESHOLD = 0.075
FLAT_STAKE = 10.0
SELECTED_ALPHA = 1.0
CFG = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "rounds": 300,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 2,
}


def clip_p(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip_p(p)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30)
    return 1 / (1 + np.exp(-z))


def payout_mult(american_odds: float) -> float:
    return american_odds / 100.0 if american_odds > 0 else 100.0 / abs(american_odds)


def binary_metrics(y, p):
    y = np.asarray(y, int)
    p = clip_p(p)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(np.mean(pred == y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def metric_block(prefix: str, y, pm, px) -> dict:
    out = {f"{prefix}_n": int(len(y))}
    if not len(y):
        for model in ("market", "v4"):
            for metric in ("accuracy", "brier", "log_loss"):
                out[f"{prefix}_{model}_{metric}"] = None
        return out
    mm = binary_metrics(y, pm)
    mx = binary_metrics(y, px)
    for k, v in mm.items():
        out[f"{prefix}_market_{k}"] = v
    for k, v in mx.items():
        out[f"{prefix}_v4_{k}"] = v
    return out


def load_core():
    market = pd.read_parquet("data/market/historical_market_outcomes.parquet").copy()
    market = market[
        (market["bookmaker"] == "legacy_consensus")
        & (market["market_key"] == "moneyline")
        & (market["result_status"] == "graded")
        & market["won"].notna()
    ].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["won"] = market["won"].astype(bool).astype(int)
    market["implied_probability"] = pd.to_numeric(market["implied_probability"], errors="coerce")
    market["american_odds"] = pd.to_numeric(market["american_odds"], errors="coerce")
    market = market.dropna(subset=["date", "implied_probability", "american_odds"]).copy()

    good = market.groupby("fight_id").size()
    good = good[good == 2].index
    ml = market[market["fight_id"].isin(good)].copy()
    ml["market_overround"] = ml.groupby("fight_id")["implied_probability"].transform("sum")
    ml["fair_market_p"] = ml["implied_probability"] / ml["market_overround"]
    red = ml[ml["outcome_side"].astype(str).eq("red")].copy()
    blue = ml[ml["outcome_side"].astype(str).eq("blue")].copy()
    if red["fight_id"].duplicated().any() or blue["fight_id"].duplicated().any():
        raise RuntimeError("Duplicate red/blue moneyline rows remain after strict two-way filter")

    fv = pd.read_parquet("data/features/moneyline_feature_view.parquet").copy()
    deny = ["winner", "result", "target", "label", "finish_round", "match_time_sec", "profit", "odds", "implied", "market", "actual", "post_"]

    def clean_numeric(c):
        return c in fv.columns and pd.api.types.is_numeric_dtype(fv[c]) and not any(x in c.lower() for x in deny)

    diff_cols = sorted([
        c for c in fv.columns
        if clean_numeric(c)
        and c.lower().endswith("_diff")
        and "abs_diff" not in c.lower()
        and "_abs_" not in c.lower()
    ])
    feature_cols = diff_cols + ["market_overround"]
    if len(feature_cols) != 141:
        raise RuntimeError(f"Frozen V4 feature count drift: expected 141, got {len(feature_cols)}")

    df = red.merge(fv[["fight_id"] + diff_cols], on="fight_id", how="inner").sort_values(["date", "fight_id"]).copy()
    if df["fight_id"].duplicated().any():
        raise RuntimeError("moneyline_feature_view produced duplicate fight rows")
    Xraw = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return ml, red, blue, df, Xraw, feature_cols


def cold_start_frame(target_date: pd.Timestamp, fight_ids) -> pd.DataFrame:
    # Locked 2026 V1 scoring rule: build fresh rolling prefight rows, then mark
    # cold if either r_pre_fights/b_pre_fights is missing/nonfinite or < 2.
    master = pd.read_parquet("data/master/ufc_master.parquet")
    roll = build_full_rolling_features(master)
    roll["date"] = pd.to_datetime(roll["date"], errors="coerce").dt.normalize()
    target = roll[(roll["date"] == target_date) & roll["fight_id"].isin(set(fight_ids))].copy()
    if target["fight_id"].duplicated().any():
        raise RuntimeError("Fresh rolling cold-start source has duplicate target fight rows")
    missing = sorted(set(map(str, fight_ids)) - set(target["fight_id"].astype(str)))
    if missing:
        raise RuntimeError(f"Missing fresh-rolling V1 cold-start rows for fight_ids: {missing}")
    rows = []
    for _, r in target.iterrows():
        rf = pd.to_numeric(pd.Series([r.get("r_pre_fights", np.nan)]), errors="coerce").iloc[0]
        bf = pd.to_numeric(pd.Series([r.get("b_pre_fights", np.nan)]), errors="coerce").iloc[0]
        r_ok = bool(np.isfinite(rf))
        b_ok = bool(np.isfinite(bf))
        cold = (not r_ok) or (not b_ok) or min(float(rf), float(bf)) < 2
        cold_names = []
        if (not r_ok) or float(rf) < 2:
            cold_names.append(str(r.get("r_name", "red")))
        if (not b_ok) or float(bf) < 2:
            cold_names.append(str(r.get("b_name", "blue")))
        prior = []
        if r_ok:
            prior.append(f"{r.get('r_name','red')}:{int(rf)}")
        if b_ok:
            prior.append(f"{r.get('b_name','blue')}:{int(bf)}")
        finite_vals = [float(v) for v in (rf, bf) if np.isfinite(v)]
        rows.append({
            "date": target_date,
            "fight_id": r["fight_id"],
            "cold_start": bool(cold),
            "min_prior_fights": min(finite_vals) if finite_vals else np.nan,
            "cold_start_fighters": " | ".join(cold_names),
            "fighter_prior_fights": " | ".join(prior),
            "r_pre_fights": float(rf) if r_ok else np.nan,
            "b_pre_fights": float(bf) if b_ok else np.nan,
        })
    return pd.DataFrame(rows)


def choose_next_card(ml: pd.DataFrame):
    cards = (ml[(ml["date"].dt.year == 2026)][["date", "event_name"]]
             .drop_duplicates().sort_values(["date", "event_name"]).reset_index(drop=True))
    if cards.empty:
        raise RuntimeError("No 2026 graded legacy_consensus moneyline cards found")
    processed = set()
    if CARD_PATH.exists():
        old = pd.read_csv(CARD_PATH)
        if len(old):
            for r in old.itertuples(index=False):
                processed.add((pd.Timestamp(r.date).normalize(), str(r.event_name)))
    for r in cards.itertuples(index=False):
        key = (pd.Timestamp(r.date).normalize(), str(r.event_name))
        if key not in processed:
            return key, cards
    return None, cards


def append_immutable(path: Path, new: pd.DataFrame, unique_cols):
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
        if combined.duplicated(unique_cols).any():
            dups = combined.loc[combined.duplicated(unique_cols, keep=False), unique_cols]
            raise RuntimeError(f"Refusing to alter/duplicate stored walk-forward rows in {path}:\n{dups}")
    else:
        combined = new.copy()
    combined.to_csv(path, index=False)
    return combined


def run_next_card():
    OUT.mkdir(parents=True, exist_ok=True)
    ml, red, blue, df, Xraw, feature_cols = load_core()
    chosen, cards = choose_next_card(ml)
    if chosen is None:
        print("All available 2026 cards are already processed.")
        return
    target_date, event_name = chosen
    print(f"Target next card: {target_date.date()} | {event_name}")

    target_market = ml[(ml["date"] == target_date) & (ml["event_name"] == event_name)].copy()
    counts = target_market.groupby("fight_id").size()
    if not counts.eq(2).all():
        raise RuntimeError("Target card contains a non-two-way moneyline fight")
    target_fights = sorted(target_market["fight_id"].unique())
    target_idx = df.index[(df["date"] == target_date) & (df["fight_id"].isin(target_fights))].to_numpy()
    scored_fights = set(df.loc[target_idx, "fight_id"].astype(str))
    missing_features = sorted(set(map(str, target_fights)) - scored_fights)
    if missing_features:
        raise RuntimeError(f"Target card fights missing frozen V4 features: {missing_features}")
    if len(target_idx) != len(target_fights):
        raise RuntimeError("Target card feature cardinality mismatch")

    train_idx = df.index[df["date"] < target_date].to_numpy()
    if not len(train_idx):
        raise RuntimeError("No pre-event training fights")
    valid_cols = [c for c in feature_cols if Xraw.loc[train_idx, c].notna().any()]
    med = Xraw.loc[train_idx, valid_cols].median(numeric_only=True)
    Xtr = Xraw.loc[train_idx, valid_cols].fillna(med).fillna(0.0)
    Xte = Xraw.loc[target_idx, valid_cols].fillna(med).fillna(0.0)
    ytr = df.loc[train_idx, "won"].astype(int).to_numpy()
    market_logit_train = logit(df.loc[train_idx, "fair_market_p"])
    market_logit_test = logit(df.loc[target_idx, "fair_market_p"])

    params = {k: v for k, v in CFG.items() if k != "rounds"}
    dtr = xgb.DMatrix(Xtr, label=ytr, base_margin=market_logit_train, feature_names=valid_cols)
    dte = xgb.DMatrix(Xte, base_margin=market_logit_test, feature_names=valid_cols)
    booster = xgb.train(params, dtr, num_boost_round=CFG["rounds"], verbose_eval=False)
    raw_margin = booster.predict(dte, output_margin=True)
    correction = raw_margin - market_logit_test
    model_logit = market_logit_test + SELECTED_ALPHA * correction
    p_red = sigmoid(model_logit)
    p_blue = 1.0 - p_red
    if float(np.max(np.abs((p_red + p_blue) - 1.0))) != 0.0:
        raise RuntimeError("V4 red/blue probabilities do not sum exactly to 1")

    pred_red = df.loc[target_idx, ["fight_id", "date", "event_name", "won", "fair_market_p", "market_overround", "outcome_label"]].copy()
    pred_red["raw_tree_correction_logit_red"] = correction
    pred_red["market_logit_red"] = market_logit_test
    pred_red["model_logit_red"] = model_logit
    pred_red["v4_model_p_red"] = p_red

    fm = cold_start_frame(target_date, target_fights)
    pred_red = pred_red.merge(fm, on=["date", "fight_id"], how="left", validate="one_to_one")
    market_card = target_market.merge(
        pred_red[["fight_id", "raw_tree_correction_logit_red", "market_logit_red", "model_logit_red", "v4_model_p_red", "cold_start", "min_prior_fights", "cold_start_fighters", "fighter_prior_fights", "r_pre_fights", "b_pre_fights"]],
        on="fight_id", how="inner", validate="many_to_one"
    )
    market_card["v4_model_p"] = np.where(
        market_card["outcome_side"].astype(str).eq("red"),
        market_card["v4_model_p_red"],
        1.0 - market_card["v4_model_p_red"],
    )
    market_card["edge"] = market_card["v4_model_p"] - market_card["fair_market_p"]
    market_card["bet_eligible"] = (~market_card["cold_start"].astype(bool)) & (market_card["edge"] >= THRESHOLD)
    winners = (market_card[market_card["won"].eq(1)][["fight_id", "outcome_label"]]
               .rename(columns={"outcome_label": "actual_winner"}))
    if winners["fight_id"].duplicated().any() or len(winners) != len(target_fights):
        raise RuntimeError("Target card winner cardinality mismatch")
    market_card = market_card.merge(winners, on="fight_id", how="left", validate="many_to_one")
    market_card["profit_if_bet"] = np.where(
        market_card["won"].eq(1),
        FLAT_STAKE * market_card["american_odds"].map(payout_mult),
        -FLAT_STAKE,
    )

    predictions = market_card.rename(columns={"outcome_label": "fighter", "outcome_side": "side"})[[
        "date", "event_name", "fight_id", "fighter", "side", "american_odds", "fair_market_p", "v4_model_p", "edge",
        "cold_start", "bet_eligible", "actual_winner", "won", "profit_if_bet", "raw_tree_correction_logit_red",
        "market_logit_red", "model_logit_red", "min_prior_fights", "cold_start_fighters", "fighter_prior_fights", "r_pre_fights", "b_pre_fights"
    ]].copy()
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime("%Y-%m-%d")
    predictions["won"] = predictions["won"].astype(int)
    predictions = predictions.sort_values(["fight_id", "side"]).reset_index(drop=True)

    bets = predictions[predictions["bet_eligible"]].copy()
    bets["flat_stake"] = FLAT_STAKE
    bets["flat_pnl"] = bets["profit_if_bet"]
    bets["units_pnl"] = bets["flat_pnl"] / FLAT_STAKE

    cr = pred_red.sort_values("fight_id").copy()
    y = cr["won"].astype(int).to_numpy()
    pm = cr["fair_market_p"].astype(float).to_numpy()
    px = cr["v4_model_p_red"].astype(float).to_numpy()
    market_pick = pm >= 0.5
    model_pick = px >= 0.5
    correct_market = market_pick == y.astype(bool)
    correct_model = model_pick == y.astype(bool)
    same = market_pick == model_pick
    actual_market_p = np.where(y == 1, pm, 1 - pm)
    actual_model_p = np.where(y == 1, px, 1 - px)
    move = actual_model_p - actual_market_p
    cold = cr["cold_start"].astype(bool).to_numpy()

    card = {
        "date": target_date.strftime("%Y-%m-%d"),
        "event_name": event_name,
        "total_fights": int(len(cr)),
        "non_cold_eligible_fights": int((~cold).sum()),
        "cold_start_fights": int(cold.sum()),
        "train_fights": int(len(train_idx)),
        "frozen_feature_count": int(len(feature_cols)),
        "train_valid_feature_count": int(len(valid_cols)),
        "same_predicted_winner_count": int(same.sum()),
        "disagreement_count": int((~same).sum()),
        "market_right_v4_wrong": int((correct_market & ~correct_model).sum()),
        "v4_right_market_wrong": int((correct_model & ~correct_market).sum()),
        "moved_toward_actual_winner": int((move > 0).sum()),
        "moved_away_from_actual_winner": int((move < 0).sum()),
        "unchanged_actual_winner_probability": int((move == 0).sum()),
        "mean_probability_movement_actual_winner": float(move.mean()),
        "median_probability_movement_actual_winner": float(np.median(move)),
        "qualifying_bets": int(len(bets)),
        "bet_wins": int(bets["won"].sum()) if len(bets) else 0,
        "bet_losses": int(len(bets) - bets["won"].sum()) if len(bets) else 0,
        "bet_flat_risk": float(FLAT_STAKE * len(bets)),
        "bet_profit": float(bets["flat_pnl"].sum()) if len(bets) else 0.0,
        "bet_units": float(bets["units_pnl"].sum()) if len(bets) else 0.0,
        "bet_roi": float(bets["flat_pnl"].sum() / (FLAT_STAKE * len(bets))) if len(bets) else None,
        "max_pair_sum_error": float((market_card.groupby("fight_id")["v4_model_p"].sum() - 1.0).abs().max()),
        "selected_alpha": SELECTED_ALPHA,
        "edge_threshold": THRESHOLD,
        "flat_stake": FLAT_STAKE,
    }
    card.update(metric_block("all", y, pm, px))
    card.update(metric_block("noncold", y[~cold], pm[~cold], px[~cold]))
    card.update(metric_block("cold", y[cold], pm[cold], px[cold]))
    card_df = pd.DataFrame([card])

    all_predictions = append_immutable(PRED_PATH, predictions, ["date", "event_name", "fight_id", "fighter"])
    all_bets = append_immutable(BETS_PATH, bets, ["date", "event_name", "fight_id", "fighter"])
    all_cards = append_immutable(CARD_PATH, card_df, ["date", "event_name"])

    wins = int(pd.to_numeric(all_bets.get("won", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(all_bets) else 0
    cumulative = {
        "processed_cards": int(len(all_cards)),
        "last_processed_card": {"date": card["date"], "event_name": event_name},
        "available_2026_cards": int(len(cards)),
        "prediction_rows": int(len(all_predictions)),
        "bet_threshold": THRESHOLD,
        "flat_stake": FLAT_STAKE,
        "bets": int(len(all_bets)),
        "wins": wins,
        "losses": int(len(all_bets) - wins),
        "profit": float(pd.to_numeric(all_bets.get("flat_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(all_bets) else 0.0,
        "units": float(pd.to_numeric(all_bets.get("units_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(all_bets) else 0.0,
    }
    risk = FLAT_STAKE * cumulative["bets"]
    cumulative["roi"] = cumulative["profit"] / risk if risk else None
    summary = {
        "frozen_v4": {
            "selected_candidate": "d1_300_strong",
            "selected_alpha": SELECTED_ALPHA,
            "feature_count": len(feature_cols),
            "xgboost": CFG,
            "market_role": "vig-free RED fair market logit supplied as base_margin; additive tree correction only",
            "cold_start_rule": "locked V1 fresh rolling: either r_pre_fights/b_pre_fights missing/nonfinite or < 2",
            "edge_threshold": THRESHOLD,
            "flat_stake": FLAT_STAKE,
        },
        "latest_card": card,
        "cumulative": cumulative,
    }
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print("\n=== V4 WALK-FORWARD CARD ===")
    print(json.dumps(card, indent=2))
    print("\n=== CARD BETS >= 7.5% / NON-COLD ===")
    if len(bets):
        print(bets[["fighter", "american_odds", "fair_market_p", "v4_model_p", "edge", "won", "flat_pnl", "units_pnl"]].to_string(index=False))
    else:
        print("No qualifying bets.")
    print("\n=== CUMULATIVE BET LEDGER ===")
    print(json.dumps(cumulative, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-card", action="store_true", help="Score exactly the next unprocessed 2026 card")
    args = parser.parse_args()
    if not args.next_card:
        parser.error("This frozen runner requires --next-card")
    run_next_card()


if __name__ == "__main__":
    main()
