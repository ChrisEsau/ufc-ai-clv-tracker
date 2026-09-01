from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research import xgboost_market_offset_v5_frozen as v5

OUT = Path("data/research/prop_mispricing")
PREDICTIONS = OUT / "xgboost_ko_conditional_ml_stack_oof_predictions.csv"
CANDIDATES = OUT / "xgboost_ko_conditional_ml_stack_edge_candidates.csv"
METRICS = OUT / "xgboost_ko_conditional_ml_stack_metrics.csv"
RULES = OUT / "xgboost_ko_conditional_ml_stack_edge_rules.csv"
ROBUSTNESS = OUT / "xgboost_ko_conditional_ml_stack_edge_robustness.csv"
FEATURES = OUT / "xgboost_ko_conditional_ml_stack_features.json"
SUMMARY = OUT / "xgboost_ko_conditional_ml_stack_summary.json"

PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 2,
}
ROUNDS = 300
TOP_K = 50
EPS = 1e-9
MIN_HISTORICAL_ML_TRAIN_ROWS = 250


def clip_p(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def logit(x):
    p = clip_p(x)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def binary_metrics(y, p):
    y = np.asarray(y, int)
    p = clip_p(p)
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "actual_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "calibration_error": float(np.mean(p) - np.mean(y)),
    }


def multiclass3_metrics(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    p = np.clip(p, EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    oh = np.eye(3)[y]
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "brier": float(np.mean(np.sum((p - oh) ** 2, axis=1))),
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def _v5_fit_predict(ml_df, xraw, feature_cols, train_mask, val_mask):
    valid = [c for c in feature_cols if xraw.loc[train_mask, c].notna().any()]
    med = xraw.loc[train_mask, valid].median(numeric_only=True)
    xtr = xraw.loc[train_mask, valid].fillna(med).fillna(0.0)
    xva = xraw.loc[val_mask, valid].fillna(med).fillna(0.0)
    ytr = ml_df.loc[train_mask, "won"].astype(int).to_numpy()
    mtr = logit(ml_df.loc[train_mask, "fair_market_p"])
    mva = logit(ml_df.loc[val_mask, "fair_market_p"])
    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=mtr, feature_names=valid)
    dva = xgb.DMatrix(xva, base_margin=mva, feature_names=valid)
    booster = xgb.train(v5.PARAMS, dtr, num_boost_round=v5.ROUNDS, verbose_eval=False)
    margin = booster.predict(dva, output_margin=True)
    return sigmoid(margin)


def build_honest_v5_stack():
    ml_df, xraw, signed = v5.build_v5_frame(method.MARKET_PATH, method.FEATURE_PATH)
    feature_cols = v5.frozen_feature_order(ml_df, xraw, signed)

    canonical, canonical_features, canonical_ll = v5.generate_oof(method.MARKET_PATH, method.FEATURE_PATH)
    if list(canonical_features) != list(feature_cols):
        raise RuntimeError("V5 feature-order reproduction mismatch")
    if abs(canonical_ll - v5.EXPECTED_OOF_LOG_LOSS) > 1e-12:
        raise RuntimeError(f"V5 OOF reproduction mismatch: {canonical_ll} vs {v5.EXPECTED_OOF_LOG_LOSS}")

    hist_parts = []
    years = sorted(int(y) for y in ml_df["date"].dt.year.dropna().unique() if int(y) < 2021)
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        tr = ml_df["date"] < start
        va = (ml_df["date"] >= start) & (ml_df["date"] <= end)
        if int(tr.sum()) < MIN_HISTORICAL_ML_TRAIN_ROWS or not va.any():
            continue
        ytr = ml_df.loc[tr, "won"].astype(int)
        if ytr.nunique() < 2:
            continue
        p = _v5_fit_predict(ml_df, xraw, feature_cols, tr, va)
        hist_parts.append(pd.DataFrame({
            "fight_id": ml_df.loc[va, "fight_id"].astype(str).to_numpy(),
            "date": ml_df.loc[va, "date"].to_numpy(),
            "ml_fold": f"historical_{year}",
            "market_p_red": ml_df.loc[va, "fair_market_p"].to_numpy(float),
            "model_p_red": p,
        }))

    historical = pd.concat(hist_parts, ignore_index=True) if hist_parts else pd.DataFrame()
    canonical = canonical.rename(columns={"fold": "ml_fold"}).copy()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["date"] = pd.to_datetime(canonical["date"])
    stack = pd.concat([historical, canonical], ignore_index=True, sort=False)
    stack = stack.sort_values(["date", "fight_id"]).drop_duplicates("fight_id", keep="last").reset_index(drop=True)
    return stack, feature_cols, canonical_ll


def winner_oriented(frame, features):
    sign = np.where(frame["target"].to_numpy(int) < 3, 1.0, -1.0)
    x = frame[features].replace([np.inf, -np.inf], np.nan).mul(sign, axis=0)
    y = (frame["target"].to_numpy(int) % 3 == 0).astype(int)
    ml = np.where(sign > 0, frame["model_p_red"].to_numpy(float), 1.0 - frame["model_p_red"].to_numpy(float))
    return x, y, ml


def rank_ko_features(df, safe_features):
    pre = df[(df["date"] <= "2020-12-31") & df["model_p_red"].notna()].copy()
    if len(pre) < 100:
        raise RuntimeError(f"insufficient pre-2021 honest-stack rows: {len(pre)}")
    x, y, _ = winner_oriented(pre, safe_features)
    valid = [c for c in safe_features if x[c].notna().any()]
    med = x[valid].median(numeric_only=True)
    x = x[valid].fillna(med).fillna(0.0)
    base = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    d = xgb.DMatrix(x, label=y, base_margin=np.full(len(y), logit([base])[0]), feature_names=valid)
    booster = xgb.train(PARAMS, d, num_boost_round=ROUNDS, verbose_eval=False)
    gain = booster.get_score(importance_type="gain")
    ranked = sorted(valid, key=lambda c: (-float(gain.get(c, 0.0)), c))
    return ranked[: min(TOP_K, len(ranked))], ranked


def fit_conditional(train, score, features, include_ml, orientation):
    xtr, ytr, mltr = winner_oriented(train, features)
    if orientation == "winner":
        xsc, _, mlsc = winner_oriented(score, features)
    elif orientation == "red":
        xsc = score[features].replace([np.inf, -np.inf], np.nan).copy()
        mlsc = score["model_p_red"].to_numpy(float)
    elif orientation == "blue":
        xsc = -score[features].replace([np.inf, -np.inf], np.nan).copy()
        mlsc = 1.0 - score["model_p_red"].to_numpy(float)
    else:
        raise ValueError(orientation)

    valid = [c for c in features if xtr[c].notna().any()]
    med = xtr[valid].median(numeric_only=True)
    xtr = xtr[valid].fillna(med).fillna(0.0)
    xsc = xsc[valid].fillna(med).fillna(0.0)
    if include_ml:
        xtr = xtr.copy()
        xsc = xsc.copy()
        xtr["v5_ml_p_side"] = mltr
        xsc["v5_ml_p_side"] = mlsc
    base = float(np.clip(ytr.mean(), 1e-4, 1 - 1e-4))
    margin = logit([base])[0]
    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=np.full(len(ytr), margin), feature_names=list(xtr.columns))
    dsc = xgb.DMatrix(xsc, base_margin=np.full(len(xsc), margin), feature_names=list(xsc.columns))
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = np.asarray(booster.predict(dsc), float)
    return clip_p(pred), len(valid) + int(include_ml), int(len(train)), base


def conditional_market_ko(frame):
    red_total = frame[["market_red_ko", "market_red_sub", "market_red_dec"]].sum(axis=1).to_numpy(float)
    blue_total = frame[["market_blue_ko", "market_blue_sub", "market_blue_dec"]].sum(axis=1).to_numpy(float)
    red_q = frame["market_red_ko"].to_numpy(float) / np.clip(red_total, EPS, None)
    blue_q = frame["market_blue_ko"].to_numpy(float) / np.clip(blue_total, EPS, None)
    red_winner = frame["target"].to_numpy(int) < 3
    return np.where(red_winner, red_q, blue_q)


def exact3_market(frame):
    red = frame["market_red_ko"].to_numpy(float)
    blue = frame["market_blue_ko"].to_numpy(float)
    no = 1.0 - red - blue
    p = np.column_stack([red, blue, no])
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def exact3_target(target):
    target = np.asarray(target, int)
    return np.where(target == 0, 0, np.where(target == 3, 1, 2))


def american_from_profit_per_100(x):
    x = np.asarray(x, float)
    return np.where(x >= 100.0, x, -10000.0 / np.clip(x, 1e-9, None))


def load_raw_ko_market():
    m = pd.read_parquet(method.MARKET_PATH).copy()
    m = m[
        (m["bookmaker"] == "legacy_consensus")
        & (m["market_key"] == "win_by_ko_tko_dq")
        & m["outcome_side"].astype(str).isin(["red", "blue"])
        & (m["result_status"] == "graded")
        & m["won"].notna()
    ].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["profit_per_100"] = pd.to_numeric(m["profit_per_100"], errors="coerce")
    m = m.dropna(subset=["fight_id", "implied_probability", "profit_per_100"])
    counts = m.groupby(["fight_id", "outcome_side"]).size()
    good = counts[counts.eq(1)].index
    good_set = set((str(a), str(b)) for a, b in good)
    m = m[m.apply(lambda r: (str(r["fight_id"]), str(r["outcome_side"])) in good_set, axis=1)].copy()
    return m[["fight_id", "outcome_side", "implied_probability", "profit_per_100"]]


def build_edge_candidates(pred):
    raw = load_raw_ko_market()
    rows = []
    for variant in ["no_ml", "with_ml"]:
        for side in ["red", "blue"]:
            qcol = f"{variant}_q_{side}_ko_given_win"
            pcol = f"{variant}_p_{side}_ko"
            part = pred[[
                "fight_id", "date", "fold", "event_name", "red_fighter", "blue_fighter", "target",
                "betting_eligible", "model_p_red", "market_p_red", qcol, pcol,
            ]].copy()
            part["variant"] = variant
            part["side"] = side
            part["fighter"] = np.where(side == "red", part["red_fighter"], part["blue_fighter"])
            part["q_ko_given_win"] = part[qcol]
            part["model_p_ko"] = part[pcol]
            part["actual_win"] = np.where(side == "red", part["target"].to_numpy(int) < 3, part["target"].to_numpy(int) >= 3).astype(int)
            part["actual_ko_win"] = np.where(side == "red", part["target"].to_numpy(int) == 0, part["target"].to_numpy(int) == 3).astype(int)
            part["v5_ml_p_side"] = np.where(side == "red", part["model_p_red"], 1 - part["model_p_red"])
            part["market_ml_p_side"] = np.where(side == "red", part["market_p_red"], 1 - part["market_p_red"])
            part["v5_projected_winner"] = part["v5_ml_p_side"] >= 0.5
            part["market_ml_favorite"] = part["market_ml_p_side"] >= 0.5
            part["winner_agreement"] = ((part["model_p_red"] >= 0.5) == (part["market_p_red"] >= 0.5))
            rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(raw, left_on=["fight_id", "side"], right_on=["fight_id", "outcome_side"], how="inner")
    out["market_raw_p"] = out["implied_probability"].astype(float)
    out["win_profit_units"] = out["profit_per_100"].astype(float) / 100.0
    out["decimal_odds"] = 1.0 + out["win_profit_units"]
    out["american_odds"] = american_from_profit_per_100(out["profit_per_100"])
    out["ev"] = out["model_p_ko"] * out["decimal_odds"] - 1.0
    out["prob_diff"] = out["model_p_ko"] - out["market_raw_p"]
    out["prob_ratio"] = out["model_p_ko"] / np.clip(out["market_raw_p"], EPS, None)
    out["logit_residual"] = logit(out["model_p_ko"]) - logit(out["market_raw_p"])
    out["profit_units"] = np.where(out["actual_ko_win"].eq(1), out["win_profit_units"], -1.0)
    out["year"] = pd.to_datetime(out["date"]).dt.year.astype(int)
    return out.sort_values(["date", "fight_id", "variant", "side"]).reset_index(drop=True)


def bet_stats(bets):
    if bets.empty:
        return {"bets": 0, "wins": 0, "profit_units": 0.0, "roi": None, "hit_rate": None, "mean_odds": None, "median_odds": None}
    profit = float(bets["profit_units"].sum())
    return {
        "bets": int(len(bets)),
        "wins": int(bets["actual_ko_win"].sum()),
        "profit_units": profit,
        "roi": profit / len(bets),
        "hit_rate": float(bets["actual_ko_win"].mean()),
        "mean_odds": float(bets["american_odds"].mean()),
        "median_odds": float(bets["american_odds"].median()),
        "mean_model_p": float(bets["model_p_ko"].mean()),
        "mean_market_p": float(bets["market_raw_p"].mean()),
        "mean_ev": float(bets["ev"].mean()),
    }


def choose_one(frame, rank):
    return frame.sort_values(["date", "fight_id", rank, "model_p_ko"], ascending=[True, True, False, False]).drop_duplicates("fight_id", keep="first")


def rule_sets(cand, variant):
    x = cand[cand["variant"].eq(variant)].copy()
    return {
        "positive_ev_one_best_ev": choose_one(x[x["ev"] > 0], "ev"),
        "residual_030_one": choose_one(x[x["logit_residual"] >= 0.30], "logit_residual"),
        "v5_winner_positive_ev_one": choose_one(x[(x["v5_projected_winner"]) & (x["ev"] > 0)], "ev"),
        "v5_winner_agree_positive_ev_one": choose_one(x[(x["v5_projected_winner"]) & (x["winner_agreement"]) & (x["ev"] > 0)], "ev"),
        "v5_winner_residual030_one": choose_one(x[(x["v5_projected_winner"]) & (x["logit_residual"] >= 0.30)], "logit_residual"),
        "v5_winner_agree_residual030_one": choose_one(x[(x["v5_projected_winner"]) & (x["winner_agreement"]) & (x["logit_residual"] >= 0.30)], "logit_residual"),
    }


def evaluate_rules(cand):
    rules_rows = []
    robust_rows = []
    for variant in ["no_ml", "with_ml"]:
        for name, bets in rule_sets(cand, variant).items():
            pooled = bet_stats(bets)
            rules_rows.append({"variant": variant, "rule": name, "scope": "pooled", **pooled})
            for year in [2021, 2022, 2023, 2024]:
                rules_rows.append({"variant": variant, "rule": name, "scope": f"year_{year}", **bet_stats(bets[bets["year"].eq(year)])})
            for omit in [2021, 2022, 2023, 2024]:
                robust_rows.append({"variant": variant, "rule": name, "check": f"leave_out_{omit}", **bet_stats(bets[~bets["year"].eq(omit)])})
            if not bets.empty:
                largest_idx = bets["profit_units"].idxmax()
                robust_rows.append({"variant": variant, "rule": name, "check": "remove_largest_winner", **bet_stats(bets.drop(index=largest_idx))})
            for cap in [500, 750, 1000]:
                capped = bets[(bets["american_odds"] < 0) | (bets["american_odds"] <= cap)]
                robust_rows.append({"variant": variant, "rule": name, "check": f"odds_cap_plus_{cap}", **bet_stats(capped)})
    return pd.DataFrame(rules_rows), pd.DataFrame(robust_rows)


def calibration_table(cand):
    rows = []
    bins = [-0.001, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 1.001]
    labels = ["<.05", ".05-.099", ".10-.149", ".15-.199", ".20-.299", ".30-.399", ".40-.549", ".55+"]
    for variant in ["no_ml", "with_ml"]:
        x = cand[cand["variant"].eq(variant)].copy()
        x["bin"] = pd.cut(x["model_p_ko"], bins=bins, labels=labels, include_lowest=True, right=False)
        for label, g in x.groupby("bin", observed=True):
            rows.append({
                "variant": variant,
                "metric": "side_exact_ko_calibration",
                "bucket": str(label),
                "n": int(len(g)),
                "mean_probability": float(g["model_p_ko"].mean()),
                "actual_rate": float(g["actual_ko_win"].mean()),
                "calibration_error": float(g["model_p_ko"].mean() - g["actual_ko_win"].mean()),
            })
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df, safe_features, excluded = method._build_rows(True, True)
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > "2024-12-31").any():
        raise RuntimeError("2025+ entered KO development frame")

    ml_stack, v5_features, v5_ll = build_honest_v5_stack()
    ml_stack["fight_id"] = ml_stack["fight_id"].astype(str)
    df["fight_id"] = df["fight_id"].astype(str)
    df = df.merge(ml_stack[["fight_id", "model_p_red", "market_p_red", "ml_fold"]], on="fight_id", how="left")

    ko_features, ranked = rank_ko_features(df, safe_features)
    parts = []
    metric_rows = []

    for fold, train_end, val_start, val_end in method.FOLDS:
        train = df[(df["date"] <= train_end) & df["model_p_red"].notna()].copy()
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end) & df["model_p_red"].notna()].copy()
        if train.empty or val.empty:
            raise RuntimeError(f"empty KO fold {fold}")

        y_cond = (val["target"].to_numpy(int) % 3 == 0).astype(int)
        market_cond = conditional_market_ko(val)
        market3 = exact3_market(val)
        y3 = exact3_target(val["target"])
        metric_rows.append({"variant": "method_market", "fold": fold, "metric": "conditional_ko", **binary_metrics(y_cond, market_cond)})
        metric_rows.append({"variant": "method_market", "fold": fold, "metric": "exact_3class", **multiclass3_metrics(y3, market3)})
        metric_rows.append({"variant": "method_market", "fold": fold, "metric": "fight_ko", **binary_metrics((y3 != 2).astype(int), market3[:, 0] + market3[:, 1])})

        out = val[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target", "betting_eligible", "cold_start", "red_prior_ufc_fights", "blue_prior_ufc_fights", "model_p_red", "market_p_red"]].copy()
        out["fold"] = fold

        for variant, include_ml in [("no_ml", False), ("with_ml", True)]:
            qwin, fc, train_n, base = fit_conditional(train, val, ko_features, include_ml, "winner")
            qred, _, _, _ = fit_conditional(train, val, ko_features, include_ml, "red")
            qblue, _, _, _ = fit_conditional(train, val, ko_features, include_ml, "blue")
            ml_red = val["model_p_red"].to_numpy(float)
            pred3 = np.column_stack([
                ml_red * qred,
                (1 - ml_red) * qblue,
                1 - (ml_red * qred) - ((1 - ml_red) * qblue),
            ])
            pred3 = np.clip(pred3, EPS, None)
            pred3 = pred3 / pred3.sum(axis=1, keepdims=True)
            metric_rows.append({"variant": variant, "fold": fold, "metric": "conditional_ko", "feature_count": fc, "train_n": train_n, "train_ko_rate": base, **binary_metrics(y_cond, qwin)})
            metric_rows.append({"variant": variant, "fold": fold, "metric": "exact_3class", "feature_count": fc, "train_n": train_n, "train_ko_rate": base, **multiclass3_metrics(y3, pred3)})
            metric_rows.append({"variant": variant, "fold": fold, "metric": "fight_ko", "feature_count": fc, "train_n": train_n, "train_ko_rate": base, **binary_metrics((y3 != 2).astype(int), pred3[:, 0] + pred3[:, 1])})
            out[f"{variant}_q_winner_ko_given_win"] = qwin
            out[f"{variant}_q_red_ko_given_win"] = qred
            out[f"{variant}_q_blue_ko_given_win"] = qblue
            out[f"{variant}_p_red_ko"] = pred3[:, 0]
            out[f"{variant}_p_blue_ko"] = pred3[:, 1]
            out[f"{variant}_p_no_ko"] = pred3[:, 2]
        parts.append(out)

    pred = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)

    y_cond = (pred["target"].to_numpy(int) % 3 == 0).astype(int)
    y3 = exact3_target(pred["target"])
    method_df = df[["fight_id", "market_red_ko", "market_red_sub", "market_red_dec", "market_blue_ko", "market_blue_sub", "market_blue_dec"]]
    eval_frame = pred.merge(method_df, on="fight_id", how="left")
    market_cond = conditional_market_ko(eval_frame)
    market3 = exact3_market(eval_frame)
    pooled = {
        "method_market": {
            "conditional_ko": binary_metrics(y_cond, market_cond),
            "exact_3class": multiclass3_metrics(y3, market3),
            "fight_ko": binary_metrics((y3 != 2).astype(int), market3[:, 0] + market3[:, 1]),
        }
    }
    for variant in ["no_ml", "with_ml"]:
        p3 = pred[[f"{variant}_p_red_ko", f"{variant}_p_blue_ko", f"{variant}_p_no_ko"]].to_numpy(float)
        pooled[variant] = {
            "conditional_ko": binary_metrics(y_cond, pred[f"{variant}_q_winner_ko_given_win"]),
            "exact_3class": multiclass3_metrics(y3, p3),
            "fight_ko": binary_metrics((y3 != 2).astype(int), p3[:, 0] + p3[:, 1]),
        }
        for metric_name, vals in pooled[variant].items():
            metric_rows.append({"variant": variant, "fold": "pooled_2021_2024", "metric": metric_name, **vals})
    for metric_name, vals in pooled["method_market"].items():
        metric_rows.append({"variant": "method_market", "fold": "pooled_2021_2024", "metric": metric_name, **vals})

    winner = min(["no_ml", "with_ml"], key=lambda z: (pooled[z]["exact_3class"]["log_loss"], pooled[z]["exact_3class"]["brier"]))

    cand = build_edge_candidates(pred)
    rule_df, robust_df = evaluate_rules(cand)
    cal_df = calibration_table(cand)
    metrics_df = pd.concat([pd.DataFrame(metric_rows), cal_df], ignore_index=True, sort=False)

    pred.to_csv(PREDICTIONS, index=False)
    cand.to_csv(CANDIDATES, index=False)
    metrics_df.to_csv(METRICS, index=False)
    rule_df.to_csv(RULES, index=False)
    robust_df.to_csv(ROBUSTNESS, index=False)
    FEATURES.write_text(json.dumps({
        "ko_feature_count": len(ko_features),
        "ko_features": ko_features,
        "pre2021_ranked_features": ranked,
        "safe_feature_count": len(safe_features),
        "excluded_signed_diff_columns": excluded,
        "v5_feature_count": len(v5_features),
        "v5_features": v5_features,
    }, indent=2))

    summary = {
        "experiment": "conditional_ko_xgboost_with_frozen_v5_ml_stack_v1",
        "purpose": "test whether frozen V5 moneyline probability adds predictive signal to a KO-vs-non-KO conditional XGBoost; exact KO = P(win)*P(KO|win)",
        "development_window": "chronological 2021-2024 OOF only",
        "reads_2025_plus": False,
        "method_market_used_as_model_feature": False,
        "roi_used_for_model_selection": False,
        "selection_metric": "pooled 2021-2024 exact red-KO / blue-KO / no-KO 3-class log loss; Brier tiebreak",
        "selected_variant": winner,
        "v5_canonical_oof_log_loss": v5_ll,
        "v5_expected_oof_log_loss": v5.EXPECTED_OOF_LOG_LOSS,
        "v5_exact_reproduction": abs(v5_ll - v5.EXPECTED_OOF_LOG_LOSS) <= 1e-12,
        "historical_ml_stack_policy": f"expanding calendar-year V5 fits using only earlier fights; minimum {MIN_HISTORICAL_ML_TRAIN_ROWS} prior ML rows; frozen pre-2021 V5 feature order",
        "ko_feature_selection": f"top {TOP_K} gain-ranked signed-difference features using pre-2021 winner-oriented KO target only",
        "ko_hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "pooled_probability_metrics": pooled,
        "delta_with_ml_vs_no_ml": {
            "conditional_log_loss": pooled["with_ml"]["conditional_ko"]["log_loss"] - pooled["no_ml"]["conditional_ko"]["log_loss"],
            "conditional_brier": pooled["with_ml"]["conditional_ko"]["brier"] - pooled["no_ml"]["conditional_ko"]["brier"],
            "exact_3class_log_loss": pooled["with_ml"]["exact_3class"]["log_loss"] - pooled["no_ml"]["exact_3class"]["log_loss"],
            "exact_3class_brier": pooled["with_ml"]["exact_3class"]["brier"] - pooled["no_ml"]["exact_3class"]["brier"],
            "fight_ko_log_loss": pooled["with_ml"]["fight_ko"]["log_loss"] - pooled["no_ml"]["fight_ko"]["log_loss"],
        },
        "oof_fights": int(len(pred)),
        "edge_candidate_rows": int(len(cand)),
        "edge_rules_are_diagnostic_only": True,
        "edge_rule_names": sorted(rule_df["rule"].dropna().unique().tolist()),
        "artifacts": [str(PREDICTIONS), str(CANDIDATES), str(METRICS), str(RULES), str(ROBUSTNESS), str(FEATURES)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
