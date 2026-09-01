from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research import xgboost_method_market_offset as method

OUT = Path("data/research/prop_mispricing")
SUMMARY_OUT = OUT / "xgboost_ko_route_reallocation_small_challenger_summary.json"
METRICS_OUT = OUT / "xgboost_ko_route_reallocation_small_challenger_metrics.csv"
OOF_OUT = OUT / "xgboost_ko_route_reallocation_small_challenger_oof.csv"
HOLDOUT_OUT = OUT / "xgboost_ko_route_reallocation_small_challenger_2025_holdout.csv"
IMPORTANCE_OUT = OUT / "xgboost_ko_route_reallocation_small_challenger_feature_importance.csv"

DEV_START = pd.Timestamp("2021-01-01")
DEV_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-12-31")
CUTOFF = VAL_END
EPS = 1e-9

PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.85,
    "min_child_weight": 15,
    "lambda": 10.0,
    "alpha": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 2,
}
ROUNDS = 250

BASE_FEATURES = [
    "height_diff",
    "reach_diff",
    "ewm_str_acc_diff",
]
ROUTE_CONTROLS = [
    "ewm_td_avg_diff",
    "ewm_ctrl_per_min_diff",
    "ewm_sub_avg_diff",
    "decision_dependency_diff",
]
INTERACTION_FEATURES = [
    "height_x_ewm_str_acc",
    "reach_x_ewm_str_acc",
]

CANDIDATES = {
    "geom_acc_additive": BASE_FEATURES,
    "geom_acc_interactions": BASE_FEATURES + INTERACTION_FEATURES,
    "geom_acc_route_controls": BASE_FEATURES + INTERACTION_FEATURES + ROUTE_CONTROLS,
}


def clip_p(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)


def logit(x: np.ndarray | pd.Series) -> np.ndarray:
    p = clip_p(x)
    return np.log(p / (1.0 - p))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = clip_p(p)
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "actual_ko_share": float(np.mean(y)),
        "mean_ko_probability": float(np.mean(p)),
        "calibration_error": float(np.mean(p) - np.mean(y)),
    }


def multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[y]
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def _build_winner_rows() -> pd.DataFrame:
    original = method.DEV_CUTOFF
    method.DEV_CUTOFF = CUTOFF
    try:
        fights, safe_features, _ = method._build_rows(True, True)
    finally:
        method.DEV_CUTOFF = original

    fights["date"] = pd.to_datetime(fights["date"], errors="coerce")
    fights = fights[fights["date"] <= VAL_END].copy()
    if (fights["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered KO route-reallocation challenger")

    required = sorted(set(BASE_FEATURES + ROUTE_CONTROLS))
    missing = [c for c in required if c not in safe_features or c not in fights.columns]
    if missing:
        raise RuntimeError(f"required signed prefight features missing: {missing}")

    red_winner = fights["target"].astype(int).lt(3).to_numpy()
    sign = np.where(red_winner, 1.0, -1.0)
    winners = fights[[
        "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
        "betting_eligible", "cold_start", "red_prior_ufc_fights", "blue_prior_ufc_fights",
        "market_red_ko", "market_red_sub", "market_red_dec",
        "market_blue_ko", "market_blue_sub", "market_blue_dec",
    ] + required].copy()

    winners["winner_side"] = np.where(red_winner, "red", "blue")
    winners["winner"] = np.where(red_winner, winners["red_fighter"], winners["blue_fighter"])
    winners["loser"] = np.where(red_winner, winners["blue_fighter"], winners["red_fighter"])
    winners["actual_method_class"] = winners["target"].astype(int) % 3
    winners["actual_ko"] = winners["actual_method_class"].eq(0).astype(int)

    for c in required:
        winners[c] = pd.to_numeric(winners[c], errors="coerce") * sign

    winners["height_x_ewm_str_acc"] = winners["height_diff"] * winners["ewm_str_acc_diff"]
    winners["reach_x_ewm_str_acc"] = winners["reach_diff"] * winners["ewm_str_acc_diff"]

    rtot = winners[["market_red_ko", "market_red_sub", "market_red_dec"]].sum(axis=1).to_numpy(float)
    btot = winners[["market_blue_ko", "market_blue_sub", "market_blue_dec"]].sum(axis=1).to_numpy(float)
    rko = winners["market_red_ko"].to_numpy(float) / np.clip(rtot, EPS, None)
    rsub = winners["market_red_sub"].to_numpy(float) / np.clip(rtot, EPS, None)
    rdec = winners["market_red_dec"].to_numpy(float) / np.clip(rtot, EPS, None)
    bko = winners["market_blue_ko"].to_numpy(float) / np.clip(btot, EPS, None)
    bsub = winners["market_blue_sub"].to_numpy(float) / np.clip(btot, EPS, None)
    bdec = winners["market_blue_dec"].to_numpy(float) / np.clip(btot, EPS, None)

    winners["market_cond_ko"] = np.where(red_winner, rko, bko)
    winners["market_cond_sub"] = np.where(red_winner, rsub, bsub)
    winners["market_cond_dec"] = np.where(red_winner, rdec, bdec)
    psum = winners[["market_cond_ko", "market_cond_sub", "market_cond_dec"]].sum(axis=1)
    if float((psum - 1.0).abs().max()) > 1e-8:
        raise RuntimeError("winner conditional market probabilities do not sum to one")

    winners["year"] = winners["date"].dt.year.astype(int)
    return winners.sort_values(["date", "fight_id"]).reset_index(drop=True)


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    xtr = train[features].replace([np.inf, -np.inf], np.nan).copy()
    xsc = score[features].replace([np.inf, -np.inf], np.nan).copy()
    valid = [c for c in features if xtr[c].notna().any()]
    if not valid:
        raise RuntimeError("no valid challenger features in training fold")
    med = xtr[valid].median(numeric_only=True)
    xtr = xtr[valid].fillna(med).fillna(0.0)
    xsc = xsc[valid].fillna(med).fillna(0.0)

    ytr = train["actual_ko"].to_numpy(int)
    base_tr = logit(train["market_cond_ko"])
    base_sc = logit(score["market_cond_ko"])
    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=base_tr, feature_names=valid)
    dsc = xgb.DMatrix(xsc, base_margin=base_sc, feature_names=valid)
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = clip_p(np.asarray(booster.predict(dsc), dtype=float))
    gain = {k: float(v) for k, v in booster.get_score(importance_type="gain").items()}
    return pred, gain


def _conditional_three_class(frame: pd.DataFrame, pko: np.ndarray) -> np.ndarray:
    pko = clip_p(pko)
    sub = frame["market_cond_sub"].to_numpy(float)
    dec = frame["market_cond_dec"].to_numpy(float)
    rest = np.clip(sub + dec, EPS, None)
    psub = (1.0 - pko) * sub / rest
    pdec = (1.0 - pko) * dec / rest
    p = np.column_stack([pko, psub, pdec])
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def _market_three_class(frame: pd.DataFrame) -> np.ndarray:
    return frame[["market_cond_ko", "market_cond_sub", "market_cond_dec"]].to_numpy(float)


def _record_metrics(rows: list[dict], period: str, variant: str, frame: pd.DataFrame, pko: np.ndarray) -> None:
    yko = frame["actual_ko"].to_numpy(int)
    y3 = frame["actual_method_class"].to_numpy(int)
    p3 = _conditional_three_class(frame, pko)
    rows.append({"period": period, "variant": variant, "metric": "binary_ko", **binary_metrics(yko, pko)})
    rows.append({"period": period, "variant": variant, "metric": "conditional_3class", **multiclass_metrics(y3, p3)})


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    winners = _build_winner_rows()
    eligible = winners[winners["betting_eligible"].astype(bool)].copy()
    dev = eligible[eligible["date"].between(DEV_START, DEV_END)].copy()
    holdout = eligible[eligible["date"].between(VAL_START, VAL_END)].copy()
    if dev.empty or holdout.empty:
        raise RuntimeError("missing development or 2025 holdout winner rows")

    # Base-margin contract: zero trees must reproduce the sportsbook conditional KO probability.
    contract_x = pd.DataFrame({"x": np.zeros(min(5, len(dev)))})
    contract_q = dev["market_cond_ko"].head(len(contract_x)).to_numpy(float)
    contract_d = xgb.DMatrix(contract_x, base_margin=logit(contract_q))
    contract_b = xgb.train({"objective": "binary:logistic", "seed": 42}, contract_d, num_boost_round=0)
    contract_pred = np.asarray(contract_b.predict(contract_d), dtype=float)
    contract_error = float(np.max(np.abs(contract_pred - contract_q)))
    if contract_error > 5e-8:
        raise RuntimeError(f"binary base-margin contract failed: {contract_error}")

    metric_rows: list[dict] = []
    importance_rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    folds = [
        ("2021", pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
        ("2022", pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("2023", pd.Timestamp("2022-12-31"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
        ("2024", pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    ]

    for fold, train_end, val_start, val_end in folds:
        train = eligible[eligible["date"] <= train_end].copy()
        val = eligible[eligible["date"].between(val_start, val_end)].copy()
        if len(train) < 100 or val.empty:
            raise RuntimeError(f"insufficient chronological rows for fold {fold}: train={len(train)} val={len(val)}")
        if train["date"].max() >= val["date"].min():
            raise RuntimeError(f"chronology violation in fold {fold}")

        out = val[[
            "fight_id", "date", "event_name", "winner", "loser", "winner_side", "actual_method_class", "actual_ko",
            "market_cond_ko", "market_cond_sub", "market_cond_dec",
        ]].copy()
        out["fold"] = fold
        _record_metrics(metric_rows, fold, "method_market", val, val["market_cond_ko"].to_numpy(float))

        for variant, features in CANDIDATES.items():
            pred, gain = _fit_predict(train, val, features)
            out[f"{variant}_pko"] = pred
            _record_metrics(metric_rows, fold, variant, val, pred)
            for feature in features:
                importance_rows.append({
                    "period": fold,
                    "variant": variant,
                    "feature": feature,
                    "gain": float(gain.get(feature, 0.0)),
                })
        oof_parts.append(out)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    dev_by_id = dev.set_index("fight_id", drop=False)
    eval_dev = dev_by_id.loc[oof["fight_id"].astype(str)].reset_index(drop=True)
    if len(eval_dev) != len(oof):
        raise RuntimeError("OOF/dev alignment failed")

    _record_metrics(metric_rows, "pooled_2021_2024", "method_market", eval_dev, eval_dev["market_cond_ko"].to_numpy(float))
    pooled_scores = {}
    for variant in CANDIDATES:
        pred = oof[f"{variant}_pko"].to_numpy(float)
        _record_metrics(metric_rows, "pooled_2021_2024", variant, eval_dev, pred)
        y3 = eval_dev["actual_method_class"].to_numpy(int)
        m3 = multiclass_metrics(y3, _conditional_three_class(eval_dev, pred))
        b = binary_metrics(eval_dev["actual_ko"].to_numpy(int), pred)
        pooled_scores[variant] = {
            "conditional_3class_log_loss": m3["log_loss"],
            "conditional_3class_brier": m3["brier"],
            "binary_ko_log_loss": b["log_loss"],
            "binary_ko_brier": b["brier"],
        }

    # Freeze candidate before any 2025 score is generated.
    selected = min(
        CANDIDATES,
        key=lambda v: (
            pooled_scores[v]["conditional_3class_log_loss"],
            pooled_scores[v]["binary_ko_log_loss"],
            pooled_scores[v]["conditional_3class_brier"],
        ),
    )
    selected_features = CANDIDATES[selected]

    train_full = eligible[eligible["date"] <= DEV_END].copy()
    holdout_pred, holdout_gain = _fit_predict(train_full, holdout, selected_features)
    _record_metrics(metric_rows, "fixed_2025_holdout", "method_market", holdout, holdout["market_cond_ko"].to_numpy(float))
    _record_metrics(metric_rows, "fixed_2025_holdout", selected, holdout, holdout_pred)
    for feature in selected_features:
        importance_rows.append({
            "period": "fixed_2025_holdout_fit_through_2024",
            "variant": selected,
            "feature": feature,
            "gain": float(holdout_gain.get(feature, 0.0)),
        })

    holdout_out = holdout[[
        "fight_id", "date", "event_name", "winner", "loser", "winner_side", "actual_method_class", "actual_ko",
        "market_cond_ko", "market_cond_sub", "market_cond_dec",
    ] + selected_features].copy()
    holdout_out["selected_variant"] = selected
    holdout_out["selected_pko"] = holdout_pred
    selected_3 = _conditional_three_class(holdout, holdout_pred)
    holdout_out["selected_psub"] = selected_3[:, 1]
    holdout_out["selected_pdec"] = selected_3[:, 2]
    holdout_out["ko_adjustment_vs_market"] = holdout_out["selected_pko"] - holdout_out["market_cond_ko"]

    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)
    metrics.to_csv(METRICS_OUT, index=False)
    oof.to_csv(OOF_OUT, index=False)
    holdout_out.to_csv(HOLDOUT_OUT, index=False)
    importance.to_csv(IMPORTANCE_OUT, index=False)

    pooled_market_3 = multiclass_metrics(
        eval_dev["actual_method_class"].to_numpy(int),
        _market_three_class(eval_dev),
    )
    pooled_market_bin = binary_metrics(
        eval_dev["actual_ko"].to_numpy(int),
        eval_dev["market_cond_ko"].to_numpy(float),
    )
    hold_market_3 = multiclass_metrics(holdout["actual_method_class"].to_numpy(int), _market_three_class(holdout))
    hold_selected_3 = multiclass_metrics(holdout["actual_method_class"].to_numpy(int), selected_3)
    hold_market_bin = binary_metrics(holdout["actual_ko"].to_numpy(int), holdout["market_cond_ko"].to_numpy(float))
    hold_selected_bin = binary_metrics(holdout["actual_ko"].to_numpy(int), holdout_pred)

    summary = {
        "experiment": "ko_route_reallocation_small_market_offset_challenger_v1",
        "purpose": "Test a small interpretable residual challenger for sportsbook P(KO | winner), centered on geometry plus striking efficiency and optional route controls.",
        "model_family": "binary XGBoost market-offset residual; KO probability base margin is sportsbook conditional KO; SUB/DEC retain sportsbook relative split after KO adjustment",
        "development_window": "chronological expanding OOF 2021-2024",
        "validation_window": "fixed calendar 2025 scored only after candidate selection",
        "reads_2026_plus": False,
        "roi_used": False,
        "candidate_feature_sets_fixed_before_run": CANDIDATES,
        "hyperparameters_fixed_before_run": {**PARAMS, "rounds": ROUNDS},
        "selection_rule": "lowest pooled 2021-2024 winner-conditional 3-class log loss; binary KO log loss then 3-class Brier as tie-breakers",
        "selected_variant": selected,
        "selected_features": selected_features,
        "base_margin_contract_max_abs_error": contract_error,
        "development_winner_rows": int(len(dev)),
        "holdout_2025_winner_rows": int(len(holdout)),
        "pooled_candidate_scores": pooled_scores,
        "pooled_market": {
            "conditional_3class": pooled_market_3,
            "binary_ko": pooled_market_bin,
        },
        "selected_dev_delta_vs_market": {
            "conditional_3class_log_loss": float(pooled_scores[selected]["conditional_3class_log_loss"] - pooled_market_3["log_loss"]),
            "binary_ko_log_loss": float(pooled_scores[selected]["binary_ko_log_loss"] - pooled_market_bin["log_loss"]),
            "conditional_3class_brier": float(pooled_scores[selected]["conditional_3class_brier"] - pooled_market_3["brier"]),
            "binary_ko_brier": float(pooled_scores[selected]["binary_ko_brier"] - pooled_market_bin["brier"]),
        },
        "fixed_2025_market": {
            "conditional_3class": hold_market_3,
            "binary_ko": hold_market_bin,
        },
        "fixed_2025_selected": {
            "conditional_3class": hold_selected_3,
            "binary_ko": hold_selected_bin,
        },
        "fixed_2025_delta_vs_market": {
            "conditional_3class_log_loss": float(hold_selected_3["log_loss"] - hold_market_3["log_loss"]),
            "binary_ko_log_loss": float(hold_selected_bin["log_loss"] - hold_market_bin["log_loss"]),
            "conditional_3class_brier": float(hold_selected_3["brier"] - hold_market_3["brier"]),
            "binary_ko_brier": float(hold_selected_bin["brier"] - hold_market_bin["brier"]),
        },
        "important_caveat": "The feature family was motivated by descriptive 2021-2024 diagnostics, so development OOF is chronological but feature-family selection is development-derived. Calendar 2025 is the untouched confirmation set for this challenger definition.",
        "artifacts": [str(METRICS_OUT), str(OOF_OUT), str(HOLDOUT_OUT), str(IMPORTANCE_OUT)],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
