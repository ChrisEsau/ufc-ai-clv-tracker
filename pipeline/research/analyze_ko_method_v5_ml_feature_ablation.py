from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("data/research/prop_mispricing")
OOF_PATH = ROOT / "xgboost_method_hierarchical_v5_oof_predictions.csv"
FEATURE_PATH = Path("data/features/moneyline_feature_view.parquet")
FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"

PRED_PATH = ROOT / "xgboost_ko_method_v5_ml_feature_ablation_predictions.csv"
METRICS_PATH = ROOT / "xgboost_ko_method_v5_ml_feature_ablation_metrics.csv"
IMPORTANCE_PATH = ROOT / "xgboost_ko_method_v5_ml_feature_ablation_feature_importance.csv"
SUMMARY_PATH = ROOT / "xgboost_ko_method_v5_ml_feature_ablation_summary.json"

# 2021 is the seed OOF-history year. Every scored fold trains only on prior OOF years.
FOLDS = [
    ("2022", [2021], 2022),
    ("2023", [2021, 2022], 2023),
    ("2024", [2021, 2022, 2023], 2024),
]

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
EPS = 1e-9
ML_FEATURE = "v5_ml_p_side"


def _clip(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _metric_row(scope: str, variant: str, family: str, y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "scope": scope,
        "variant": variant,
        "family": family,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "observed_rate": float(y.mean()) if len(y) else None,
        "mean_probability": float(p.mean()) if len(p) else None,
        "log_loss": _log_loss(y, p),
        "brier": _brier(y, p),
        "auc": _auc(y, p),
        "calibration_error_mean_p_minus_rate": float(p.mean() - y.mean()) if len(y) else None,
    }


def _load() -> tuple[pd.DataFrame, list[str]]:
    with FEATURE_LIST_PATH.open() as f:
        spec = json.load(f)
    features = list(spec["features"])
    if len(features) != 140 or len(set(features)) != len(features):
        raise RuntimeError(f"expected exactly 140 unique frozen method features, got {len(features)}")
    if any(not c.endswith("_diff") for c in features):
        raise RuntimeError("frozen method feature list contains a non-difference feature")

    oof = pd.read_csv(OOF_PATH)
    required_oof = {
        "fight_id", "date", "fold", "target", "v5_model_p_red",
        "hier_red_ko", "hier_blue_ko",
    }
    missing = sorted(required_oof - set(oof.columns))
    if missing:
        raise RuntimeError(f"OOF file missing required columns: {missing}")
    oof["fight_id"] = oof["fight_id"].astype(str)
    oof["date"] = pd.to_datetime(oof["date"], errors="raise")
    oof["year"] = oof["date"].dt.year.astype(int)
    oof["fold"] = pd.to_numeric(oof["fold"], errors="raise").astype(int)
    oof["target"] = pd.to_numeric(oof["target"], errors="raise").astype(int)
    oof["v5_model_p_red"] = pd.to_numeric(oof["v5_model_p_red"], errors="raise")
    if oof["fight_id"].duplicated().any():
        raise RuntimeError("hierarchical V5 OOF file contains duplicate fight_id")
    if not set(oof["year"].unique()).issubset({2021, 2022, 2023, 2024}):
        raise RuntimeError(f"unexpected OOF years: {sorted(oof['year'].unique())}")
    if not (oof["year"] == oof["fold"]).all():
        raise RuntimeError("OOF fold/year mismatch")
    if not oof["target"].between(0, 5).all():
        raise RuntimeError("target outside frozen six-way class range 0..5")
    if not ((oof["v5_model_p_red"] > 0) & (oof["v5_model_p_red"] < 1)).all():
        raise RuntimeError("V5 ML OOF probabilities must be strictly inside (0,1)")

    fv = pd.read_parquet(FEATURE_PATH, columns=["fight_id"] + features).copy()
    fv["fight_id"] = fv["fight_id"].astype(str)
    if fv["fight_id"].duplicated().any():
        raise RuntimeError("feature view contains duplicate fight_id")
    for c in features:
        fv[c] = pd.to_numeric(fv[c], errors="coerce")

    df = oof.merge(fv, on="fight_id", how="left", validate="one_to_one", indicator=True)
    missing_feature_rows = df.loc[df["_merge"] != "both", "fight_id"].tolist()
    if missing_feature_rows:
        raise RuntimeError(f"OOF fights missing from feature view: {missing_feature_rows[:20]}")
    df = df.drop(columns="_merge").sort_values(["date", "fight_id"]).reset_index(drop=True)

    df["actual_winner_side"] = np.where(df["target"] <= 2, "red", "blue")
    df["actual_ko"] = df["target"].isin([0, 3]).astype(int)
    df["actual_red_ko"] = (df["target"] == 0).astype(int)
    df["actual_blue_ko"] = (df["target"] == 3).astype(int)
    df["projected_winner_side"] = np.where(df["v5_model_p_red"] >= 0.5, "red", "blue")
    df["projected_winner_ko"] = np.where(
        df["projected_winner_side"].eq("red"),
        df["actual_red_ko"],
        df["actual_blue_ko"],
    ).astype(int)

    red_ml = _clip(df["v5_model_p_red"])
    blue_ml = _clip(1.0 - df["v5_model_p_red"])
    df["existing_cond_ko_red"] = np.clip(df["hier_red_ko"].to_numpy(float) / red_ml, 0.0, 1.0)
    df["existing_cond_ko_blue"] = np.clip(df["hier_blue_ko"].to_numpy(float) / blue_ml, 0.0, 1.0)
    return df, features


def _oriented_x(df: pd.DataFrame, features: list[str], side: str | pd.Series) -> pd.DataFrame:
    x = df[features].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    if isinstance(side, str):
        if side == "red":
            return x
        if side == "blue":
            return -x
        raise ValueError(side)
    sign = np.where(pd.Series(side, index=df.index).astype(str).eq("red"), 1.0, -1.0)
    return x.mul(sign, axis=0)


def _fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[xgb.Booster, np.ndarray]:
    valid = [c for c in feature_names if x_train[c].notna().any()]
    if not valid:
        raise RuntimeError("no valid training features")
    med = x_train[valid].median(numeric_only=True)
    tr = x_train[valid].fillna(med).fillna(0.0)
    te = x_test[valid].fillna(med).fillna(0.0)
    dtr = xgb.DMatrix(tr, label=np.asarray(y_train, dtype=int), feature_names=valid)
    dte = xgb.DMatrix(te, feature_names=valid)
    model = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    p = np.asarray(model.predict(dte), dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise RuntimeError("invalid XGBoost prediction")
    return model, p


def _importance_rows(model: xgb.Booster, fold: str, variant: str, all_features: list[str]) -> list[dict]:
    gain = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    cover = model.get_score(importance_type="cover")
    ordered = sorted(all_features, key=lambda c: (-gain.get(c, 0.0), c))
    rank = {c: i + 1 for i, c in enumerate(ordered)}
    return [
        {
            "fold": fold,
            "variant": variant,
            "feature": c,
            "gain": float(gain.get(c, 0.0)),
            "weight": float(weight.get(c, 0.0)),
            "cover": float(cover.get(c, 0.0)),
            "gain_rank": int(rank[c]),
            "used": bool(c in gain or c in weight),
        }
        for c in all_features
    ]


def _run_fold(df: pd.DataFrame, features: list[str], fold: str, train_years: list[int], test_year: int):
    train = df[df["year"].isin(train_years)].copy()
    test = df[df["year"].eq(test_year)].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"empty train/test for {fold}")
    if int(train["year"].max()) >= test_year:
        raise RuntimeError(f"non-chronological fold {fold}")

    # Conditional KO model training rows are oriented to the historical ACTUAL winner.
    xtr_base = _oriented_x(train, features, train["actual_winner_side"])
    ytr = train["actual_ko"].to_numpy(dtype=int)
    train_ml_side = np.where(
        train["actual_winner_side"].eq("red"),
        train["v5_model_p_red"].to_numpy(float),
        1.0 - train["v5_model_p_red"].to_numpy(float),
    )
    xtr_ml = xtr_base.copy()
    xtr_ml[ML_FEATURE] = train_ml_side

    # Score both possible winner sides for each held-out fight.
    xtest_red = _oriented_x(test, features, "red")
    xtest_blue = _oriented_x(test, features, "blue")
    xtest_both_base = pd.concat([xtest_red, xtest_blue], ignore_index=True)

    xtest_red_ml = xtest_red.copy()
    xtest_red_ml[ML_FEATURE] = test["v5_model_p_red"].to_numpy(float)
    xtest_blue_ml = xtest_blue.copy()
    xtest_blue_ml[ML_FEATURE] = 1.0 - test["v5_model_p_red"].to_numpy(float)
    xtest_both_ml = pd.concat([xtest_red_ml, xtest_blue_ml], ignore_index=True)

    base_model, base_both = _fit_model(xtr_base, ytr, xtest_both_base, features)
    ml_model, ml_both = _fit_model(xtr_ml, ytr, xtest_both_ml, features + [ML_FEATURE])
    n = len(test)
    if len(base_both) != 2 * n or len(ml_both) != 2 * n:
        raise RuntimeError("unexpected side-score row count")

    out = test[[
        "fight_id", "date", "year", "fold", "event_name", "red_fighter", "blue_fighter",
        "target", "v5_model_p_red", "actual_winner_side", "actual_ko", "actual_red_ko",
        "actual_blue_ko", "projected_winner_side", "projected_winner_ko",
        "hier_red_ko", "hier_blue_ko", "existing_cond_ko_red", "existing_cond_ko_blue",
    ]].copy()
    out["ablation_fold"] = fold
    out["train_years"] = ",".join(str(x) for x in train_years)
    out["base_cond_ko_red"] = base_both[:n]
    out["base_cond_ko_blue"] = base_both[n:]
    out["ml_cond_ko_red"] = ml_both[:n]
    out["ml_cond_ko_blue"] = ml_both[n:]

    p_red = out["v5_model_p_red"].to_numpy(float)
    p_blue = 1.0 - p_red
    out["base_exact_ko_red"] = p_red * out["base_cond_ko_red"].to_numpy(float)
    out["base_exact_ko_blue"] = p_blue * out["base_cond_ko_blue"].to_numpy(float)
    out["ml_exact_ko_red"] = p_red * out["ml_cond_ko_red"].to_numpy(float)
    out["ml_exact_ko_blue"] = p_blue * out["ml_cond_ko_blue"].to_numpy(float)
    out["existing_exact_ko_red"] = out["hier_red_ko"].to_numpy(float)
    out["existing_exact_ko_blue"] = out["hier_blue_ko"].to_numpy(float)

    importance = []
    importance.extend(_importance_rows(base_model, fold, "ko_specialist_no_ml", features))
    importance.extend(_importance_rows(ml_model, fold, "ko_specialist_plus_v5_ml", features + [ML_FEATURE]))
    return out, importance


def _metrics_for_scope(d: pd.DataFrame, scope: str) -> list[dict]:
    rows: list[dict] = []
    variants = {
        "existing_hierarchical_v5": ("existing_cond_ko_red", "existing_cond_ko_blue", "existing_exact_ko_red", "existing_exact_ko_blue"),
        "ko_specialist_no_ml": ("base_cond_ko_red", "base_cond_ko_blue", "base_exact_ko_red", "base_exact_ko_blue"),
        "ko_specialist_plus_v5_ml": ("ml_cond_ko_red", "ml_cond_ko_blue", "ml_exact_ko_red", "ml_exact_ko_blue"),
    }
    actual_red = d["actual_winner_side"].eq("red").to_numpy()
    projected_red = d["projected_winner_side"].eq("red").to_numpy()
    for variant, (cond_r, cond_b, exact_r, exact_b) in variants.items():
        cond_winner = np.where(actual_red, d[cond_r].to_numpy(float), d[cond_b].to_numpy(float))
        rows.append(_metric_row(scope, variant, "conditional_ko_given_actual_win", d["actual_ko"].to_numpy(int), cond_winner))

        y_side = np.concatenate([d["actual_red_ko"].to_numpy(int), d["actual_blue_ko"].to_numpy(int)])
        p_side = np.concatenate([d[exact_r].to_numpy(float), d[exact_b].to_numpy(float)])
        rows.append(_metric_row(scope, variant, "exact_side_ko_two_rows_per_fight", y_side, p_side))

        fight_p = d[exact_r].to_numpy(float) + d[exact_b].to_numpy(float)
        rows.append(_metric_row(scope, variant, "fight_ko_event", d["actual_ko"].to_numpy(int), fight_p))

        projected_p = np.where(projected_red, d[exact_r].to_numpy(float), d[exact_b].to_numpy(float))
        rows.append(_metric_row(scope, variant, "projected_winner_exact_ko", d["projected_winner_ko"].to_numpy(int), projected_p))
    return rows


def _nested_metrics(metrics: pd.DataFrame, scope: str) -> dict:
    out: dict[str, dict] = {}
    for _, r in metrics[metrics["scope"].eq(scope)].iterrows():
        out.setdefault(str(r["family"]), {})[str(r["variant"])] = {
            "n": int(r["n"]),
            "positives": int(r["positives"]),
            "observed_rate": float(r["observed_rate"]),
            "mean_probability": float(r["mean_probability"]),
            "log_loss": float(r["log_loss"]),
            "brier": float(r["brier"]),
            "auc": None if pd.isna(r["auc"]) else float(r["auc"]),
            "calibration_error_mean_p_minus_rate": float(r["calibration_error_mean_p_minus_rate"]),
        }
    return out


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    df, features = _load()

    pred_parts = []
    importance_rows: list[dict] = []
    fold_audit = []
    for fold, train_years, test_year in FOLDS:
        out, imp = _run_fold(df, features, fold, train_years, test_year)
        pred_parts.append(out)
        importance_rows.extend(imp)
        fold_audit.append({
            "fold": fold,
            "train_years": train_years,
            "test_year": test_year,
            "train_n": int(df["year"].isin(train_years).sum()),
            "test_n": int(df["year"].eq(test_year).sum()),
        })

    pred = pd.concat(pred_parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    expected_scored_years = {2022, 2023, 2024}
    if set(pred["year"].unique()) != expected_scored_years:
        raise RuntimeError(f"unexpected scored years: {sorted(pred['year'].unique())}")
    pred.to_csv(PRED_PATH, index=False)

    metric_rows: list[dict] = []
    for year in sorted(pred["year"].unique()):
        metric_rows.extend(_metrics_for_scope(pred[pred["year"].eq(year)], str(year)))
    metric_rows.extend(_metrics_for_scope(pred, "pooled_2022_2024"))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRICS_PATH, index=False)

    importance = pd.DataFrame(importance_rows)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    pooled = _nested_metrics(metrics, "pooled_2022_2024")
    deltas = {}
    for family, vals in pooled.items():
        chal = vals["ko_specialist_plus_v5_ml"]
        base = vals["ko_specialist_no_ml"]
        existing = vals["existing_hierarchical_v5"]
        deltas[family] = {
            "plus_ml_minus_no_ml_log_loss": chal["log_loss"] - base["log_loss"],
            "plus_ml_minus_no_ml_brier": chal["brier"] - base["brier"],
            "plus_ml_minus_existing_log_loss": chal["log_loss"] - existing["log_loss"],
            "plus_ml_minus_existing_brier": chal["brier"] - existing["brier"],
            "negative_delta_is_better": True,
        }

    ml_imp = importance[
        importance["variant"].eq("ko_specialist_plus_v5_ml")
        & importance["feature"].eq(ML_FEATURE)
    ].sort_values("fold")
    ml_feature_importance = [
        {
            "fold": str(r["fold"]),
            "gain": float(r["gain"]),
            "weight": float(r["weight"]),
            "cover": float(r["cover"]),
            "gain_rank_of_141": int(r["gain_rank"]),
            "used": bool(r["used"]),
        }
        for _, r in ml_imp.iterrows()
    ]

    per_year = {str(y): _nested_metrics(metrics, str(y)) for y in [2022, 2023, 2024]}
    summary = {
        "objective": "Leakage-safe ablation: does frozen V5 OOF moneyline probability improve a KO-vs-non-KO conditional XGBoost specialist?",
        "selection_rule": "Predictive metrics only; no ROI or sportsbook price used for training or model selection.",
        "data_guardrails": {
            "source_oof_years": [2021, 2022, 2023, 2024],
            "scored_years": [2022, 2023, 2024],
            "why_2021_not_scored": "2021 is the seed OOF-history year so every ML feature used in model training is itself an already-held-out frozen V5 prediction.",
            "2025_plus_touched": False,
            "sportsbook_prices_used_as_features": False,
            "frozen_method_feature_count": len(features),
            "new_challenger_feature": ML_FEATURE,
            "new_feature_definition": "Frozen V5 OOF P(oriented side wins).",
            "winner_orientation": "Historical training rows use actual winner side solely to define KO|win conditional target; red features are +diff, blue features are -diff.",
            "exact_ko_definition": "Frozen V5 side win probability multiplied by predicted P(KO | side wins).",
        },
        "model": {
            "params": PARAMS,
            "rounds": ROUNDS,
            "baseline_features": len(features),
            "challenger_features": len(features) + 1,
        },
        "folds": fold_audit,
        "cohort": {
            "source_fights_2021_2024": int(len(df)),
            "scored_fights_2022_2024": int(len(pred)),
            "scored_fights_by_year": {str(y): int((pred["year"] == y).sum()) for y in [2022, 2023, 2024]},
            "ko_fights_scored": int(pred["actual_ko"].sum()),
        },
        "pooled_metrics": pooled,
        "pooled_deltas": deltas,
        "per_year_metrics": per_year,
        "v5_ml_feature_importance": ml_feature_importance,
        "artifacts": {
            "predictions": str(PRED_PATH),
            "metrics": str(METRICS_PATH),
            "feature_importance": str(IMPORTANCE_PATH),
            "summary": str(SUMMARY_PATH),
        },
    }
    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "scored_fights": len(pred),
        "pooled_deltas": deltas,
        "v5_ml_feature_importance": ml_feature_importance,
        "outputs": summary["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
