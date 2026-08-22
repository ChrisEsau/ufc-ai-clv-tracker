from __future__ import annotations

"""Run chronological development-only raw signal discovery.

The reserved 2024+ outer period is loaded only to count and report its rows; it
is never used for model selection, fitting, importance, or scoring here.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG = Path("pipeline/research/raw_signal_discovery_v1/config.yaml")
NON_FEATURE_COLUMNS = {
    "fight_id", "event_date", "fighter_id", "opponent_id", "fighter_win",
    "history_max_date", "opponent_history_max_date", "sample_weight",
}


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _clean_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def _usable_features(train: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    X = _clean_matrix(train, feature_cols)
    keep = []
    for c in feature_cols:
        s = X[c]
        if s.notna().mean() < 0.02:
            continue
        if s.dropna().nunique() <= 1:
            continue
        keep.append(c)
    return keep


def _symmetrize(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    temp = frame[["fight_id", "fighter_id", "opponent_id"]].copy()
    temp["raw"] = np.asarray(raw, float)
    lookup = {(str(r.fight_id), str(r.fighter_id)): float(r.raw) for r in temp.itertuples(index=False)}
    out = []
    for r in temp.itertuples(index=False):
        opp_p = lookup[(str(r.fight_id), str(r.opponent_id))]
        out.append(0.5 * (float(r.raw) + (1.0 - opp_p)))
    return np.clip(np.asarray(out, float), 1e-6, 1 - 1e-6)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
    }


def _baseline_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return _metrics(frame["fighter_win"].to_numpy(int), np.full(len(frame), 0.5))


def _fit_logistic(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: dict[str, Any]) -> np.ndarray:
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=float(cfg["C"]),
            max_iter=int(cfg["max_iter"]),
            solver="liblinear",
            random_state=0,
        )),
    ])
    model.fit(
        _clean_matrix(train, features),
        train["fighter_win"].to_numpy(int),
        model__sample_weight=train["sample_weight"].to_numpy(float),
    )
    raw = model.predict_proba(_clean_matrix(valid, features))[:, 1]
    return _symmetrize(valid, raw)


def _fit_xgb(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: dict[str, Any], seed: int):
    params = dict(cfg)
    early_stopping_rounds = int(params.pop("early_stopping_rounds"))
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    X_train = _clean_matrix(train, features)
    X_valid = _clean_matrix(valid, features)
    model.fit(
        X_train,
        train["fighter_win"].to_numpy(int),
        sample_weight=train["sample_weight"].to_numpy(float),
        eval_set=[(X_valid, valid["fighter_win"].to_numpy(int))],
        verbose=False,
    )
    raw = model.predict_proba(X_valid)[:, 1]
    pred = _symmetrize(valid, raw)
    return model, X_valid, pred


def _fold_importance(model: xgb.XGBClassifier, X_valid: pd.DataFrame, fold: int) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    dmatrix = xgb.DMatrix(X_valid, feature_names=list(X_valid.columns))
    contrib = booster.predict(dmatrix, pred_contribs=True)
    shap_abs = np.mean(np.abs(contrib[:, :-1]), axis=0)
    out = pd.DataFrame({
        "fold": fold,
        "feature": list(X_valid.columns),
        "gain": [float(gain.get(c, 0.0)) for c in X_valid.columns],
        "mean_abs_shap": shap_abs.astype(float),
    })
    out["shap_rank"] = out["mean_abs_shap"].rank(method="min", ascending=False)
    return out


def _single_feature_screen(dev: pd.DataFrame, features: list[str], threshold: float) -> pd.DataFrame:
    y = dev["fighter_win"].to_numpy(int)
    rows = []
    for c in features:
        s = pd.to_numeric(dev[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = s.notna().to_numpy()
        if mask.sum() < 200 or len(np.unique(y[mask])) < 2:
            continue
        try:
            auc = float(roc_auc_score(y[mask], s.to_numpy(float)[mask]))
        except ValueError:
            continue
        auc_strength = max(auc, 1.0 - auc)
        if auc_strength >= threshold:
            rows.append({"feature": c, "auc": auc, "auc_strength": auc_strength, "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("auc_strength", ascending=False) if rows else pd.DataFrame(columns=["feature", "auc", "auc_strength", "n"])


def run(config_path: Path = DEFAULT_CONFIG) -> None:
    config = _load_config(config_path)
    outputs = config["outputs"]
    bank = pd.read_parquet(outputs["prefight_feature_bank"])
    bank["event_date"] = pd.to_datetime(bank["event_date"], errors="raise").dt.normalize()
    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    dev = bank[bank["event_date"] < outer_start].copy()
    outer_rows = int((bank["event_date"] >= outer_start).sum())

    feature_cols = [c for c in bank.columns if c not in NON_FEATURE_COLUMNS]
    suspicious = _single_feature_screen(
        dev,
        feature_cols,
        float(config["validation"]["suspicious_single_feature_auc"]),
    )
    suspicious_path = Path(outputs["root"]) / "suspicious_single_feature_auc.csv"
    suspicious.to_csv(suspicious_path, index=False)
    if not suspicious.empty:
        print("WARNING: suspicious single-feature predictive strength found; review before interpretation")
        print(suspicious.head(20).to_string(index=False))

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []
    seed = int(config["model"]["random_seed"])

    for year in [int(x) for x in config["validation"]["development_years"]]:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        train = dev[dev["event_date"] < start].copy()
        valid = dev[(dev["event_date"] >= start) & (dev["event_date"] < end)].copy()
        train_fights = int(train["fight_id"].nunique())
        valid_fights = int(valid["fight_id"].nunique())
        if train_fights < int(config["validation"]["minimum_train_fights"]) or valid.empty:
            print(f"skip {year}: train fights={train_fights}, valid fights={valid_fights}")
            continue
        features = _usable_features(train, feature_cols)
        print(f"fold {year}: train fights={train_fights:,} valid fights={valid_fights:,} features={len(features):,}")

        y_valid = valid["fighter_win"].to_numpy(int)
        baseline = _baseline_metrics(valid)
        metric_rows.append({"fold": year, "model": "coinflip", "train_fights": train_fights, "valid_fights": valid_fights, "features": 0, **baseline})

        p_log = _fit_logistic(train, valid, features, config["model"]["logistic"])
        log_metrics = _metrics(y_valid, p_log)
        metric_rows.append({"fold": year, "model": "logistic", "train_fights": train_fights, "valid_fights": valid_fights, "features": len(features), **log_metrics})

        xgb_model, X_valid, p_xgb = _fit_xgb(train, valid, features, config["model"]["xgb"], seed + year)
        xgb_metrics = _metrics(y_valid, p_xgb)
        metric_rows.append({"fold": year, "model": "xgboost", "train_fights": train_fights, "valid_fights": valid_fights, "features": len(features), **xgb_metrics})

        pred = valid[["fight_id", "event_date", "fighter_id", "opponent_id", "fighter_win", "sample_weight"]].copy()
        pred["fold"] = year
        pred["p_logistic"] = p_log
        pred["p_xgboost"] = p_xgb
        prediction_rows.append(pred)
        importance_rows.append(_fold_importance(xgb_model, X_valid, year))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()

    if not importance.empty:
        stability = importance.groupby("feature", as_index=False).agg(
            folds=("fold", "nunique"),
            mean_gain=("gain", "mean"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            median_shap_rank=("shap_rank", "median"),
            top100_folds=("shap_rank", lambda s: int((s <= 100).sum())),
        )
        stability = stability.sort_values(["top100_folds", "mean_abs_shap"], ascending=[False, False])
    else:
        stability = pd.DataFrame()

    metrics.to_csv(outputs["development_metrics"], index=False)
    predictions.to_csv(outputs["development_predictions"], index=False)
    importance.to_csv(outputs["fold_feature_importance"], index=False)
    stability.to_csv(outputs["signal_stability"], index=False)

    summary = {
        "protocol": "development-only; 2024+ reserved",
        "outer_start": str(outer_start.date()),
        "outer_rows_reserved_not_scored": outer_rows,
        "development_rows": int(len(dev)),
        "development_fights": int(dev["fight_id"].nunique()),
        "candidate_features": int(len(feature_cols)),
        "suspicious_single_features": int(len(suspicious)),
        "folds_completed": sorted(metrics["fold"].unique().tolist()) if not metrics.empty else [],
        "mean_metrics_by_model": metrics.groupby("model")[["log_loss", "brier", "auc", "accuracy"]].mean().to_dict(orient="index") if not metrics.empty else {},
    }
    Path(outputs["research_summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDEVELOPMENT METRICS")
    print(metrics.to_string(index=False))
    if not stability.empty:
        print("\nTOP STABLE XGBOOST SIGNALS")
        print(stability.head(40).to_string(index=False))
    print(f"\n2024+ rows reserved and not scored: {outer_rows:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
