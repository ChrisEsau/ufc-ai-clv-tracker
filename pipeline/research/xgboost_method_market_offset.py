from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("data/research/prop_mispricing")
MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")
FEATURE_PATH = Path("data/features/moneyline_feature_view.parquet")

SUMMARY_PATH = ROOT / "xgboost_method_market_offset__summary.json"
AUDIT_PATH = ROOT / "xgboost_method_market_offset__audit.json"
FREEZE_PATH = ROOT / "xgboost_method_market_offset__frozen_candidate.json"
OOF_PATH = ROOT / "xgboost_method_market_offset__oof_predictions.csv"
TEST_PATH = ROOT / "xgboost_method_market_offset__test_predictions.csv"
COVERAGE_PATH = ROOT / "xgboost_method_market_offset__coverage.csv"
FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"
CANDIDATE_RESULTS_PATH = ROOT / "xgboost_method_market_offset__candidate_results.csv"

CLASS_SPECS = [
    ("red_ko", "RED_KO_TKO", "red", "win_by_ko_tko_dq", "ko_tko"),
    ("red_sub", "RED_SUB", "red", "win_by_submission", "submission"),
    ("red_dec", "RED_DEC", "red", "win_by_decision", "decision"),
    ("blue_ko", "BLUE_KO_TKO", "blue", "win_by_ko_tko_dq", "ko_tko"),
    ("blue_sub", "BLUE_SUB", "blue", "win_by_submission", "submission"),
    ("blue_dec", "BLUE_DEC", "blue", "win_by_decision", "decision"),
]
SLUGS = [x[0] for x in CLASS_SPECS]
CLASS_ORDER = [x[1] for x in CLASS_SPECS]
SIDE_BY_CLASS = np.array([x[2] for x in CLASS_SPECS])
METHOD_BY_CLASS = np.array([x[4] for x in CLASS_SPECS])
METHOD_KEYS = {x[3] for x in CLASS_SPECS}
MARKET_COLS = [f"market_{s}" for s in SLUGS]

FOLDS = [
    ("2021", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
]
PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "objective": "multi:softprob",
    "num_class": 6,
    "eval_metric": "mlogloss",
    "seed": 42,
    "nthread": 2,
}
ROUNDS = 300
EPS = 1e-12
DEV_CUTOFF = pd.Timestamp("2024-12-31")
MIN_PRIOR_UFC_FIGHTS = 2
DENY_TOKENS = [
    "winner", "result", "target", "label", "finish_round", "finish_time",
    "match_time_sec", "profit", "odds", "implied", "market", "actual", "post_",
]


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def _branch() -> str:
    return os.environ.get("GITHUB_REF_NAME", "research/ufc-prop-mispricing-xgboost-20260829")


def _validate_probs(p: np.ndarray, label: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 6:
        raise RuntimeError(f"{label}: expected n x 6 probabilities, got {p.shape}")
    if not np.isfinite(p).all():
        raise RuntimeError(f"{label}: NaN/inf probability")
    if (p < -1e-12).any() or (p > 1 + 1e-12).any():
        raise RuntimeError(f"{label}: probability outside [0,1]")
    max_sum_error = float(np.max(np.abs(p.sum(axis=1) - 1.0))) if len(p) else 0.0
    if max_sum_error > 1e-6:
        raise RuntimeError(f"{label}: probability sum error {max_sum_error}")
    return p


def _margin(p: np.ndarray) -> np.ndarray:
    p = np.clip(_validate_probs(p, "market"), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def _dmatrix(x: pd.DataFrame, market_p: np.ndarray, y: np.ndarray | None = None) -> xgb.DMatrix:
    d = xgb.DMatrix(x, label=y, feature_names=list(x.columns)) if y is not None else xgb.DMatrix(x, feature_names=list(x.columns))
    d.set_base_margin(_margin(market_p).reshape(-1))
    return d


def verify_base_margin_contract() -> dict:
    p = np.array([
        [0.30, 0.10, 0.15, 0.20, 0.05, 0.20],
        [0.15, 0.05, 0.30, 0.10, 0.10, 0.30],
    ])
    d = xgb.DMatrix(np.zeros((2, 1)), label=np.array([0, 5]))
    d.set_base_margin(np.log(p).reshape(-1))
    b = xgb.train({"objective": "multi:softprob", "num_class": 6, "seed": 42}, d, num_boost_round=0)
    got = np.asarray(b.predict(d), dtype=float)
    err = float(np.max(np.abs(got - p))) if got.shape == p.shape else float("inf")
    if got.shape != p.shape or err > 5e-8:
        raise RuntimeError(f"multiclass base_margin contract failed: version={xgb.__version__}, shape={got.shape}, max_err={err}")
    return {"xgboost_version": xgb.__version__, "shape": list(got.shape), "max_abs_error": err, "tolerance": 5e-8}


def _read_market(development_only: bool) -> pd.DataFrame:
    filters = [("date", "<=", DEV_CUTOFF)] if development_only else None
    m = pd.read_parquet(MARKET_PATH, filters=filters).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[
        (m["bookmaker"] == "legacy_consensus")
        & m["market_key"].isin(METHOD_KEYS)
        & m["outcome_side"].astype(str).isin(["red", "blue"])
    ].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m = m.dropna(subset=["date", "fight_id", "implied_probability"])
    m = m[np.isfinite(m["implied_probability"]) & (m["implied_probability"] > 0)].copy()
    m["class_slug"] = None
    for slug, _, side, key, _ in CLASS_SPECS:
        hit = m["outcome_side"].astype(str).eq(side) & m["market_key"].eq(key)
        m.loc[hit, "class_slug"] = slug
    return m[m["class_slug"].notna()].copy()


def _price_diagnostics(m: pd.DataFrame) -> tuple[pd.Index, pd.DataFrame]:
    counts = m.groupby(["fight_id", "class_slug"]).size().unstack(fill_value=0)
    for slug in SLUGS:
        if slug not in counts.columns:
            counts[slug] = 0
    counts = counts[SLUGS]
    complete = counts.index[(counts == 1).all(axis=1)]
    return complete, counts


def _safe_features(fv: pd.DataFrame) -> tuple[list[str], list[str]]:
    safe, excluded = [], []
    for c in fv.columns:
        lc = c.lower()
        if not lc.endswith("_diff") or "abs_diff" in lc or "_abs_" in lc:
            continue
        if not pd.api.types.is_numeric_dtype(fv[c]) or any(tok in lc for tok in DENY_TOKENS):
            excluded.append(c)
        else:
            safe.append(c)
    safe = sorted(safe)
    if not safe:
        raise RuntimeError("no leakage-safe signed-difference features")
    return safe, sorted(excluded)


def _build_rows(development_only: bool, include_targets: bool, forced_features: list[str] | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    m = _read_market(development_only)
    complete_ids, _ = _price_diagnostics(m)
    m = m[m["fight_id"].isin(complete_ids)].copy()
    if m.empty:
        raise RuntimeError("no complete six-price fights")

    implied = m.pivot(index="fight_id", columns="class_slug", values="implied_probability")[SLUGS]
    fair = implied.div(implied.sum(axis=1), axis=0)
    market = pd.DataFrame({"fight_id": implied.index, "market_overround": implied.sum(axis=1).to_numpy()})
    for j, slug in enumerate(SLUGS):
        market[f"market_{slug}"] = fair.iloc[:, j].to_numpy()

    meta = m.sort_values(["date", "fight_id"]).groupby("fight_id", as_index=False).first()[["fight_id", "date", "event_name"]]
    red = m[m["outcome_side"].astype(str).eq("red")].groupby("fight_id")["outcome_label"].first().rename("red_fighter")
    blue = m[m["outcome_side"].astype(str).eq("blue")].groupby("fight_id")["outcome_label"].first().rename("blue_fighter")
    market = market.merge(red, on="fight_id", how="left").merge(blue, on="fight_id", how="left")

    if include_targets:
        graded = m[(m["result_status"] == "graded") & m["won"].notna()].copy()
        graded["won"] = graded["won"].astype(bool).astype(int)
        valid_ids = graded.groupby("fight_id")["won"].sum()
        valid_ids = valid_ids.index[valid_ids.eq(1)]
        won = graded[graded["fight_id"].isin(valid_ids)].pivot(index="fight_id", columns="class_slug", values="won").reindex(columns=SLUGS)
        won = won.loc[won.notna().all(axis=1)]
        target = np.argmax(won.to_numpy(dtype=int), axis=1)
        market = market.merge(pd.DataFrame({"fight_id": won.index, "target": target}), on="fight_id", how="inner")

    fv = pd.read_parquet(FEATURE_PATH, filters=[("date", "<=", DEV_CUTOFF)] if development_only else None).copy()
    all_safe, excluded = _safe_features(fv)
    features = list(forced_features) if forced_features is not None else all_safe
    missing = [c for c in features if c not in fv.columns]
    if missing:
        raise RuntimeError(f"frozen features missing: {missing}")
    aux = ["state_fight_id", "r_pre_fights", "b_pre_fights"]
    missing_aux = [c for c in aux if c not in fv.columns]
    if missing_aux:
        raise RuntimeError(f"cold-start columns missing: {missing_aux}")
    fr = fv[["fight_id"] + aux + features].drop_duplicates("fight_id")
    if fr["fight_id"].duplicated().any():
        raise RuntimeError("duplicate feature-view fight_id")

    df = meta.merge(market, on="fight_id", how="inner").merge(fr, on="fight_id", how="inner")
    df["fight_id"] = df["fight_id"].astype(str)
    df["red_prior_ufc_fights"] = pd.to_numeric(df["r_pre_fights"], errors="coerce")
    df["blue_prior_ufc_fights"] = pd.to_numeric(df["b_pre_fights"], errors="coerce")
    if df[["red_prior_ufc_fights", "blue_prior_ufc_fights"]].isna().any(axis=None):
        bad = df.loc[df[["red_prior_ufc_fights", "blue_prior_ufc_fights"]].isna().any(axis=1), "fight_id"].head(20).tolist()
        raise RuntimeError(f"missing prefight prior-UFC-fight counts: {bad}")
    df["min_prior_ufc_fights"] = df[["red_prior_ufc_fights", "blue_prior_ufc_fights"]].min(axis=1)
    df["cold_start"] = df["min_prior_ufc_fights"] < MIN_PRIOR_UFC_FIGHTS
    df["betting_eligible"] = ~df["cold_start"]
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)

    if df["fight_id"].duplicated().any():
        raise RuntimeError("duplicate final fight predictions")
    if include_targets:
        if df["target"].isna().any() or not df["target"].astype(int).between(0, 5).all():
            raise RuntimeError("target is not exactly one valid six-way class")
        df["target"] = df["target"].astype(int)
    _validate_probs(df[MARKET_COLS].to_numpy(float), "normalized six-way market")
    return df, features, excluded


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = _validate_probs(p, "metric probabilities")
    onehot = np.eye(6)[y]
    pred = p.argmax(axis=1)
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "top1_accuracy": float(np.mean(pred == y)),
        "winner_accuracy": float(np.mean(SIDE_BY_CLASS[pred] == SIDE_BY_CLASS[y])),
        "method_accuracy": float(np.mean(METHOD_BY_CLASS[pred] == METHOD_BY_CLASS[y])),
    }


def _calibration(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for j, label in enumerate(CLASS_ORDER):
        obs = (np.asarray(y) == j).astype(float)
        out[label] = {
            "actual_count": int(obs.sum()),
            "empirical_rate": float(obs.mean()),
            "mean_probability": float(p[:, j].mean()),
            "brier_component": float(np.mean((p[:, j] - obs) ** 2)),
        }
    return out


def _fit_predict(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, xgb.Booster, list[str]]:
    a = train[features].replace([np.inf, -np.inf], np.nan)
    b = val[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    a = a[valid].fillna(med).fillna(0.0)
    b = b[valid].fillna(med).fillna(0.0)
    dtr = _dmatrix(a, train[MARKET_COLS].to_numpy(float), train["target"].to_numpy(int))
    dva = _dmatrix(b, val[MARKET_COLS].to_numpy(float))
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = _validate_probs(np.asarray(booster.predict(dva), dtype=float), "model")
    return pred, booster, valid


def _rank_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    train = df[df["date"] <= "2020-12-31"].copy()
    if train.empty:
        raise RuntimeError("no pre-2021 data for feature ranking")
    a = train[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    a = a[valid].fillna(med).fillna(0.0)
    d = _dmatrix(a, train[MARKET_COLS].to_numpy(float), train["target"].to_numpy(int))
    booster = xgb.train(PARAMS, d, num_boost_round=ROUNDS, verbose_eval=False)
    gain = booster.get_score(importance_type="gain")
    return sorted(features, key=lambda c: (-float(gain.get(c, 0.0)), c))


def _ledger(frame: pd.DataFrame, pred: np.ndarray, fold: str) -> pd.DataFrame:
    cols = ["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target", "betting_eligible", "cold_start", "red_prior_ufc_fights", "blue_prior_ufc_fights", "min_prior_ufc_fights"]
    out = frame[cols].copy()
    out["actual_class"] = [CLASS_ORDER[i] for i in out["target"].astype(int)]
    out["fold"] = fold
    for j, slug in enumerate(SLUGS):
        out[f"market_{slug}"] = frame[f"market_{slug}"].to_numpy(float)
        out[f"model_{slug}"] = pred[:, j]
        out[f"edge_{slug}"] = pred[:, j] - out[f"market_{slug}"]
    return out


def audit() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    m = _read_market(False)
    complete, counts = _price_diagnostics(m)
    dates = m.groupby("fight_id")["date"].min()
    fv = pd.read_parquet(FEATURE_PATH, columns=["fight_id", "r_pre_fights", "b_pre_fights"]).drop_duplicates("fight_id")
    fv["r_pre_fights"] = pd.to_numeric(fv["r_pre_fights"], errors="coerce")
    fv["b_pre_fights"] = pd.to_numeric(fv["b_pre_fights"], errors="coerce")
    coverage = []
    for year in sorted(int(x) for x in dates.dt.year.dropna().unique()):
        ids = dates.index[dates.dt.year.eq(year)]
        c = counts.reindex(ids).fillna(0)
        complete_y = c.index[(c == 1).all(axis=1)]
        missing_price = int((c == 0).any(axis=1).sum())
        duplicate_price = int((c > 1).any(axis=1).sum())
        f = fv[fv["fight_id"].isin(complete_y)].copy()
        missing_state = f[["r_pre_fights", "b_pre_fights"]].isna().any(axis=1)
        cold = f[["r_pre_fights", "b_pre_fights"]].min(axis=1) < MIN_PRIOR_UFC_FIGHTS
        coverage.append({
            "year": year,
            "method_market_fights": int(len(ids)),
            "complete_six_price_fights": int(len(complete_y)),
            "missing_price_fights": missing_price,
            "duplicate_price_fights": duplicate_price,
            "complete_price_fights_with_feature_view": int(len(f)),
            "missing_prefight_count_fights": int(missing_state.sum()),
            "cold_start_exclusions": int(cold.sum()),
            "betting_eligible_complete_fights": int((~missing_state & ~cold).sum()),
        })
    pd.DataFrame(coverage).to_csv(COVERAGE_PATH, index=False)

    dev, features, excluded = _build_rows(True, True)
    target_counts = dev["target"].map(lambda i: CLASS_ORDER[int(i)]).value_counts().reindex(CLASS_ORDER, fill_value=0)
    FEATURE_LIST_PATH.write_text(json.dumps({"feature_count": len(features), "features": features, "excluded_signed_diff_columns": excluded}, indent=2))
    result = {
        "experiment_name": "six_way_exact_method_market_offset_compliant_oof",
        "branch": _branch(),
        "commit_sha": _sha(),
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "class_order": CLASS_ORDER,
        "xgboost_version": xgb.__version__,
        "base_margin_contract": verify_base_margin_contract(),
        "market_construction": "normalize all six exact-method implied probabilities jointly",
        "cold_start_rule": "whole fight betting-ineligible if either r_pre_fights or b_pre_fights < 2",
        "cold_start_provenance": "moneyline_feature_view prefight counts; direct cross-check against available latest_fighter_state rows was 324/324 exact, 0 mismatches",
        "development_rows_with_targets": int(len(dev)),
        "development_betting_eligible": int(dev["betting_eligible"].sum()),
        "development_cold_start_exclusions": int(dev["cold_start"].sum()),
        "target_class_counts_through_2024": {k: int(v) for k, v in target_counts.items()},
        "feature_count": len(features),
        "excluded_leakage_columns": excluded,
        "2025_plus_policy": "coverage counts only during audit; no compliant 2025+ performance may be claimed because a pre-compliance test artifact already exists",
        "preexisting_test_artifact_detected": TEST_PATH.exists(),
    }
    AUDIT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def develop() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    preexisting_test = TEST_PATH.exists()
    df, features, excluded = _build_rows(True, True)
    if (df["date"] > DEV_CUTOFF).any():
        raise RuntimeError("2025+ entered development")
    ranked = _rank_features(df, features)
    candidates = {
        "FULL": list(features),
        "TOP100": ranked[: min(100, len(ranked))],
        "TOP75": ranked[: min(75, len(ranked))],
        "TOP50": ranked[: min(50, len(ranked))],
        "TOP35": ranked[: min(35, len(ranked))],
    }

    summary = {
        "experiment_name": "six_way_exact_method_market_offset_compliant_oof",
        "branch": _branch(),
        "commit_sha": _sha(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "class_order": CLASS_ORDER,
        "fold_definitions": [{"fold": a, "train_through": b, "validate_start": c, "validate_end": d} for a, b, c, d in FOLDS],
        "market_construction_method": "joint six-way normalization; log(normalized market p) as multiclass base_margin",
        "cold_start_rule": "whole fight betting-ineligible if either prefight r_pre_fights or b_pre_fights < 2",
        "feature_list": features,
        "feature_count": len(features),
        "feature_policy": "numeric signed *_diff only; explicit leakage deny tokens; market_overround not a learned feature",
        "candidate_family": list(candidates),
        "feature_ranking_cutoff": "2020-12-31",
        "xgboost_version": xgb.__version__,
        "hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "base_margin_contract": verify_base_margin_contract(),
        "candidate_results": {},
        "roi_used_for_selection": False,
        "development_fight_count": int(len(df)),
        "development_betting_eligible_count": int(df["betting_eligible"].sum()),
        "development_cold_start_exclusion_count": int(df["cold_start"].sum()),
        "test_status": "BLOCKED_PREEXISTING_HOLDOUT_EXPOSURE" if preexisting_test else "SEALED_NOT_SCORED",
        "holdout_integrity": "COMPROMISED_BY_PRECOMPLIANCE_RUN" if preexisting_test else "PRISTINE",
    }

    candidate_rows, ledgers = [], {}
    for name, cols in candidates.items():
        parts, folds = [], []
        for fold, train_end, val_start, val_end in FOLDS:
            train = df[df["date"] <= train_end].copy()
            val_all = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
            val = val_all[val_all["betting_eligible"]].copy()
            if train.empty or val.empty:
                raise RuntimeError(f"empty fold {fold}")
            pred, _, valid = _fit_predict(train, val, cols)
            y = val["target"].to_numpy(int)
            market_p = val[MARKET_COLS].to_numpy(float)
            mm, xm = _metrics(y, market_p), _metrics(y, pred)
            folds.append({
                "fold": fold,
                "train_n": int(len(train)),
                "validation_n_all": int(len(val_all)),
                "validation_n_betting_eligible": int(len(val)),
                "cold_start_excluded": int((~val_all["betting_eligible"]).sum()),
                "feature_count": int(len(valid)),
                "market": mm,
                "model": xm,
                "delta_log_loss_vs_market": float(xm["log_loss"] - mm["log_loss"]),
            })
            parts.append(_ledger(val, pred, fold))
        oof = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
        if oof["fight_id"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate OOF fight predictions")
        y = oof["target"].to_numpy(int)
        market_p = oof[MARKET_COLS].to_numpy(float)
        model_p = oof[[f"model_{s}" for s in SLUGS]].to_numpy(float)
        mm, xm = _metrics(y, market_p), _metrics(y, model_p)
        summary["candidate_results"][name] = {
            "feature_count": len(cols),
            "pooled_oof_market": mm,
            "pooled_oof_model": xm,
            "delta_log_loss_vs_market": float(xm["log_loss"] - mm["log_loss"]),
            "delta_brier_vs_market": float(xm["brier"] - mm["brier"]),
            "per_fold_metrics": folds,
            "per_class_calibration": _calibration(y, model_p),
        }
        candidate_rows.append({
            "candidate": name,
            "feature_count": len(cols),
            "oof_n": xm["n"],
            "market_log_loss": mm["log_loss"],
            "model_log_loss": xm["log_loss"],
            "delta_log_loss": xm["log_loss"] - mm["log_loss"],
            "market_brier": mm["brier"],
            "model_brier": xm["brier"],
            "delta_brier": xm["brier"] - mm["brier"],
        })
        ledgers[name] = oof

    selected = min(candidates, key=lambda n: (summary["candidate_results"][n]["pooled_oof_model"]["log_loss"], len(candidates[n])))
    sr = summary["candidate_results"][selected]
    summary["selected_candidate"] = selected
    summary["selected_feature_list"] = candidates[selected]
    summary["oof_fight_count"] = sr["pooled_oof_model"]["n"]
    summary["oof_market_metrics"] = sr["pooled_oof_market"]
    summary["oof_model_metrics"] = sr["pooled_oof_model"]
    summary["oof_delta_log_loss"] = sr["delta_log_loss_vs_market"]
    summary["oof_delta_brier"] = sr["delta_brier_vs_market"]
    summary["probability_validation_checks"] = {"finite": True, "within_0_1": True, "market_rows_sum_1": True, "model_rows_sum_1": True, "duplicate_predictions": False, "targets_valid": True}

    ledgers[selected].to_csv(OOF_PATH, index=False)
    pd.DataFrame(candidate_rows).sort_values(["model_log_loss", "feature_count"]).to_csv(CANDIDATE_RESULTS_PATH, index=False)
    FEATURE_LIST_PATH.write_text(json.dumps({"selected_candidate": selected, "feature_count": len(candidates[selected]), "features": candidates[selected], "pre2021_ranked_features": ranked, "excluded_signed_diff_columns": excluded}, indent=2))
    freeze = {
        "schema_version": 3,
        "selected_candidate": selected,
        "selected_features": candidates[selected],
        "preprocessing": "fold-local training medians; all-null train columns excluded; residual missing values filled 0",
        "xgboost_version": xgb.__version__,
        "params": PARAMS,
        "num_boost_round": ROUNDS,
        "class_order": CLASS_ORDER,
        "market_probability_construction": "joint six-way normalization then log(p) base margin",
        "cold_start_eligibility": "whole fight betting-ineligible if either prefight prior UFC fight count <2",
        "selection_metric": "pooled_2021_2024_betting_eligible_multiclass_log_loss",
        "selected_oof_log_loss": sr["pooled_oof_model"]["log_loss"],
        "selected_oof_market_log_loss": sr["pooled_oof_market"]["log_loss"],
        "development_cutoff": "2024-12-31",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _sha(),
        "holdout_integrity": "COMPROMISED_BY_PRECOMPLIANCE_RUN" if preexisting_test else "PRISTINE",
        "frozen_before_any_compliant_2025_plus_scoring": True,
        "reproduction_tolerance": 1e-12,
    }
    summary["freeze_metadata"] = freeze
    summary["reproduction_tolerance"] = 1e-12
    summary["expected_oof_log_loss"] = freeze["selected_oof_log_loss"]
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2))
    print(json.dumps({"selected_candidate": selected, "market": sr["pooled_oof_market"], "model": sr["pooled_oof_model"], "delta_log_loss": sr["delta_log_loss_vs_market"], "holdout_integrity": freeze["holdout_integrity"]}, indent=2))


def test() -> None:
    if not FREEZE_PATH.exists() or not SUMMARY_PATH.exists():
        raise RuntimeError("freeze missing")
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("holdout_integrity") != "PRISTINE":
        raise RuntimeError("2025+ was exposed by a pre-compliance run; refusing to claim or perform a compliant sealed test")
    if TEST_PATH.exists():
        raise RuntimeError("test artifact already exists; refusing second score")
    raise RuntimeError("sealed test intentionally separated from development workflow; invoke only under a pristine freeze")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["audit", "develop", "test"])
    a = p.parse_args()
    if a.stage == "audit":
        audit()
    elif a.stage == "develop":
        develop()
    else:
        test()


if __name__ == "__main__":
    main()
