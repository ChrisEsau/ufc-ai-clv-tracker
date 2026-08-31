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
STATE_PATH = Path("data/features/latest_fighter_state.parquet")

SUMMARY_PATH = ROOT / "xgboost_method_market_offset__summary.json"
FREEZE_PATH = ROOT / "xgboost_method_market_offset__frozen_candidate.json"
OOF_PATH = ROOT / "xgboost_method_market_offset__oof_predictions.csv"
TEST_PATH = ROOT / "xgboost_method_market_offset__test_predictions.csv"
COVERAGE_PATH = ROOT / "xgboost_method_market_offset__coverage.csv"
FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"
CANDIDATE_RESULTS_PATH = ROOT / "xgboost_method_market_offset__candidate_results.csv"

CLASS_SPECS = [
    ("red_ko", "red", "win_by_ko_tko_dq", "ko_tko"),
    ("red_sub", "red", "win_by_submission", "submission"),
    ("red_dec", "red", "win_by_decision", "decision"),
    ("blue_ko", "blue", "win_by_ko_tko_dq", "ko_tko"),
    ("blue_sub", "blue", "win_by_submission", "submission"),
    ("blue_dec", "blue", "win_by_decision", "decision"),
]
CLASS_NAMES = [x[0] for x in CLASS_SPECS]
SIDE_BY_CLASS = np.array([x[1] for x in CLASS_SPECS])
METHOD_BY_CLASS = np.array([x[3] for x in CLASS_SPECS])
METHOD_KEYS = {x[2] for x in CLASS_SPECS}
MKT_COLS = [f"market_{c}" for c in CLASS_NAMES]

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
COLD_START_MIN_PRIOR_UFC_FIGHTS = 2
DENY_TOKENS = [
    "winner", "result", "target", "label", "finish_round", "finish_time",
    "match_time_sec", "profit", "odds", "implied", "market", "actual", "post_",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def _branch() -> str:
    return os.environ.get("GITHUB_REF_NAME", "research/ufc-prop-mispricing-xgboost-20260829")


def _validate_probs(p: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 6:
        raise RuntimeError(f"{name}: expected n x 6, got {p.shape}")
    if not np.isfinite(p).all():
        raise RuntimeError(f"{name}: non-finite probability")
    if (p < -1e-12).any() or (p > 1 + 1e-12).any():
        raise RuntimeError(f"{name}: probability outside [0,1]")
    err = float(np.max(np.abs(p.sum(axis=1) - 1.0))) if len(p) else 0.0
    if err > 1e-6:
        raise RuntimeError(f"{name}: row sum error {err}")
    return p


def _market_margin(p: np.ndarray) -> np.ndarray:
    p = np.clip(_validate_probs(p, "market probabilities"), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def _dmatrix(x: pd.DataFrame, p_market: np.ndarray, y: np.ndarray | None = None) -> xgb.DMatrix:
    d = xgb.DMatrix(x, label=y, feature_names=list(x.columns)) if y is not None else xgb.DMatrix(x, feature_names=list(x.columns))
    d.set_base_margin(_market_margin(p_market).reshape(-1))
    return d


def verify_base_margin_contract() -> dict:
    p = np.array([
        [0.30, 0.10, 0.15, 0.20, 0.05, 0.20],
        [0.15, 0.05, 0.30, 0.10, 0.10, 0.30],
    ], dtype=float)
    d = xgb.DMatrix(np.zeros((2, 1)), label=np.array([0, 5]))
    d.set_base_margin(np.log(p).reshape(-1))
    b = xgb.train({"objective": "multi:softprob", "num_class": 6, "seed": 42}, d, num_boost_round=0)
    got = np.asarray(b.predict(d), dtype=float)
    err = float(np.max(np.abs(got - p))) if got.shape == p.shape else float("inf")
    if got.shape != p.shape or err > 5e-8:
        raise RuntimeError(f"multiclass base_margin contract failed: xgboost={xgb.__version__}, shape={got.shape}, max_err={err}")
    return {"xgboost_version": xgb.__version__, "shape": list(got.shape), "max_abs_error": err, "tolerance": 5e-8}


def _read_market(develop_only: bool) -> pd.DataFrame:
    filters = [("date", "<=", DEV_CUTOFF)] if develop_only else None
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
    m["method_class"] = None
    for cname, side, key, _ in CLASS_SPECS:
        hit = m["outcome_side"].astype(str).eq(side) & m["market_key"].eq(key)
        m.loc[hit, "method_class"] = cname
    return m[m["method_class"].notna()].copy()


def _complete_price_ids(m: pd.DataFrame) -> tuple[pd.Index, pd.DataFrame]:
    counts = m.groupby(["fight_id", "method_class"]).size().unstack(fill_value=0)
    for cname in CLASS_NAMES:
        if cname not in counts.columns:
            counts[cname] = 0
    counts = counts[CLASS_NAMES]
    complete = counts.index[(counts == 1).all(axis=1)]
    return complete, counts


def _cold_start_flags(fight_ids: pd.Series | list[str], develop_only: bool) -> pd.DataFrame:
    st = pd.read_parquet(STATE_PATH).copy()
    st["date"] = pd.to_datetime(st["date"], errors="coerce")
    if develop_only:
        st = st[st["date"] <= DEV_CUTOFF].copy()
    st["fight_id"] = st["fight_id"].astype(str)
    st = st[st["fight_id"].isin(set(map(str, fight_ids)))].copy()
    st["fights"] = pd.to_numeric(st["fights"], errors="coerce")
    rows = []
    for (fid, dt), g in st.groupby(["fight_id", "date"], dropna=False):
        vals = g["fights"]
        if vals.isna().any() or len(vals) != 2:
            continue
        red_hit = g["corner"].astype(str).str.lower().eq("red")
        blue_hit = g["corner"].astype(str).str.lower().eq("blue")
        rows.append({
            "fight_id": str(fid),
            "state_date": dt,
            "red_prior_ufc_fights": float(g.loc[red_hit, "fights"].iloc[0]) if red_hit.sum() == 1 else np.nan,
            "blue_prior_ufc_fights": float(g.loc[blue_hit, "fights"].iloc[0]) if blue_hit.sum() == 1 else np.nan,
            "min_prior_ufc_fights": float(vals.min()),
            "cold_start": bool((vals < COLD_START_MIN_PRIOR_UFC_FIGHTS).any()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no cold-start fighter-state rows found")
    out["betting_eligible"] = ~out["cold_start"]
    if out["fight_id"].duplicated().any():
        raise RuntimeError("duplicate cold-start fight state")
    return out


def _safe_features(fv: pd.DataFrame) -> tuple[list[str], list[str]]:
    safe, excluded = [], []
    for c in fv.columns:
        lc = c.lower()
        if not lc.endswith("_diff") or "abs_diff" in lc or "_abs_" in lc:
            continue
        reason = None
        if not pd.api.types.is_numeric_dtype(fv[c]):
            reason = "non_numeric"
        elif any(tok in lc for tok in DENY_TOKENS):
            reason = "leakage_deny_token"
        if reason:
            excluded.append(c)
        else:
            safe.append(c)
    safe = sorted(safe)
    if not safe:
        raise RuntimeError("no leakage-safe signed-difference features")
    return safe, sorted(excluded)


def _build_rows(develop_only: bool, include_targets: bool, forced_features: list[str] | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    m = _read_market(develop_only)
    complete_ids, _ = _complete_price_ids(m)
    m = m[m["fight_id"].isin(complete_ids)].copy()
    if m.empty:
        raise RuntimeError("no complete six-price fights")

    implied = m.pivot(index="fight_id", columns="method_class", values="implied_probability")[CLASS_NAMES]
    overround = implied.sum(axis=1)
    fair = implied.div(overround, axis=0)
    market_frame = pd.DataFrame({"fight_id": implied.index, "market_overround": overround.to_numpy()})
    for j, cname in enumerate(CLASS_NAMES):
        market_frame[f"market_{cname}"] = fair.iloc[:, j].to_numpy()

    meta = m.sort_values(["date", "fight_id"]).groupby("fight_id", as_index=False).first()[["fight_id", "date", "event_name"]]
    red_name = m[m["outcome_side"].astype(str).eq("red")].groupby("fight_id")["outcome_label"].first().rename("red_fighter")
    blue_name = m[m["outcome_side"].astype(str).eq("blue")].groupby("fight_id")["outcome_label"].first().rename("blue_fighter")
    market_frame = market_frame.merge(red_name, on="fight_id", how="left").merge(blue_name, on="fight_id", how="left")

    if include_targets:
        graded = m[(m["result_status"] == "graded") & m["won"].notna()].copy()
        graded["won"] = graded["won"].astype(bool).astype(int)
        target_counts = graded.groupby("fight_id")["won"].sum()
        valid_target_ids = target_counts.index[target_counts.eq(1)]
        graded = graded[graded["fight_id"].isin(valid_target_ids)]
        won = graded.pivot(index="fight_id", columns="method_class", values="won").reindex(columns=CLASS_NAMES)
        good = won.index[won.notna().all(axis=1)]
        won = won.loc[good]
        target = np.argmax(won.to_numpy(dtype=int), axis=1)
        market_frame = market_frame.merge(pd.DataFrame({"fight_id": won.index, "target": target}), on="fight_id", how="inner")

    fv = pd.read_parquet(FEATURE_PATH, filters=[("date", "<=", DEV_CUTOFF)] if develop_only else None).copy()
    all_safe, excluded = _safe_features(fv)
    features = list(forced_features) if forced_features is not None else all_safe
    missing = [c for c in features if c not in fv.columns]
    if missing:
        raise RuntimeError(f"frozen features missing: {missing}")
    feature_rows = fv[["fight_id"] + features].drop_duplicates("fight_id")
    if feature_rows["fight_id"].duplicated().any():
        raise RuntimeError("duplicate feature-view fight_id")

    df = meta.merge(market_frame, on="fight_id", how="inner").merge(feature_rows, on="fight_id", how="inner")
    df["fight_id"] = df["fight_id"].astype(str)
    flags = _cold_start_flags(df["fight_id"], develop_only)
    df = df.merge(flags.drop(columns=["state_date"]), on="fight_id", how="left")
    if df["betting_eligible"].isna().any():
        miss = df.loc[df["betting_eligible"].isna(), "fight_id"].head(20).tolist()
        raise RuntimeError(f"missing authoritative cold-start state for fights: {miss}")
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)

    if df["fight_id"].duplicated().any():
        raise RuntimeError("duplicate final fight rows")
    if include_targets:
        if df["target"].isna().any() or not df["target"].astype(int).between(0, 5).all():
            raise RuntimeError("invalid six-way target")
        df["target"] = df["target"].astype(int)
    _validate_probs(df[MKT_COLS].to_numpy(float), "normalized six-way market")
    return df, features, excluded


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = _validate_probs(p, "metric probabilities")
    chosen = np.clip(p[np.arange(len(y)), y], EPS, 1.0)
    onehot = np.eye(6)[y]
    pred = p.argmax(axis=1)
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(np.log(chosen))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "top1_accuracy": float(np.mean(pred == y)),
        "winner_accuracy": float(np.mean(SIDE_BY_CLASS[pred] == SIDE_BY_CLASS[y])),
        "method_accuracy": float(np.mean(METHOD_BY_CLASS[pred] == METHOD_BY_CLASS[y])),
    }


def _class_calibration(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for j, cname in enumerate(CLASS_NAMES):
        obs = (np.asarray(y) == j).astype(float)
        out[cname] = {
            "n": int(len(obs)),
            "actual_count": int(obs.sum()),
            "empirical_rate": float(obs.mean()),
            "mean_probability": float(p[:, j].mean()),
            "brier_component": float(np.mean((p[:, j] - obs) ** 2)),
        }
    return out


def _fit_predict(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, xgb.Booster, list[str]]:
    raw_train = train[features].replace([np.inf, -np.inf], np.nan)
    raw_val = val[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if raw_train[c].notna().any()]
    med = raw_train[valid].median(numeric_only=True)
    xtr = raw_train[valid].fillna(med).fillna(0.0)
    xva = raw_val[valid].fillna(med).fillna(0.0)
    dtr = _dmatrix(xtr, train[MKT_COLS].to_numpy(float), train["target"].to_numpy(int))
    dva = _dmatrix(xva, val[MKT_COLS].to_numpy(float))
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = _validate_probs(np.asarray(booster.predict(dva), dtype=float), "model prediction")
    return pred, booster, valid


def _rank_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    train = df[df["date"] <= "2020-12-31"].copy()
    if train.empty:
        raise RuntimeError("no pre-2021 rows for feature ranking")
    _, booster, _ = _fit_predict(train, train.iloc[:1].copy(), features)
    gain = booster.get_score(importance_type="gain")
    return sorted(features, key=lambda c: (-float(gain.get(c, 0.0)), c))


def _prediction_rows(frame: pd.DataFrame, p: np.ndarray, fold: str | None = None) -> pd.DataFrame:
    cols = ["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
            "betting_eligible", "cold_start", "red_prior_ufc_fights", "blue_prior_ufc_fights", "min_prior_ufc_fights"]
    out = frame[cols].copy()
    out["actual_class"] = [CLASS_NAMES[i] for i in out["target"].astype(int)]
    if fold is not None:
        out["fold"] = fold
    for j, cname in enumerate(CLASS_NAMES):
        out[f"market_{cname}"] = frame[f"market_{cname}"].to_numpy(float)
        out[f"model_{cname}"] = p[:, j]
        out[f"edge_{cname}"] = p[:, j] - out[f"market_{cname}"]
    return out


def audit() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    m = _read_market(False)
    complete_ids, counts = _complete_price_ids(m)
    fight_dates = m.groupby("fight_id")["date"].min()
    rows = []
    years = sorted(int(y) for y in fight_dates.dt.year.dropna().unique())
    for year in years:
        ids = fight_dates.index[fight_dates.dt.year.eq(year)]
        sub = counts.reindex(ids).fillna(0)
        complete = sub.index[(sub[CLASS_NAMES] == 1).all(axis=1)]
        flags = _cold_start_flags(list(map(str, complete)), False) if len(complete) else pd.DataFrame()
        rows.append({
            "year": year,
            "market_fights": int(len(ids)),
            "complete_six_price_fights": int(len(complete)),
            "duplicate_or_missing_six_price_fights": int(len(ids) - len(complete)),
            "cold_start_exclusions": int(flags["cold_start"].sum()) if len(flags) else 0,
            "betting_eligible_complete_fights": int(flags["betting_eligible"].sum()) if len(flags) else 0,
        })
    pd.DataFrame(rows).to_csv(COVERAGE_PATH, index=False)

    dev, features, excluded = _build_rows(True, include_targets=True)
    target_counts = dev["target"].map(lambda i: CLASS_NAMES[int(i)]).value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()
    FEATURE_LIST_PATH.write_text(json.dumps({
        "feature_policy": "numeric signed *_diff features with explicit leakage deny tokens; no market_overround learned feature",
        "feature_count": len(features),
        "features": features,
        "excluded_signed_diff_columns": excluded,
    }, indent=2), encoding="utf-8")
    audit_summary = {
        "experiment_name": "six_way_exact_method_market_offset_v2_compliance",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": _branch(),
        "commit_sha": _git_sha(),
        "class_order": CLASS_NAMES,
        "xgboost_version": xgb.__version__,
        "base_margin_contract": verify_base_margin_contract(),
        "cold_start_rule": "betting eligible iff both fighters have >=2 prior UFC fights at prefight latest_fighter_state.fights; whole fight excluded otherwise",
        "development_rows_with_targets": int(len(dev)),
        "development_betting_eligible": int(dev["betting_eligible"].sum()),
        "development_cold_start_exclusions": int(dev["cold_start"].sum()),
        "target_class_counts_through_2024": {k: int(v) for k, v in target_counts.items()},
        "feature_count": len(features),
        "excluded_leakage_columns": excluded,
        "2025_plus_audit_policy": "coverage counts only; no 2025+ target/model performance inspected",
        "preexisting_test_artifact_detected": TEST_PATH.exists(),
    }
    (ROOT / "xgboost_method_market_offset__audit.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))


def develop() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    preexisting_test = TEST_PATH.exists()
    df, features, excluded = _build_rows(True, include_targets=True)
    if (df["date"] > DEV_CUTOFF).any():
        raise RuntimeError("sealed 2025+ row entered development")
    ranked = _rank_features(df, features)
    candidates = {"full": list(features)}
    for n in [100, 75, 50, 35]:
        candidates[f"top_{n}"] = ranked[: min(n, len(ranked))]

    summary = {
        "experiment_name": "six_way_exact_method_market_offset_v2_compliance",
        "branch": _branch(),
        "commit_sha": _git_sha(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "class_order": CLASS_NAMES,
        "date_ranges": {"development_max": "2024-12-31", "oof": "2021-01-01..2024-12-31"},
        "fold_definitions": [{"fold": a, "train_through": b, "validate_start": c, "validate_end": d} for a, b, c, d in FOLDS],
        "market_construction": "six raw exact-method implied probabilities normalized jointly within fight; log(normalized p) supplied as multiclass base_margin",
        "cold_start_policy": {
            "source": "data/features/latest_fighter_state.parquet:fights",
            "rule": "whole fight betting-ineligible if either fighter has <2 prior UFC fights",
            "selection_metrics_use": "OOF candidate selection and headline metrics restricted to betting-eligible validation fights; training retains all historical target rows",
        },
        "feature_policy": "leakage-screened numeric signed *_diff family; no market_overround learned feature",
        "feature_count": len(features),
        "feature_list": features,
        "excluded_leakage_columns": excluded,
        "candidate_family": list(candidates),
        "feature_ranking": "gain learned only from data through 2020-12-31",
        "xgboost_version": xgb.__version__,
        "hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "base_margin_contract": verify_base_margin_contract(),
        "candidate_results": {},
        "development_fight_count": int(len(df)),
        "development_betting_eligible_count": int(df["betting_eligible"].sum()),
        "development_cold_start_exclusion_count": int(df["cold_start"].sum()),
        "roi_used_for_selection": False,
        "test_status": "BLOCKED_PREEXISTING_HOLDOUT_EXPOSURE" if preexisting_test else "SEALED_NOT_SCORED",
        "holdout_integrity": {
            "preexisting_test_artifact_detected_before_compliant_freeze": bool(preexisting_test),
            "scientific_status": "COMPROMISED_BY_PRECOMPLIANCE_RUN" if preexisting_test else "PRISTINE",
        },
    }

    candidate_rows = []
    all_oof = {}
    for cname, cols in candidates.items():
        fold_summaries, pred_parts = [], []
        for fold, train_end, val_start, val_end in FOLDS:
            train = df[df["date"] <= train_end].copy()
            val_all = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
            val = val_all[val_all["betting_eligible"]].copy()
            if train.empty or val.empty:
                raise RuntimeError(f"empty train/eligible-validation for fold {fold}: {len(train)}/{len(val)}")
            pred, _, valid = _fit_predict(train, val, cols)
            y = val["target"].to_numpy(int)
            market_p = val[MKT_COLS].to_numpy(float)
            mm, mx = _metrics(y, market_p), _metrics(y, pred)
            fold_summaries.append({
                "fold": fold,
                "train_n_all": int(len(train)),
                "validation_n_all": int(len(val_all)),
                "validation_n_betting_eligible": int(len(val)),
                "cold_start_excluded_from_scoring": int((~val_all["betting_eligible"]).sum()),
                "feature_count": int(len(valid)),
                "market": mm, "model": mx,
                "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
                "delta_brier_vs_market": float(mx["brier"] - mm["brier"]),
            })
            pred_parts.append(_prediction_rows(val, pred, fold))
        oof = pd.concat(pred_parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
        if oof["fight_id"].duplicated().any():
            raise RuntimeError(f"{cname}: duplicate OOF fight predictions")
        y = oof["target"].to_numpy(int)
        market_p = oof[[f"market_{c}" for c in CLASS_NAMES]].to_numpy(float)
        model_p = oof[[f"model_{c}" for c in CLASS_NAMES]].to_numpy(float)
        mm, mx = _metrics(y, market_p), _metrics(y, model_p)
        summary["candidate_results"][cname] = {
            "feature_count": len(cols),
            "pooled_oof_market": mm,
            "pooled_oof_model": mx,
            "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
            "delta_brier_vs_market": float(mx["brier"] - mm["brier"]),
            "folds": fold_summaries,
            "per_class_calibration": _class_calibration(y, model_p),
        }
        candidate_rows.append({
            "candidate": cname, "feature_count": len(cols), "oof_n": mx["n"],
            "market_log_loss": mm["log_loss"], "model_log_loss": mx["log_loss"],
            "delta_log_loss": mx["log_loss"] - mm["log_loss"],
            "market_brier": mm["brier"], "model_brier": mx["brier"],
            "delta_brier": mx["brier"] - mm["brier"],
        })
        all_oof[cname] = oof

    selected = min(candidates, key=lambda k: (summary["candidate_results"][k]["pooled_oof_model"]["log_loss"], len(candidates[k])))
    selected_features = candidates[selected]
    selected_result = summary["candidate_results"][selected]
    summary["selected_candidate"] = selected
    summary["selected_features"] = selected_features
    summary["oof_fight_count"] = selected_result["pooled_oof_model"]["n"]
    summary["oof_market_metrics"] = selected_result["pooled_oof_market"]
    summary["oof_model_metrics"] = selected_result["pooled_oof_model"]
    summary["oof_delta_log_loss"] = selected_result["delta_log_loss_vs_market"]
    summary["oof_delta_brier"] = selected_result["delta_brier_vs_market"]

    all_oof[selected].to_csv(OOF_PATH, index=False)
    pd.DataFrame(candidate_rows).sort_values(["model_log_loss", "feature_count"]).to_csv(CANDIDATE_RESULTS_PATH, index=False)
    FEATURE_LIST_PATH.write_text(json.dumps({
        "selected_candidate": selected,
        "feature_count": len(selected_features),
        "features": selected_features,
        "pre2021_ranked_features": ranked,
        "excluded_signed_diff_columns": excluded,
    }, indent=2), encoding="utf-8")

    freeze = {
        "schema_version": 2,
        "frozen_before_2025_plus_scoring": not preexisting_test,
        "holdout_integrity": "COMPROMISED_BY_PRECOMPLIANCE_RUN" if preexisting_test else "PRISTINE",
        "selected_candidate": selected,
        "selected_features": selected_features,
        "class_order": CLASS_NAMES,
        "params": PARAMS,
        "num_boost_round": ROUNDS,
        "preprocessing": "fold-local training medians; remaining all-null train features excluded; residual NaN filled 0",
        "market_probability_construction": "joint six-way normalization; log(p_market) multiclass base_margin",
        "cold_start_eligibility": "whole fight betting-ineligible if either prefight latest_fighter_state.fights < 2; OOF scoring/selection eligible only",
        "selection_metric": "pooled_2021_2024_betting_eligible_multiclass_log_loss",
        "selected_oof_log_loss": summary["oof_model_metrics"]["log_loss"],
        "selected_oof_market_log_loss": summary["oof_market_metrics"]["log_loss"],
        "development_cutoff": "2024-12-31",
        "development_fights_all": int(len(df)),
        "oof_fights_betting_eligible": int(summary["oof_fight_count"]),
        "xgboost_version": xgb.__version__,
        "commit_sha": _git_sha(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({
        "stage": "develop",
        "selected_candidate": selected,
        "oof_market": summary["oof_market_metrics"],
        "oof_model": summary["oof_model_metrics"],
        "delta_log_loss": summary["oof_delta_log_loss"],
        "holdout_integrity": freeze["holdout_integrity"],
    }, indent=2))


def test() -> None:
    if not FREEZE_PATH.exists() or not SUMMARY_PATH.exists() or not OOF_PATH.exists():
        raise RuntimeError("development freeze artifacts missing")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("holdout_integrity") != "PRISTINE":
        raise RuntimeError("2025+ holdout integrity was already compromised by a pre-compliance score; refusing to claim or perform a compliant sealed test")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("test_status") != "SEALED_NOT_SCORED":
        raise RuntimeError(f"test already scored or blocked: {summary.get('test_status')}")
    if TEST_PATH.exists():
        raise RuntimeError("test prediction artifact already exists; refusing second score")

    features = list(freeze["selected_features"])
    df, _, _ = _build_rows(False, include_targets=True, forced_features=features)
    train = df[df["date"] <= DEV_CUTOFF].copy()
    test_all = df[df["date"] >= "2025-01-01"].copy()
    test_df = test_all[test_all["betting_eligible"]].copy()
    pred, booster, valid = _fit_predict(train, test_df, features)
    market_p = test_df[MKT_COLS].to_numpy(float)
    y = test_df["target"].to_numpy(int)
    mm, mx = _metrics(y, market_p), _metrics(y, pred)
    _prediction_rows(test_df, pred).to_csv(TEST_PATH, index=False)
    summary["test_status"] = "SCORED_ONCE_AFTER_FREEZE"
    summary["final_test"] = {
        "date_min": test_df["date"].min().date().isoformat(),
        "date_max": test_df["date"].max().date().isoformat(),
        "test_n_all_complete": int(len(test_all)),
        "test_n_betting_eligible": int(len(test_df)),
        "cold_start_exclusions": int((~test_all["betting_eligible"]).sum()),
        "feature_count": len(valid),
        "market": mm, "model": mx,
        "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
        "delta_brier_vs_market": float(mx["brier"] - mm["brier"]),
        "per_class_calibration": _class_calibration(y, pred),
        "top_features_by_gain": [{"feature": k, "gain": float(v)} for k, v in sorted(booster.get_score(importance_type="gain").items(), key=lambda z: z[1], reverse=True)[:30]],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "test", "market": mm, "model": mx}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["audit", "develop", "test"])
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
    elif args.stage == "develop":
        develop()
    else:
        test()


if __name__ == "__main__":
    main()
