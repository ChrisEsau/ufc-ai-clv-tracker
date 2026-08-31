from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("data/research/prop_mispricing")
MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")
FEATURE_PATH = Path("data/features/moneyline_feature_view.parquet")
SUMMARY_PATH = ROOT / "xgboost_method_market_offset__summary.json"
FREEZE_PATH = ROOT / "xgboost_method_market_offset__frozen_candidate.json"
OOF_PATH = ROOT / "xgboost_method_market_offset__oof_predictions.csv"
TEST_PATH = ROOT / "xgboost_method_market_offset__test_predictions.csv"

CLASS_SPECS = [
    ("red_ko_tko", "red", "win_by_ko_tko_dq", "ko_tko"),
    ("red_sub", "red", "win_by_submission", "submission"),
    ("red_dec", "red", "win_by_decision", "decision"),
    ("blue_ko_tko", "blue", "win_by_ko_tko_dq", "ko_tko"),
    ("blue_sub", "blue", "win_by_submission", "submission"),
    ("blue_dec", "blue", "win_by_decision", "decision"),
]
CLASS_NAMES = [x[0] for x in CLASS_SPECS]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}
SIDE_BY_CLASS = np.array([x[1] for x in CLASS_SPECS])
METHOD_BY_CLASS = np.array([x[3] for x in CLASS_SPECS])
METHOD_KEYS = {x[2] for x in CLASS_SPECS}

FOLDS = [
    ("2021", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
]
CANDIDATE_SIZES = [None, 100, 75, 50, 35]
PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
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


def _read_parquet_cutoff(path: Path, stage: str) -> pd.DataFrame:
    # Development is physically filtered at parquet read time so 2025+ rows are
    # not exposed to model/candidate selection. Test stage is a separate process.
    if stage == "develop":
        return pd.read_parquet(path, filters=[("date", "<=", pd.Timestamp("2024-12-31"))])
    if stage == "test":
        return pd.read_parquet(path)
    raise ValueError(stage)


def _validate_probs(p: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 6:
        raise RuntimeError(f"{name}: expected n x 6 probabilities, got {p.shape}")
    if not np.isfinite(p).all():
        raise RuntimeError(f"{name}: non-finite probability")
    if (p < -1e-12).any() or (p > 1 + 1e-12).any():
        raise RuntimeError(f"{name}: probability outside [0,1]")
    err = np.max(np.abs(p.sum(axis=1) - 1.0)) if len(p) else 0.0
    if err > 1e-8:
        raise RuntimeError(f"{name}: row probability sum error {err}")
    return p


def _market_margin(p: np.ndarray) -> np.ndarray:
    # softmax(log(p)) == p. Flatten row-major for multiclass DMatrix base_margin.
    p = np.clip(_validate_probs(p, "market"), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def _dmatrix(x: pd.DataFrame, p_market: np.ndarray, y: np.ndarray | None = None) -> xgb.DMatrix:
    d = xgb.DMatrix(x, label=y, feature_names=list(x.columns)) if y is not None else xgb.DMatrix(x, feature_names=list(x.columns))
    margin = _market_margin(p_market)
    d.set_base_margin(margin.reshape(-1))
    return d


def _method_market(stage: str) -> pd.DataFrame:
    m = _read_parquet_cutoff(MARKET_PATH, stage).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[
        (m["bookmaker"] == "legacy_consensus")
        & (m["result_status"] == "graded")
        & m["won"].notna()
        & m["market_key"].isin(METHOD_KEYS)
        & m["outcome_side"].astype(str).isin(["red", "blue"])
    ].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m = m.dropna(subset=["date", "fight_id", "implied_probability"]).copy()
    m = m[np.isfinite(m["implied_probability"]) & (m["implied_probability"] > 0)].copy()
    m["won"] = m["won"].astype(bool).astype(int)
    m["method_class"] = None
    for cname, side, key, _ in CLASS_SPECS:
        hit = m["outcome_side"].astype(str).eq(side) & m["market_key"].eq(key)
        m.loc[hit, "method_class"] = cname
    m = m[m["method_class"].notna()].copy()

    # Exactly one quote for each of the six mutually-exclusive outcomes and one winner.
    counts = m.groupby(["fight_id", "method_class"]).size().unstack(fill_value=0)
    for cname in CLASS_NAMES:
        if cname not in counts.columns:
            counts[cname] = 0
    complete = counts.index[(counts[CLASS_NAMES] == 1).all(axis=1)]
    m = m[m["fight_id"].isin(complete)].copy()
    won_sum = m.groupby("fight_id")["won"].sum()
    good = won_sum.index[won_sum.eq(1)]
    m = m[m["fight_id"].isin(good)].copy()
    return m.sort_values(["date", "fight_id", "method_class"]).reset_index(drop=True)


def _build_rows(stage: str, forced_features: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    m = _method_market(stage)
    idx = m[["fight_id", "date", "event_name"]].drop_duplicates("fight_id").copy()

    implied = m.pivot(index="fight_id", columns="method_class", values="implied_probability")
    implied = implied[CLASS_NAMES]
    overround = implied.sum(axis=1)
    fair = implied.div(overround, axis=0)
    won = m.pivot(index="fight_id", columns="method_class", values="won")[CLASS_NAMES]
    target = np.argmax(won.to_numpy(dtype=int), axis=1)

    market_frame = pd.DataFrame({"fight_id": implied.index, "market_overround": overround.to_numpy(), "target": target})
    for j, cname in enumerate(CLASS_NAMES):
        market_frame[f"market_p_{cname}"] = fair.iloc[:, j].to_numpy()

    red_name = m[m["outcome_side"].astype(str).eq("red")].groupby("fight_id")["outcome_label"].first().rename("red_fighter")
    blue_name = m[m["outcome_side"].astype(str).eq("blue")].groupby("fight_id")["outcome_label"].first().rename("blue_fighter")
    market_frame = market_frame.merge(red_name, on="fight_id", how="left").merge(blue_name, on="fight_id", how="left")

    fv = _read_parquet_cutoff(FEATURE_PATH, stage).copy()
    fv["date"] = pd.to_datetime(fv["date"], errors="coerce") if "date" in fv.columns else pd.NaT
    deny = ["winner", "result", "target", "label", "finish_round", "match_time_sec", "profit", "odds", "implied", "market", "actual", "post_"]
    if forced_features is None:
        diff_cols = sorted([
            c for c in fv.columns
            if c.lower().endswith("_diff")
            and "abs_diff" not in c.lower()
            and "_abs_" not in c.lower()
            and not any(x in c.lower() for x in deny)
            and pd.api.types.is_numeric_dtype(fv[c])
        ])
    else:
        diff_cols = list(forced_features)
        missing = [c for c in diff_cols if c not in fv.columns]
        if missing:
            raise RuntimeError(f"frozen features missing from feature view: {missing}")

    if not diff_cols:
        raise RuntimeError("no leakage-safe signed diff features found")
    f = fv[["fight_id"] + diff_cols].drop_duplicates("fight_id")
    df = idx.merge(market_frame, on="fight_id", how="inner").merge(f, on="fight_id", how="inner")
    df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)
    market_cols = [f"market_p_{c}" for c in CLASS_NAMES]
    _validate_probs(df[market_cols].to_numpy(float), "normalized six-way market")
    return df, diff_cols


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
            "empirical_rate": float(obs.mean()),
            "mean_probability": float(p[:, j].mean()),
            "brier_component": float(np.mean((p[:, j] - obs) ** 2)),
        }
    return out


def _fit_predict(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, xgb.Booster, list[str]]:
    raw_train = train[features + ["market_overround"]].replace([np.inf, -np.inf], np.nan)
    raw_val = val[features + ["market_overround"]].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in raw_train.columns if raw_train[c].notna().any()]
    med = raw_train[valid].median(numeric_only=True)
    xtr = raw_train[valid].fillna(med).fillna(0.0)
    xva = raw_val[valid].fillna(med).fillna(0.0)
    mcols = [f"market_p_{c}" for c in CLASS_NAMES]
    ptr = train[mcols].to_numpy(float)
    pva = val[mcols].to_numpy(float)
    ytr = train["target"].to_numpy(int)
    dtr = _dmatrix(xtr, ptr, ytr)
    dva = _dmatrix(xva, pva)
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = np.asarray(booster.predict(dva), dtype=float)
    pred = _validate_probs(pred, "model prediction")
    return pred, booster, valid


def _rank_features(df: pd.DataFrame, diff_cols: list[str]) -> list[str]:
    train = df[df["date"] <= "2020-12-31"].copy()
    if train.empty:
        raise RuntimeError("no pre-2021 rows available for feature ranking")
    _, booster, _ = _fit_predict(train, train.iloc[:1].copy(), diff_cols)
    gain = booster.get_score(importance_type="gain")
    return sorted(diff_cols, key=lambda c: (-float(gain.get(c, 0.0)), c))


def _prediction_rows(frame: pd.DataFrame, p: np.ndarray, fold: str | None = None) -> pd.DataFrame:
    out = frame[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target"]].copy()
    out["actual_class"] = [CLASS_NAMES[i] for i in out["target"].astype(int)]
    if fold is not None:
        out["fold"] = fold
    for j, cname in enumerate(CLASS_NAMES):
        out[f"market_p_{cname}"] = frame[f"market_p_{cname}"].to_numpy(float)
        out[f"model_p_{cname}"] = p[:, j]
        out[f"edge_{cname}"] = p[:, j] - out[f"market_p_{cname}"]
    return out


def develop() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    df, diff_cols = _build_rows("develop")
    if (df["date"] > pd.Timestamp("2024-12-31")).any():
        raise RuntimeError("sealed 2025+ row entered development dataset")

    ranked = _rank_features(df, diff_cols)
    candidates: dict[str, list[str]] = {"full_v5_signed": list(diff_cols)}
    for n in [100, 75, 50, 35]:
        candidates[f"top_{n}_pre2021_gain"] = ranked[: min(n, len(ranked))]

    summary = {
        "design": {
            "name": "six_way_exact_method_market_offset_v1",
            "classes": CLASS_NAMES,
            "market_prior": "six raw exact-method implied probabilities normalized within fight; log(probability) supplied as multiclass base_margin",
            "features": "same leakage-screened signed *_diff family used by moneyline V4/V5; six-way market overround appended as a feature",
            "architecture": {**PARAMS, "num_boost_round": ROUNDS},
            "feature_ranking": "multiclass gain from model trained only through 2020-12-31",
            "candidate_selection": "strict minimum pooled chronological 2021-2024 OOF six-way log loss; exact tie -> fewer features",
            "sealed_test": "2025+ is parquet-filtered out of the development process and scored only by separate --stage test invocation after frozen candidate file exists",
            "roi_used_for_selection": False,
            "betting_policy_used_for_selection": False,
            "cold_start_note": "This experiment selects a probability model, not a betting policy; no cold-start betting-eligibility rule is used to select/tune the model.",
        },
        "development_coverage": {
            "eligible_complete_six_way_fights_through_2024": int(len(df)),
            "date_min": df["date"].min().date().isoformat() if len(df) else None,
            "date_max": df["date"].max().date().isoformat() if len(df) else None,
            "signed_diff_feature_count": int(len(diff_cols)),
        },
        "pre2021_ranked_features": ranked,
        "candidates": {},
    }

    all_parts: dict[str, list[pd.DataFrame]] = {}
    for cname, features in candidates.items():
        fold_summaries = []
        parts = []
        for fold, train_end, val_start, val_end in FOLDS:
            train = df[df["date"] <= train_end].copy()
            val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
            if train.empty or val.empty:
                raise RuntimeError(f"empty train/validation for fold {fold}: {len(train)}/{len(val)}")
            pred, _, valid = _fit_predict(train, val, features)
            mcols = [f"market_p_{c}" for c in CLASS_NAMES]
            market_p = val[mcols].to_numpy(float)
            y = val["target"].to_numpy(int)
            mm = _metrics(y, market_p)
            mx = _metrics(y, pred)
            fold_summaries.append({
                "fold": fold,
                "train_n": int(len(train)),
                "validation_n": int(len(val)),
                "feature_count": int(len(valid)),
                "market": mm,
                "model": mx,
                "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
                "delta_brier_vs_market": float(mx["brier"] - mm["brier"]),
            })
            parts.append(_prediction_rows(val, pred, fold))
        oof = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
        y = oof["target"].to_numpy(int)
        market_p = oof[[f"market_p_{c}" for c in CLASS_NAMES]].to_numpy(float)
        model_p = oof[[f"model_p_{c}" for c in CLASS_NAMES]].to_numpy(float)
        mm = _metrics(y, market_p)
        mx = _metrics(y, model_p)
        summary["candidates"][cname] = {
            "feature_count": int(len(features) + 1),
            "folds": fold_summaries,
            "market_oof": mm,
            "model_oof": mx,
            "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
            "delta_brier_vs_market": float(mx["brier"] - mm["brier"]),
            "class_calibration": _class_calibration(y, model_p),
        }
        all_parts[cname] = parts

    selected = min(
        candidates,
        key=lambda k: (summary["candidates"][k]["model_oof"]["log_loss"], len(candidates[k])),
    )
    selected_features = candidates[selected]
    summary["selected_candidate"] = selected
    summary["selected_features"] = selected_features
    summary["selected_oof"] = summary["candidates"][selected]["model_oof"]
    summary["selected_delta_log_loss_vs_market"] = summary["candidates"][selected]["delta_log_loss_vs_market"]
    summary["test_status"] = "SEALED_NOT_SCORED"

    selected_oof = pd.concat(all_parts[selected], ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    selected_oof.to_csv(OOF_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    freeze = {
        "schema_version": 1,
        "frozen_before_2025_plus_scoring": True,
        "selected_candidate": selected,
        "selected_features": selected_features,
        "class_names": CLASS_NAMES,
        "params": PARAMS,
        "num_boost_round": ROUNDS,
        "selection_metric": "pooled_2021_2024_multiclass_log_loss",
        "selected_oof_log_loss": summary["selected_oof"]["log_loss"],
        "development_cutoff": "2024-12-31",
        "development_fights": int(len(df)),
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "develop", "selected_candidate": selected, "selected_oof": summary["selected_oof"], "market_oof": summary["candidates"][selected]["market_oof"]}, indent=2))


def test() -> None:
    if not FREEZE_PATH.exists() or not SUMMARY_PATH.exists() or not OOF_PATH.exists():
        raise RuntimeError("development freeze artifacts missing; refusing to score 2025+")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if not freeze.get("frozen_before_2025_plus_scoring"):
        raise RuntimeError("candidate is not marked frozen")
    features = list(freeze["selected_features"])

    # Build all rows only in this separate frozen test process, then split exactly at cutoff.
    df, _ = _build_rows("test", forced_features=features)
    train = df[df["date"] <= "2024-12-31"].copy()
    test_df = df[df["date"] >= "2025-01-01"].copy()
    if train.empty or test_df.empty:
        raise RuntimeError(f"empty final train/test: {len(train)}/{len(test_df)}")

    pred, booster, valid = _fit_predict(train, test_df, features)
    mcols = [f"market_p_{c}" for c in CLASS_NAMES]
    market_p = test_df[mcols].to_numpy(float)
    y = test_df["target"].to_numpy(int)
    mm = _metrics(y, market_p)
    mx = _metrics(y, pred)
    rows = _prediction_rows(test_df, pred)
    rows.to_csv(TEST_PATH, index=False)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["test_status"] = "SCORED_ONCE_AFTER_FREEZE"
    summary["final_test"] = {
        "date_min": test_df["date"].min().date().isoformat(),
        "date_max": test_df["date"].max().date().isoformat(),
        "train_n": int(len(train)),
        "test_n": int(len(test_df)),
        "feature_count": int(len(valid)),
        "market": mm,
        "model": mx,
        "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
        "delta_brier_vs_market": float(mx["brier"] - mm["brier"]),
        "class_calibration": _class_calibration(y, pred),
        "top_features_by_gain": [
            {"feature": k, "gain": float(v)}
            for k, v in sorted(booster.get_score(importance_type="gain").items(), key=lambda z: z[1], reverse=True)[:30]
        ],
        "max_probability_sum_error": float(np.max(np.abs(pred.sum(axis=1) - 1.0))),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "test", "selected_candidate": freeze["selected_candidate"], "market": mm, "model": mx, "delta_log_loss_vs_market": summary["final_test"]["delta_log_loss_vs_market"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["develop", "test"])
    args = parser.parse_args()
    if args.stage == "develop":
        develop()
    else:
        test()


if __name__ == "__main__":
    main()
