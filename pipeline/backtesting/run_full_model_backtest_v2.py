from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from pipeline.backtesting.model_outcomes import build_backtest_model_outcomes
from pipeline.backtesting.run_backtest_v2 import (
    apply_filters,
    bucket_summary,
    score_bets,
    standardize_market,
    summarize,
)
from pipeline.common.paths import MARKET_DIR, MODEL_LAB_DIR, MONEYLINE_FEATURE_VIEW_PATH, ensure_data_dirs

DEFAULT_CONFIG_PATH = Path("configs/models/moneyline_xgboost_v6_dev.yaml")
DEFAULT_FEATURE_VIEW_PATH = MONEYLINE_FEATURE_VIEW_PATH
DEFAULT_MARKET_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
DEFAULT_OUTPUT_ROOT = MODEL_LAB_DIR / "backtests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full V2 model backtest from features, model, and historical markets.")
    parser.add_argument("--model-config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--feature-view-path", default=str(DEFAULT_FEATURE_VIEW_PATH))
    parser.add_argument("--historical-market-path", default=str(DEFAULT_MARKET_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--market-key", default="moneyline")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--starting-bankroll", type=float, default=10000.0)
    parser.add_argument("--flat-stake", type=float, default=100.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-stake-pct", type=float, default=0.05)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-odds", type=float, default=-1000.0)
    parser.add_argument("--max-odds", type=float, default=3000.0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a dict: {path}")
    return config


def model_artifact_path(config: dict[str, Any]) -> Path:
    output_dir = Path(config["artifacts"]["output_dir"])
    calibrated = output_dir / "calibrated_model.joblib"
    raw = output_dir / "raw_model.joblib"
    if calibrated.exists():
        return calibrated
    if raw.exists():
        return raw
    raise FileNotFoundError(f"No model artifact found in {output_dir}")


def predict_positive_probability(model: Any, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if getattr(probs, "ndim", 1) == 2:
            return pd.Series(probs[:, 1], index=X.index, dtype="float64")
        return pd.Series(probs, index=X.index, dtype="float64")
    if hasattr(model, "predict"):
        return pd.Series(model.predict(X), index=X.index, dtype="float64")
    raise ValueError("Loaded model does not expose predict_proba or predict.")


def clip_probabilities(probs: pd.Series, config: dict[str, Any]) -> pd.Series:
    probability_config = (config.get("prediction") or {}).get("probability", {}) or {}
    low = float(probability_config.get("clip_low", 0.0))
    high = float(probability_config.get("clip_high", 1.0))
    return probs.clip(lower=low, upper=high)


def prepare_feature_view(path: Path, config: dict[str, Any], start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature view not found: {path}")
    df = pd.read_parquet(path).copy()
    if "date" not in df.columns:
        raise ValueError("Feature view must contain date column for era filtering.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)].copy()
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)].copy()
    features = list(config["features"]["feature_columns"])
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature view missing model features: {missing[:20]} total_missing={len(missing)}")
    return df


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    run_time = datetime.now(timezone.utc)
    ts = run_time.isoformat()
    config = load_config(Path(args.model_config_path))
    args.model_id = str(config["model_id"])
    backtest_id = f"{args.model_id}_{args.market_key}_full_{run_time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / backtest_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = prepare_feature_view(Path(args.feature_view_path), config, args.start_date, args.end_date)
    feature_cols = list(config["features"]["feature_columns"])
    model = joblib.load(model_artifact_path(config))
    X = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probs = clip_probabilities(predict_positive_probability(model, X), config)
    model_outcomes = build_backtest_model_outcomes(
        feature_df=feature_df,
        probabilities=probs,
        model_config=config,
        prediction_run_id=backtest_id,
        prediction_timestamp=ts,
    )

    market_raw = pd.read_parquet(args.historical_market_path)
    market_raw = market_raw[market_raw["market_key"].astype(str).str.lower() == args.market_key.lower()].copy()
    market = standardize_market(market_raw)
    joined = model_outcomes.merge(market, on=["fight_id", "market_key", "outcome_join_key"], how="inner", suffixes=("_model", "_market"))
    candidates = apply_filters(joined, args)
    scored = score_bets(candidates, args)
    summary = summarize(scored, args, backtest_id)
    summary.update({
        "mode": "full_model",
        "prediction_format": str((config.get("prediction") or {}).get("format", "binary_matchup")),
        "feature_rows": int(len(feature_df)),
        "historical_model_outcome_rows": int(len(model_outcomes)),
        "joined_rows": int(len(joined)),
        "model_config_path": str(args.model_config_path),
        "feature_view_path": str(args.feature_view_path),
        "historical_market_path": str(args.historical_market_path),
    })
    buckets = bucket_summary(scored)
    config_payload = vars(args) | {"backtest_id": backtest_id, "created_at_utc": ts, "mode": "full_model"}

    (output_dir / "backtest_config.json").write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    (output_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    model_outcomes.to_parquet(output_dir / "historical_model_outcomes.parquet", index=False)
    joined.to_parquet(output_dir / "joined_model_market_rows.parquet", index=False)
    scored.to_parquet(output_dir / "backtest_bets.parquet", index=False)
    buckets.to_parquet(output_dir / "backtest_bucket_summary.parquet", index=False)

    registry_path = Path(args.output_root) / "backtest_registry.parquet"
    registry_row = pd.DataFrame([summary | {"output_dir": str(output_dir), "created_at_utc": ts}])
    registry = pd.concat([pd.read_parquet(registry_path), registry_row], ignore_index=True) if registry_path.exists() else registry_row
    registry.to_parquet(registry_path, index=False)

    print("=" * 80)
    print("FULL MODEL BACKTEST V2")
    print("=" * 80)
    print("Backtest ID:", backtest_id)
    print("Start date:", args.start_date)
    print("End date:", args.end_date)
    print("Prediction format:", str((config.get("prediction") or {}).get("format", "binary_matchup")))
    print("Feature rows:", len(feature_df))
    print("Model outcome rows:", len(model_outcomes))
    print("Market rows:", len(market))
    print("Joined rows:", len(joined))
    print("Bet candidates:", len(scored))
    print(json.dumps(summary, indent=2, default=str))
    print("Output dir:", output_dir)


if __name__ == "__main__":
    main()
