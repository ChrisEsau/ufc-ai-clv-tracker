"""Run live predictions for market-aware ensemble models.

Run from repo root:

    python -m pipeline.modeling.run_ensemble_prediction \
        --config configs/models/moneyline_xgboost_v13_ensemble_top60.yaml

This runner is intentionally standalone for Phase 4 validation. It does not
modify the existing single-model prediction path.

Inputs:
    data/predictions/live_model_features.parquet
    data/market/market_outcomes.parquet
    models/moneyline/<model_id>/ensemble_manifest.json
    models/moneyline/<model_id>/members/<member_id>/

Outputs:
    data/predictions/model_outcomes.parquet
    data/predictions/by_model/<model_id>/model_outcomes.parquet
    data/predictions/by_model/<model_id>/ensemble_details.parquet
    data/predictions/by_model/<model_id>/ensemble_exclusions.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from pipeline.training.calibration import predict_positive_class_probability
from pipeline.modeling.confidence import score_prediction_confidence
from pipeline.prediction.live_feature_builder import (
    build_live_model_features,
    write_live_feature_outputs,
)


DEFAULT_CONFIG_PATH = "configs/models/moneyline_xgboost_v13_ensemble_top60.yaml"
DEFAULT_LIVE_FEATURES_PATH = "data/predictions/live_model_features.parquet"
DEFAULT_MARKET_OUTCOMES_PATH = "data/market/market_outcomes.parquet"


MARKET_FEATURES = [
    "favorite_odds",
    "dog_odds",
    "favorite_implied_probability",
    "dog_implied_probability",
    "market_prob_gap",
    "abs_market_prob_gap",
    "favorite_odds_abs",
    "dog_odds_abs",
    "market_price_width",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live ensemble model predictions.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to ensemble config YAML.")
    parser.add_argument("--live-features", default=DEFAULT_LIVE_FEATURES_PATH, help="Live model features parquet.")
    parser.add_argument("--market-outcomes", default=DEFAULT_MARKET_OUTCOMES_PATH, help="Live market outcomes parquet.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw_model.joblib for members when available instead of calibrated_model.joblib.",
    )
    parser.add_argument(
        "--write-canonical",
        action="store_true",
        help="Also write data/predictions/model_outcomes.parquet. Use only when promoting/routing this model.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML did not load to dict: {path}")
    return value


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_id(config: dict[str, Any]) -> str:
    return str(config["model_id"])


def get_output_dir(config: dict[str, Any]) -> Path:
    return Path(config["artifacts"]["output_dir"])


def get_prediction_output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    model_id = get_model_id(config)
    prediction_config = config.get("prediction", {}) or {}
    output_config = prediction_config.get("output", {}) or {}

    scoped_template = output_config.get(
        "model_scoped_path_template",
        "data/predictions/by_model/{model_id}/model_outcomes.parquet",
    )
    details_template = output_config.get(
        "ensemble_details_path_template",
        "data/predictions/by_model/{model_id}/ensemble_details.parquet",
    )

    return Path(str(scoped_template).format(model_id=model_id)), Path(
        str(details_template).format(model_id=model_id)
    )


def canonical_output_path(config: dict[str, Any]) -> Path:
    prediction_config = config.get("prediction", {}) or {}
    output_config = prediction_config.get("output", {}) or {}
    return Path(output_config.get("path", "data/predictions/model_outcomes.parquet"))


def resolve_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Could not resolve {label}. Tried: {candidates}")


def series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a single column as Series even if duplicate column names exist."""
    value = df.loc[:, column]
    if isinstance(value, pd.DataFrame):
        return value.iloc[:, 0]
    return value


def safe_numeric(value: Any) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)


def american_profit_per_100(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds
    return 10000.0 / abs(odds)


def normalize_live_feature_columns(live_df: pd.DataFrame) -> pd.DataFrame:
    """Add stable column aliases used by the ensemble adapter."""
    out = live_df.copy()

    alias_pairs = [
        ("event_id", ["event_id"]),
        ("event_name", ["event_name"]),
        ("commence_time", ["commence_time", "date"]),
        ("fight_id", ["fight_id"]),
        ("red_fighter", ["red_fighter", "r_name"]),
        ("blue_fighter", ["blue_fighter", "b_name"]),
        ("red_fighter_id", ["red_fighter_id", "r_id"]),
        ("blue_fighter_id", ["blue_fighter_id", "b_id"]),
    ]

    for target, candidates in alias_pairs:
        if target in out.columns:
            continue
        for candidate in candidates:
            if candidate in out.columns:
                out[target] = series(out, candidate)
                break

    required = [
        "event_name",
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
    ]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Live features missing required columns after aliasing: {missing}")

    if "event_id" not in out.columns:
        out["event_id"] = pd.NA
    if "commence_time" not in out.columns:
        out["commence_time"] = pd.NA

    return out


def prepare_moneyline_market(market_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "fight_id",
        "market_key",
        "bookmaker",
        "outcome_fighter_id",
        "outcome_label",
        "outcome_fighter_name",
        "american_odds",
        "implied_probability",
    ]
    missing = [column for column in required if column not in market_df.columns]
    if missing:
        raise ValueError(f"Market outcomes missing required columns: {missing}")

    ml = market_df[market_df["market_key"].eq("moneyline")].copy()
    if ml.empty:
        raise ValueError("No moneyline rows found in market outcomes.")

    # Keep all available bookmakers here. Book preference is applied below
    # per fight and fighter, allowing FanDuel to serve as a fallback whenever
    # DraftKings does not offer both sides of a specific matchup.

    ml["american_odds"] = safe_numeric(ml["american_odds"])
    ml["implied_probability"] = safe_numeric(ml["implied_probability"])
    ml = ml.dropna(subset=["fight_id", "outcome_fighter_id", "american_odds", "implied_probability"]).copy()

    if "snapshot_timestamp" in ml.columns:
        ml["_snapshot_sort"] = pd.to_datetime(ml["snapshot_timestamp"], errors="coerce")
    else:
        ml["_snapshot_sort"] = pd.NaT

    priority = {"DraftKings": 0, "FanDuel": 1}
    ml["_book_priority"] = ml["bookmaker"].map(priority).fillna(99)

    ml = (
        ml.sort_values(
            ["fight_id", "outcome_fighter_id", "_book_priority", "_snapshot_sort"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["fight_id", "outcome_fighter_id"], keep="first")
        .copy()
    )

    return ml


def join_live_market(live_df: pd.DataFrame, market_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join red/blue moneyline prices to live feature rows."""
    live = normalize_live_feature_columns(live_df)
    ml = prepare_moneyline_market(market_df)

    price_cols = [
        "fight_id",
        "outcome_fighter_id",
        "outcome_label",
        "outcome_fighter_name",
        "american_odds",
        "implied_probability",
        "bookmaker",
        "snapshot_run_id",
        "snapshot_timestamp",
    ]
    price_cols = [column for column in price_cols if column in ml.columns]

    red_prices = ml[price_cols].rename(
        columns={
            "outcome_fighter_id": "red_fighter_id",
            "outcome_label": "red_market_label",
            "outcome_fighter_name": "red_market_fighter_name",
            "american_odds": "red_american_odds",
            "implied_probability": "red_implied_probability",
            "bookmaker": "red_bookmaker",
            "snapshot_run_id": "red_snapshot_run_id",
            "snapshot_timestamp": "red_snapshot_timestamp",
        }
    )
    blue_prices = ml[price_cols].rename(
        columns={
            "outcome_fighter_id": "blue_fighter_id",
            "outcome_label": "blue_market_label",
            "outcome_fighter_name": "blue_market_fighter_name",
            "american_odds": "blue_american_odds",
            "implied_probability": "blue_implied_probability",
            "bookmaker": "blue_bookmaker",
            "snapshot_run_id": "blue_snapshot_run_id",
            "snapshot_timestamp": "blue_snapshot_timestamp",
        }
    )

    joined = live.merge(red_prices, on=["fight_id", "red_fighter_id"], how="left")
    joined = joined.merge(blue_prices, on=["fight_id", "blue_fighter_id"], how="left")

    joined["has_red_odds"] = joined["red_american_odds"].notna()
    joined["has_blue_odds"] = joined["blue_american_odds"].notna()
    joined["has_both_odds"] = joined["has_red_odds"] & joined["has_blue_odds"]

    exclusions = joined[~joined["has_both_odds"]].copy()
    usable = joined[joined["has_both_odds"]].copy()

    return usable, exclusions


def add_favorite_dog_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    red_is_favorite = (
        out["red_implied_probability"].gt(out["blue_implied_probability"])
        | (
            out["red_implied_probability"].eq(out["blue_implied_probability"])
            & out["red_american_odds"].le(out["blue_american_odds"])
        )
    )

    out["favorite_side"] = np.where(red_is_favorite, "red", "blue")
    out["dog_side"] = np.where(red_is_favorite, "blue", "red")

    out["favorite_fighter_id"] = np.where(red_is_favorite, out["red_fighter_id"], out["blue_fighter_id"])
    out["dog_fighter_id"] = np.where(red_is_favorite, out["blue_fighter_id"], out["red_fighter_id"])

    out["favorite_fighter_name"] = np.where(red_is_favorite, out["red_fighter"], out["blue_fighter"])
    out["dog_fighter_name"] = np.where(red_is_favorite, out["blue_fighter"], out["red_fighter"])

    out["favorite_odds"] = np.where(red_is_favorite, out["red_american_odds"], out["blue_american_odds"])
    out["dog_odds"] = np.where(red_is_favorite, out["blue_american_odds"], out["red_american_odds"])

    out["favorite_implied_probability"] = np.where(
        red_is_favorite,
        out["red_implied_probability"],
        out["blue_implied_probability"],
    )
    out["dog_implied_probability"] = np.where(
        red_is_favorite,
        out["blue_implied_probability"],
        out["red_implied_probability"],
    )

    out["favorite_bookmaker"] = np.where(red_is_favorite, out["red_bookmaker"], out["blue_bookmaker"])
    out["dog_bookmaker"] = np.where(red_is_favorite, out["blue_bookmaker"], out["red_bookmaker"])

    out["market_prob_gap"] = out["favorite_implied_probability"] - out["dog_implied_probability"]
    out["abs_market_prob_gap"] = out["market_prob_gap"].abs()
    out["favorite_odds_abs"] = out["favorite_odds"].abs()
    out["dog_odds_abs"] = out["dog_odds"].abs()
    out["market_price_width"] = (out["favorite_odds"] - out["dog_odds"]).abs()

    out["favorite_profit_per_100"] = out["favorite_odds"].apply(american_profit_per_100)
    out["dog_profit_per_100"] = out["dog_odds"].apply(american_profit_per_100)

    out["_favorite_sign"] = np.where(red_is_favorite, 1.0, -1.0)

    return out



def prefer_calibrated_from_config(config: Mapping[str, Any]) -> bool:
    """Return whether ensemble members should use calibrated_model.joblib.

    Default remains calibrated_model for backward compatibility.
    Set prediction.probability_artifact: raw_model to reproduce raw-probability experiments.
    """
    prediction_config = config.get("prediction", {}) or {}
    artifact = str(prediction_config.get("probability_artifact", "calibrated_model")).strip().lower()

    aliases = {
        "calibrated": "calibrated_model",
        "calibrated_model.joblib": "calibrated_model",
        "raw": "raw_model",
        "raw_model.joblib": "raw_model",
    }
    artifact = aliases.get(artifact, artifact)

    if artifact == "calibrated_model":
        return True
    if artifact == "raw_model":
        return False

    raise ValueError(
        "Unsupported prediction.probability_artifact: "
        f"{prediction_config.get('probability_artifact')!r}. "
        "Expected raw_model or calibrated_model."
    )


def load_member_model(member_dir: Path, *, prefer_calibrated: bool = True) -> tuple[Any, Path]:
    calibrated = member_dir / "calibrated_model.joblib"
    raw = member_dir / "raw_model.joblib"

    if prefer_calibrated and calibrated.exists():
        return joblib.load(calibrated), calibrated
    if raw.exists():
        return joblib.load(raw), raw
    if calibrated.exists():
        return joblib.load(calibrated), calibrated

    raise FileNotFoundError(f"No member model artifact found in {member_dir}")


def load_feature_columns(member_dir: Path) -> list[str]:
    json_path = member_dir / "feature_columns.json"
    joblib_path = member_dir / "feature_columns.joblib"

    if json_path.exists():
        values = load_json(json_path)
    elif joblib_path.exists():
        values = joblib.load(joblib_path)
    else:
        raise FileNotFoundError(f"No feature columns found in {member_dir}")

    if not isinstance(values, list):
        raise ValueError(f"Feature columns must be a list for {member_dir}")

    return [str(value) for value in values]


def build_member_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}

    favorite_sign = safe_numeric(df["_favorite_sign"]).fillna(1.0)

    for column in feature_columns:
        if column in MARKET_FEATURES:
            data[column] = safe_numeric(series(df, column)).fillna(0.0)
            continue

        if column.startswith("favpersp__"):
            clean_column = column.removeprefix("favpersp__")
            if clean_column not in df.columns:
                raise ValueError(f"Live feature matrix missing clean feature for {column}: {clean_column}")
            data[column] = safe_numeric(series(df, clean_column)).fillna(0.0) * favorite_sign
            continue

        if column.startswith("dogpersp__"):
            clean_column = column.removeprefix("dogpersp__")
            if clean_column not in df.columns:
                raise ValueError(f"Live feature matrix missing clean feature for {column}: {clean_column}")
            data[column] = safe_numeric(series(df, clean_column)).fillna(0.0) * favorite_sign * -1.0
            continue

        if column not in df.columns:
            raise ValueError(f"Live feature matrix missing member feature: {column}")
        data[column] = safe_numeric(series(df, column)).fillna(0.0)

    return pd.DataFrame(data, index=df.index)[feature_columns]


def member_configs_from_manifest_or_config(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    config_members = (config.get("ensemble") or {}).get("members") or []
    config_by_id = {str(member["member_id"]): member for member in config_members}

    if not manifest:
        return config_members

    members = []
    for manifest_member in manifest.get("members", []):
        member_id = str(manifest_member["member_id"])
        merged = dict(config_by_id.get(member_id, {}))
        merged.update(manifest_member)
        merged["member_id"] = member_id
        members.append(merged)

    return members or config_members


def resolve_live_builder_feature_columns(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> list[str]:
    """Resolve clean feature columns needed by the live feature builder.

    Ensemble member artifacts store perspective columns such as:

        favpersp__ewm_splm_diff
        dogpersp__ewm_splm_diff

    The normal Prediction V2 live feature builder expects the clean feature name:

        ewm_splm_diff

    Market features are excluded here because they come from live odds, not the
    fighter feature builder.
    """

    output_dir = get_output_dir(config)
    members = member_configs_from_manifest_or_config(config=config, manifest=manifest)

    clean_features: list[str] = []

    for member in members:
        member_id = str(member["member_id"])
        member_dir = output_dir / "members" / member_id
        feature_columns = load_feature_columns(member_dir)

        for feature in feature_columns:
            if feature in MARKET_FEATURES:
                continue

            if feature.startswith("favpersp__"):
                clean_feature = feature.removeprefix("favpersp__")
            elif feature.startswith("dogpersp__"):
                clean_feature = feature.removeprefix("dogpersp__")
            else:
                clean_feature = feature

            if clean_feature not in clean_features:
                clean_features.append(clean_feature)

    if not clean_features:
        raise ValueError("No non-market live-builder feature columns resolved for ensemble.")

    return clean_features


def score_members(
    *,
    df: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any] | None,
    prefer_calibrated: bool,
) -> pd.DataFrame:
    out = df.copy()
    output_dir = get_output_dir(config)

    members = member_configs_from_manifest_or_config(config=config, manifest=manifest)
    if not members:
        raise ValueError("No ensemble members found in config/manifest.")

    for member in members:
        member_id = str(member["member_id"])
        probability_group = str(member.get("probability_group", ""))
        weight = float(member.get("weight", 1.0))

        member_dir = output_dir / "members" / member_id
        model, model_path = load_member_model(member_dir, prefer_calibrated=prefer_calibrated)
        feature_columns = load_feature_columns(member_dir)
        X = build_member_feature_matrix(out, feature_columns)

        probabilities = predict_positive_class_probability(model, X)

        out[f"member_probability__{member_id}"] = probabilities
        out[f"member_weight__{member_id}"] = weight
        out[f"member_group__{member_id}"] = probability_group
        out[f"member_model_path__{member_id}"] = str(model_path)

    return out


def combine_member_probabilities(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    members = (config.get("ensemble") or {}).get("members") or []

    for group_name in ["favorite", "dog"]:
        weighted = 0.0
        weight_sum = 0.0
        member_count = 0

        for member in members:
            member_id = str(member["member_id"])
            if str(member.get("probability_group")) != group_name:
                continue

            prob_col = f"member_probability__{member_id}"
            weight_col = f"member_weight__{member_id}"
            if prob_col not in out.columns:
                continue

            weight = safe_numeric(series(out, weight_col)).fillna(float(member.get("weight", 1.0)))
            weighted = weighted + safe_numeric(series(out, prob_col)).fillna(0.0) * weight
            weight_sum = weight_sum + weight
            member_count += 1

        if member_count == 0:
            raise ValueError(f"No scored members found for probability group: {group_name}")

        out[f"ensemble_{group_name}_probability"] = weighted / weight_sum
        out[f"ensemble_{group_name}_member_count"] = member_count

    out["ensemble_favorite_edge"] = out["ensemble_favorite_probability"] - out["favorite_implied_probability"]
    out["ensemble_dog_edge"] = out["ensemble_dog_probability"] - out["dog_implied_probability"]

    return out


def apply_decision_rule(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    rule = config.get("decision_rule", {}) or {}

    fav_rule = rule.get("favorite", {}) or {}
    dog_rule = rule.get("dog", {}) or {}

    out["favorite_qualifies_rule"] = (
        out["ensemble_favorite_probability"].ge(float(fav_rule.get("favorite_probability_min", 0.0)))
        & out["ensemble_dog_probability"].le(float(fav_rule.get("dog_probability_max", 1.0)))
        & out["ensemble_favorite_edge"].ge(float(fav_rule.get("favorite_edge_min", -1.0)))
    )

    out["dog_qualifies_rule"] = (
        out["ensemble_favorite_probability"].le(float(dog_rule.get("favorite_probability_max", 1.0)))
        & out["ensemble_dog_probability"].ge(float(dog_rule.get("dog_probability_min", 0.0)))
        & out["ensemble_dog_edge"].ge(float(dog_rule.get("dog_edge_min", -1.0)))
        & out["dog_odds"].le(float(dog_rule.get("dog_odds_max", 999999)))
    )

    out["ensemble_pick_side"] = np.where(
        out["ensemble_favorite_probability"].ge(out["ensemble_dog_probability"]),
        "favorite",
        "dog",
    )
    out["ensemble_pick_probability"] = np.where(
        out["ensemble_pick_side"].eq("favorite"),
        out["ensemble_favorite_probability"],
        out["ensemble_dog_probability"],
    )
    out["ensemble_pick_fighter"] = np.where(
        out["ensemble_pick_side"].eq("favorite"),
        out["favorite_fighter_name"],
        out["dog_fighter_name"],
    )
    out["ensemble_pick_edge"] = np.where(
        out["ensemble_pick_side"].eq("favorite"),
        out["ensemble_favorite_edge"],
        out["ensemble_dog_edge"],
    )
    out["ensemble_pick_qualifies_rule"] = np.where(
        out["ensemble_pick_side"].eq("favorite"),
        out["favorite_qualifies_rule"],
        out["dog_qualifies_rule"],
    )

    return out


def build_model_outcomes(details: pd.DataFrame, config: dict[str, Any], prediction_run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    model_id = get_model_id(config)
    model_family = str(config.get("model_family", "moneyline"))
    algorithm = str(config.get("algorithm", "xgboost"))

    quality_cols = [
        "passes_model_data_quality",
        "passes_feature_validation",
        "nonzero_feature_count",
        "zero_feature_pct",
        "feature_count_expected",
        "feature_count_actual",
        "red_feature_match",
        "blue_feature_match",
        "feature_match_type",
        "confidence_score",
        "confidence_pct",
        "confidence_tier",
        "confidence_data_quality",
        "confidence_feature_coverage",
        "confidence_family_coverage",
        "confidence_history_depth",
        "confidence_calibration_reliability",
        "confidence_bucket",
        "confidence_bucket_accuracy",
        "confidence_bucket_fight_count",
        "confidence_bucket_calibration_error",
        "confidence_penalty_reason",
    ]

    for _, row in details.iterrows():
        side_specs = [
            {
                "ensemble_side": "favorite",
                "fighter_id": row["favorite_fighter_id"],
                "fighter_name": row["favorite_fighter_name"],
                "actual_side": row["favorite_side"],
                "probability": row["ensemble_favorite_probability"],
                "edge": row["ensemble_favorite_edge"],
                "odds": row["favorite_odds"],
                "implied": row["favorite_implied_probability"],
                "qualifies": row["favorite_qualifies_rule"],
                "bookmaker": row["favorite_bookmaker"],
            },
            {
                "ensemble_side": "dog",
                "fighter_id": row["dog_fighter_id"],
                "fighter_name": row["dog_fighter_name"],
                "actual_side": row["dog_side"],
                "probability": row["ensemble_dog_probability"],
                "edge": row["ensemble_dog_edge"],
                "odds": row["dog_odds"],
                "implied": row["dog_implied_probability"],
                "qualifies": row["dog_qualifies_rule"],
                "bookmaker": row["dog_bookmaker"],
            },
        ]

        model_pick_fighter = row["ensemble_pick_fighter"]
        model_pick_probability = row["ensemble_pick_probability"]

        confidence_payload = score_prediction_confidence(
            row,
            model_pick_probability=float(model_pick_probability),
        ).to_dict()

        for spec in side_specs:
            output_row = {
                "prediction_run_id": prediction_run_id,
                "prediction_timestamp": now,
                "model_id": model_id,
                "model_family": model_family,
                "algorithm": algorithm,
                "prediction_type": "binary_matchup_ensemble",
                "event_id": row.get("event_id"),
                "event_name": row.get("event_name"),
                "commence_time": row.get("commence_time"),
                "fight_id": row.get("fight_id"),
                "red_fighter": row.get("red_fighter"),
                "blue_fighter": row.get("blue_fighter"),
                "red_fighter_id": row.get("red_fighter_id"),
                "blue_fighter_id": row.get("blue_fighter_id"),
                "market_key": "moneyline",
                "outcome_label": spec["fighter_name"],
                "outcome_join_key": f"fighter:{spec['fighter_id']}",
                "outcome_fighter_id": spec["fighter_id"],
                "outcome_side": spec["actual_side"],
                "ensemble_side": spec["ensemble_side"],
                "model_probability": float(spec["probability"]),
                "model_pick_probability": float(model_pick_probability),
                "is_model_pick": bool(spec["fighter_name"] == model_pick_fighter),
                "model_pick": model_pick_fighter,
                "model_confidence": float(model_pick_probability),
                "bookmaker": spec["bookmaker"],
                "american_odds": float(spec["odds"]),
                "implied_probability": float(spec["implied"]),
                "model_edge": float(spec["edge"]),
                "qualifies_decision_rule": bool(spec["qualifies"]),
                "ensemble_pick_side": row["ensemble_pick_side"],
                "ensemble_pick_edge": float(row["ensemble_pick_edge"]),
                "ensemble_pick_qualifies_rule": bool(row["ensemble_pick_qualifies_rule"]),
            }

            for col in quality_cols:
                if col in details.columns:
                    output_row[col] = row.get(col)

            output_row.update(confidence_payload)

            rows.append(output_row)

    return pd.DataFrame(rows)


def write_outputs(
    *,
    details: pd.DataFrame,
    exclusions: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
    write_canonical: bool,
) -> tuple[Path, Path, Path]:
    scoped_path, details_path = get_prediction_output_paths(config)
    exclusions_path = details_path.with_name("ensemble_exclusions.parquet")
    canonical_path = canonical_output_path(config)

    scoped_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)

    outcomes.to_parquet(scoped_path, index=False)
    details.to_parquet(details_path, index=False)
    exclusions.to_parquet(exclusions_path, index=False)

    if write_canonical:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        outcomes.to_parquet(canonical_path, index=False)

    return scoped_path, details_path, exclusions_path


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    live_features_path = Path(args.live_features)
    market_outcomes_path = Path(args.market_outcomes)

    config = load_yaml(config_path)
    model_id = get_model_id(config)
    output_dir = get_output_dir(config)
    manifest_path = output_dir / "ensemble_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else None

    prediction_run_id = f"pred_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{model_id}"

    print("=" * 80)
    print("RUN UFC ENSEMBLE PREDICTION")
    print("=" * 80)
    print(f"Config path         : {config_path}")
    print(f"Model ID            : {model_id}")
    print(f"Output dir          : {output_dir}")
    print(f"Manifest path       : {manifest_path}")
    print(f"Live features path  : {live_features_path}")
    print(f"Market outcomes path: {market_outcomes_path}")
    print(f"Prediction run ID   : {prediction_run_id}")

    if not market_outcomes_path.exists():
        raise FileNotFoundError(market_outcomes_path)
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    live_builder_features = resolve_live_builder_feature_columns(
        config=config,
        manifest=manifest,
    )

    print(f"Live builder features: {len(live_builder_features)}")

    # Rebuild live features with the exact clean feature union required by the
    # ensemble members. This reuses the production Prediction V2 feature builder,
    # including its existing *_diff and *_abs_diff materialization logic.
    live_result = build_live_model_features(
        feature_columns=live_builder_features,
    )
    write_live_feature_outputs(
        live_result,
        live_feature_output_path=live_features_path,
    )

    live_df = live_result.live_feature_df
    market_df = pd.read_parquet(market_outcomes_path)

    print(f"Live feature shape  : {live_df.shape}")
    print(f"Live feature output : {live_features_path}")
    print(f"Market shape        : {market_df.shape}")

    usable, exclusions = join_live_market(live_df, market_df)
    print(f"Rows with both odds : {len(usable)}")
    print(f"Excluded rows       : {len(exclusions)}")

    if usable.empty:
        raise ValueError("No live fights had both red and blue moneyline odds.")

    details = add_favorite_dog_market_columns(usable)
    prefer_calibrated = prefer_calibrated_from_config(config)
    if args.raw:
        prefer_calibrated = False

    probability_artifact = "calibrated_model" if prefer_calibrated else "raw_model"
    print(f"Probability artifact: {probability_artifact}")

    details = score_members(
        df=details,
        config=config,
        manifest=manifest,
        prefer_calibrated=prefer_calibrated,
    )
    details = combine_member_probabilities(details, config=config)
    details = apply_decision_rule(details, config=config)

    outcomes = build_model_outcomes(
        details=details,
        config=config,
        prediction_run_id=prediction_run_id,
    )

    scoped_path, details_path, exclusions_path = write_outputs(
        details=details,
        exclusions=exclusions,
        outcomes=outcomes,
        config=config,
        write_canonical=args.write_canonical,
    )

    print()
    print("=" * 80)
    print("ENSEMBLE PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Scored fights       : {len(details)}")
    print(f"Outcome rows        : {len(outcomes)}")
    print(f"Favorite rule hits  : {int(details['favorite_qualifies_rule'].sum())}")
    print(f"Dog rule hits       : {int(details['dog_qualifies_rule'].sum())}")
    print(f"Model picks by side :")
    print(details["ensemble_pick_side"].value_counts(dropna=False).to_string())

    print()
    show_cols = [
        "event_name",
        "red_fighter",
        "blue_fighter",
        "favorite_fighter_name",
        "dog_fighter_name",
        "favorite_odds",
        "dog_odds",
        "ensemble_favorite_probability",
        "ensemble_dog_probability",
        "ensemble_favorite_edge",
        "ensemble_dog_edge",
        "favorite_qualifies_rule",
        "dog_qualifies_rule",
        "ensemble_pick_fighter",
        "ensemble_pick_probability",
        "ensemble_pick_edge",
        "ensemble_pick_qualifies_rule",
    ]
    print(details[show_cols].sort_values("ensemble_pick_edge", ascending=False).to_string(index=False))

    print()
    print(f"Saved scoped outcomes : {scoped_path}")
    print(f"Saved ensemble details: {details_path}")
    print(f"Saved exclusions      : {exclusions_path}")
    if args.write_canonical:
        print(f"Saved canonical       : {canonical_output_path(config)}")
    else:
        print("Saved canonical       : skipped")
    print("DONE")


if __name__ == "__main__":
    main()
