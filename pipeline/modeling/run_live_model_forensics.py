from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from pipeline.common.paths import AUDITS_DIR, PREDICTIONS_DIR
from pipeline.modeling.model_config import get_model_id, load_model_config
from pipeline.modeling.model_loader import load_model_bundle

DEFAULT_MODEL_CONFIG_PATH = "configs/models/moneyline_xgboost_v6_dev.yaml"
DEFAULT_LIVE_FEATURES_PATH = PREDICTIONS_DIR / "live_model_features.parquet"
DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_live_model_forensics.parquet"
DEFAULT_PREVIEW_PATH = AUDITS_DIR / "ufc_live_model_forensics_preview.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain live model behavior for one fight.")
    parser.add_argument("--fighter-a", default="Michael Chandler")
    parser.add_argument("--fighter-b", default="Mauricio Ruffy")
    parser.add_argument("--model-config-path", default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--live-features-path", default=str(DEFAULT_LIVE_FEATURES_PATH))
    parser.add_argument("--model-outcomes-path", default=str(DEFAULT_MODEL_OUTCOMES_PATH))
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-path", default=str(DEFAULT_PREVIEW_PATH))
    return parser.parse_args()


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def contains_fighter(row: pd.Series, fighter: str) -> bool:
    target = norm(fighter)
    name_cols = [
        "red_fighter",
        "blue_fighter",
        "r_name",
        "b_name",
        "fighter_name",
        "outcome_label",
        "outcome_label_model",
    ]
    return any(target in norm(row.get(col)) for col in name_cols if col in row.index)


def filter_fight(df: pd.DataFrame, fighter_a: str, fighter_b: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask_a = df.apply(lambda row: contains_fighter(row, fighter_a), axis=1)
    mask_b = df.apply(lambda row: contains_fighter(row, fighter_b), axis=1)
    return df.loc[mask_a & mask_b].copy()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": pd.to_datetime(stat.st_mtime, unit="s", utc=True).isoformat(),
        "sha256": sha256_file(path),
    }


def unwrap_estimator(model: Any) -> Any:
    """Best-effort unwrap calibrated/sklearn containers to the underlying estimator."""
    # CalibratedClassifierCV often stores calibrated_classifiers_ entries.
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        first = calibrated[0]
        for attr in ["estimator", "base_estimator", "classifier"]:
            inner = getattr(first, attr, None)
            if inner is not None:
                return unwrap_estimator(inner)

    # FrozenEstimator / wrappers often expose estimator.
    for attr in ["estimator", "base_estimator", "classifier"]:
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            return unwrap_estimator(inner)

    return model


def predict_probability(model: Any, x: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(x)
    if proba is None or len(proba) == 0:
        return None
    if getattr(proba, "shape", (0, 0))[1] >= 2:
        return float(proba[0, 1])
    return float(proba[0][0])


def extract_feature_importance(estimator: Any, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(estimator, "feature_importances_"):
        vals = list(getattr(estimator, "feature_importances_"))
        return pd.DataFrame(
            {
                "feature": feature_columns[: len(vals)],
                "importance": vals,
                "importance_source": "feature_importances_",
            }
        ).sort_values("importance", ascending=False)

    booster = getattr(estimator, "get_booster", lambda: None)()
    if booster is not None:
        score = booster.get_score(importance_type="gain")
        rows = []
        for key, value in score.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                feature = feature_columns[idx] if idx < len(feature_columns) else key
            else:
                feature = key
            rows.append({"feature": feature, "importance": value, "importance_source": "booster_gain"})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance", "importance_source"])


def try_shap(estimator: Any, x: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap  # type: ignore
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "row_type": "shap_error",
                    "feature": "SHAP unavailable",
                    "diagnostic_value": str(exc),
                }
            ]
        )

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(x)
        base_value = explainer.expected_value
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        if hasattr(values, "values"):
            vals = values.values
            base_value = values.base_values
        else:
            vals = values
        vals = vals[0]
        base = base_value[1] if isinstance(base_value, (list, tuple)) and len(base_value) > 1 else base_value
        if hasattr(base, "__len__") and not isinstance(base, (str, bytes)):
            try:
                base = base[0]
            except Exception:
                pass
        out = pd.DataFrame(
            {
                "row_type": "shap",
                "feature": list(x.columns),
                "feature_value": x.iloc[0].values,
                "shap_value": vals,
                "abs_shap_value": [abs(float(v)) for v in vals],
                "base_value": float(base) if base is not None and not isinstance(base, str) else pd.NA,
            }
        )
        return out.sort_values("abs_shap_value", ascending=False)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "row_type": "shap_error",
                    "feature": "SHAP failed",
                    "diagnostic_value": str(exc),
                }
            ]
        )


def main() -> None:
    args = parse_args()
    model_config_path = Path(args.model_config_path)
    live_features_path = Path(args.live_features_path)
    model_outcomes_path = Path(args.model_outcomes_path)

    config = load_model_config(model_config_path, require_prediction=True)
    bundle = load_model_bundle(config, prefer_calibrated=True)
    model_id = get_model_id(config)

    live_features = pd.read_parquet(live_features_path)
    live_match = filter_fight(live_features, args.fighter_a, args.fighter_b)
    if live_match.empty:
        raise ValueError(f"No live feature row matched {args.fighter_a!r} vs {args.fighter_b!r}")
    live_row = live_match.iloc[[0]].copy()

    missing_features = [c for c in bundle.feature_columns if c not in live_row.columns]
    if missing_features:
        raise ValueError(f"Live row missing model features: {missing_features[:25]} total={len(missing_features)}")

    x = live_row[bundle.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    loaded_prob = predict_probability(bundle.model, x)
    estimator = unwrap_estimator(bundle.model)
    raw_prob = predict_probability(estimator, x)

    outcomes = pd.read_parquet(model_outcomes_path)
    fight_ids = set(live_match["fight_id"].astype(str)) if "fight_id" in live_match.columns else set()
    if fight_ids and "fight_id" in outcomes.columns:
        outcome_match = outcomes[outcomes["fight_id"].astype(str).isin(fight_ids)].copy()
    else:
        outcome_match = filter_fight(outcomes, args.fighter_a, args.fighter_b)

    artifact_info = file_info(bundle.model_artifact_path)
    raw_info = file_info(bundle.artifact_dir / "raw_model.joblib")
    calibrated_info = file_info(bundle.artifact_dir / "calibrated_model.joblib")
    features_json_info = file_info(bundle.artifact_dir / "feature_columns.json")
    features_joblib_info = file_info(bundle.artifact_dir / "feature_columns.joblib")
    metrics_info = file_info(bundle.artifact_dir / "metrics.json")

    metadata_rows = [
        {
            "row_type": "metadata",
            "model_id": model_id,
            "model_config_path": str(model_config_path),
            "artifact_dir": str(bundle.artifact_dir),
            "model_artifact_path": str(bundle.model_artifact_path),
            "uses_calibrated_model": bundle.uses_calibrated_model,
            "feature_count": len(bundle.feature_columns),
            "live_features_path": str(live_features_path),
            "model_outcomes_path": str(model_outcomes_path),
            "fight_id": live_row.iloc[0].get("fight_id"),
            "red_fighter": live_row.iloc[0].get("red_fighter"),
            "blue_fighter": live_row.iloc[0].get("blue_fighter"),
            "loaded_model_probability_red": loaded_prob,
            "raw_estimator_probability_red": raw_prob,
            "artifact_sha256": artifact_info.get("sha256"),
            "artifact_mtime_utc": artifact_info.get("mtime_utc"),
            "artifact_size_bytes": artifact_info.get("size_bytes"),
            "raw_model_sha256": raw_info.get("sha256"),
            "calibrated_model_sha256": calibrated_info.get("sha256"),
            "feature_columns_json_sha256": features_json_info.get("sha256"),
            "feature_columns_joblib_sha256": features_joblib_info.get("sha256"),
            "metrics_json_sha256": metrics_info.get("sha256"),
        }
    ]

    importance = extract_feature_importance(estimator, bundle.feature_columns).head(args.top_n).copy()
    if not importance.empty:
        importance.insert(0, "row_type", "importance")

    shap_df = try_shap(estimator, x).head(args.top_n).copy()

    outcome_rows = []
    outcome_cols = [
        "model_id",
        "prediction_run_id",
        "event_name",
        "fight_id",
        "outcome_label",
        "outcome_fighter_id",
        "model_probability",
        "confidence_score",
        "model_confidence",
    ]
    for _, row in outcome_match.iterrows():
        outcome_rows.append({"row_type": "outcome", **{c: row.get(c) for c in outcome_cols if c in row.index}})

    combined = pd.concat(
        [pd.DataFrame(metadata_rows), importance, shap_df, pd.DataFrame(outcome_rows)],
        ignore_index=True,
        sort=False,
    )

    output_path = Path(args.output_path)
    preview_path = Path(args.preview_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    combined.to_csv(preview_path, index=False)

    print("=" * 80)
    print("LIVE MODEL FORENSICS")
    print("=" * 80)
    print("Fighter A:", args.fighter_a)
    print("Fighter B:", args.fighter_b)
    print("Model ID:", model_id)
    print("Config path:", model_config_path)
    print("Artifact dir:", bundle.artifact_dir)
    print("Model artifact:", bundle.model_artifact_path)
    print("Uses calibrated model:", bundle.uses_calibrated_model)
    print("Feature count:", len(bundle.feature_columns))
    print("Artifact mtime UTC:", artifact_info.get("mtime_utc"))
    print("Artifact sha256:", artifact_info.get("sha256"))
    print("Loaded model P(red):", loaded_prob)
    print("Raw estimator P(red):", raw_prob)
    print("Red fighter:", live_row.iloc[0].get("red_fighter"))
    print("Blue fighter:", live_row.iloc[0].get("blue_fighter"))
    print()
    print("========== MODEL OUTCOMES ==========")
    if outcome_match.empty:
        print("No model outcome rows found.")
    else:
        print(outcome_match[[c for c in outcome_cols if c in outcome_match.columns]].to_string(index=False))
    print()
    print("========== TOP MODEL IMPORTANCES ==========")
    if importance.empty:
        print("No feature importances available.")
    else:
        print(importance[["feature", "importance", "importance_source"]].head(args.top_n).to_string(index=False))
    print()
    print("========== TOP SHAP CONTRIBUTIONS ==========")
    if shap_df.empty:
        print("No SHAP output.")
    else:
        cols = [c for c in ["feature", "feature_value", "shap_value", "abs_shap_value", "base_value", "diagnostic_value"] if c in shap_df.columns]
        print(shap_df[cols].to_string(index=False))
    print()
    print("Saved forensics:", output_path)
    print("Saved preview:", preview_path)


if __name__ == "__main__":
    main()
