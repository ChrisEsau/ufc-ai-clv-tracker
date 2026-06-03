from pathlib import Path
import json
import pickle

import pandas as pd


MODEL_DIR = Path("UFC_Model_v5_Experiment")

MODEL_ARTIFACTS = {
    "Production Config JSON": MODEL_DIR / "production_config.json",
    "Production Config Pickle": MODEL_DIR / "production_config.pkl",
    "Feature Columns": MODEL_DIR / "feature_columns.pkl",
    "Best Threshold": MODEL_DIR / "best_threshold.pkl",
    "Calibrated Model": MODEL_DIR / "calibrated_model.pkl",
    "Raw Model": MODEL_DIR / "raw_model.pkl",
    "Model Quality Summary": MODEL_DIR / "model_quality_summary.csv",
    "SHAP Importance": MODEL_DIR / "shap_importance.csv",
}

LIVE_AUDIT_ARTIFACTS = {
    "Model Predictions": Path("ufc_model_predictions.parquet"),
    "Live Feature Audit": Path("ufc_live_feature_audit.parquet"),
    "Live Match Audit": Path("ufc_live_match_audit.parquet"),
}


def format_file_size(size_bytes):
    if size_bytes is None:
        return ""

    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"

        size /= 1024

    return f"{size:.1f} TB"


def artifact_status_rows(artifacts):
    rows = []

    for artifact_name, path in artifacts.items():
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else None

        rows.append(
            {
                "artifact": artifact_name,
                "path": str(path),
                "exists": exists,
                "size": format_file_size(size_bytes),
                "size_bytes": size_bytes,
                "status": "Ready" if exists else "Missing",
            }
        )

    return rows


def get_model_artifact_status():
    return pd.DataFrame(artifact_status_rows(MODEL_ARTIFACTS))


def get_live_audit_artifact_status():
    return pd.DataFrame(artifact_status_rows(LIVE_AUDIT_ARTIFACTS))


def load_json(path):
    path = Path(path)

    if not path.exists():
        return {}, f"Missing file: {path}"

    with path.open("r", encoding="utf-8") as f:
        return json.load(f), None


def load_pickle(path):
    path = Path(path)

    if not path.exists():
        return None, f"Missing file: {path}"

    with path.open("rb") as f:
        return pickle.load(f), None


def load_csv(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(), f"Missing file: {path}"

    return pd.read_csv(path), None


def load_parquet(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(), f"Missing file: {path}"

    return pd.read_parquet(path), None


def load_production_config():
    return load_json(MODEL_ARTIFACTS["Production Config JSON"])


def load_feature_columns():
    feature_columns, error = load_pickle(MODEL_ARTIFACTS["Feature Columns"])

    if error:
        return [], error

    return list(feature_columns), None


def load_best_threshold():
    threshold, error = load_pickle(MODEL_ARTIFACTS["Best Threshold"])

    if error:
        return None, error

    return threshold, None


def load_model_quality_summary():
    return load_csv(MODEL_ARTIFACTS["Model Quality Summary"])


def load_shap_importance():
    df, error = load_csv(MODEL_ARTIFACTS["SHAP Importance"])

    if error or df.empty:
        return df, error

    if "mean_abs_shap" in df.columns:
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df, None


def load_model_predictions():
    return load_parquet(LIVE_AUDIT_ARTIFACTS["Model Predictions"])


def load_live_feature_audit():
    return load_parquet(LIVE_AUDIT_ARTIFACTS["Live Feature Audit"])


def load_live_match_audit():
    return load_parquet(LIVE_AUDIT_ARTIFACTS["Live Match Audit"])


def quality_metric_value(quality_df, metric_name):
    if quality_df.empty or not {"metric", "value"}.issubset(quality_df.columns):
        return None

    matches = quality_df.loc[quality_df["metric"] == metric_name, "value"]

    if matches.empty:
        return None

    return matches.iloc[0]


def percent_text(value):
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value) * 100:.2f}%"


def number_text(value, decimals=3):
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):,.{decimals}f}"


def integer_text(value):
    if value is None or pd.isna(value):
        return "—"

    return f"{int(float(value)):,}"
