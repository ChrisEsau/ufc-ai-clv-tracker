from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

import utils.model_lab_workflows as mlw


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _artifact_dir(context: dict[str, Any]) -> Path:
    return Path(str(context.get("artifact_dir") or ""))


def _artifact_path(context: dict[str, Any], filename: str) -> Path:
    return _artifact_dir(context) / filename


def _safe_label_for_filename(label: str) -> str:
    token = str(label).strip().lower()
    for char in ["/", "\\", " ", "-", "."]:
        token = token.replace(char, "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "class"


def _read_class_labels_artifact(context: dict[str, Any]) -> list[str]:
    path = _artifact_path(context, "class_labels.json")
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(label) for label in payload if str(label).strip()]


def _class_labels(context: dict[str, Any]) -> list[str]:
    artifact_labels = _read_class_labels_artifact(context)
    if artifact_labels:
        return artifact_labels
    prediction = (context.get("config") or {}).get("prediction") or {}
    labels = prediction.get("class_labels") or prediction.get("classes") or []
    return [str(label) for label in labels if str(label).strip()]


def _discover_class_shap_files(context: dict[str, Any]) -> dict[str, str]:
    discovered: dict[str, str] = {}
    for label in _class_labels(context):
        discovered[label] = f"shap_importance_{_safe_label_for_filename(label)}.csv"
    artifact_dir = _artifact_dir(context)
    if artifact_dir.exists():
        for path in sorted(artifact_dir.glob("shap_importance_*.csv")):
            if path.name == "shap_importance.csv":
                continue
            suffix = path.stem.replace("shap_importance_", "", 1)
            matching_label = next((label for label in discovered if _safe_label_for_filename(label) == suffix), suffix)
            discovered.setdefault(matching_label, path.name)
    return discovered


def _is_multiclass_model(context: dict[str, Any]) -> bool:
    prediction = (context.get("config") or {}).get("prediction") or {}
    return str(prediction.get("format") or "").strip().lower() == "multiclass" or bool(_read_class_labels_artifact(context)) or bool(_discover_class_shap_files(context))


def _read_parquet_artifact(context: dict[str, Any], filename: str) -> pd.DataFrame:
    path = _artifact_path(context, filename)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _read_csv_artifact(context: dict[str, Any], filename: str) -> pd.DataFrame:
    path = _artifact_path(context, filename)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        if "mean_abs_shap" in df.columns:
            df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        elif "importance" in df.columns:
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _format_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["actual_rate", "predicted_rate", "mean_probability", "one_vs_rest_f1"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def render_multiclass_summary(context: dict[str, Any], *, compact: bool = False) -> None:
    if not _is_multiclass_model(context):
        return
    class_metrics = _format_rate_columns(_read_parquet_artifact(context, "class_metrics.parquet"))
    confusion = _read_parquet_artifact(context, "confusion_matrix.parquet")
    st.markdown("### Multiclass Breakdown")
    if not class_metrics.empty:
        st.dataframe(class_metrics, use_container_width=True, hide_index=True)
    if not compact and not confusion.empty:
        st.markdown("### Confusion Matrix")
        st.dataframe(confusion, use_container_width=True, hide_index=True)


def _read_shap_importance(context: dict[str, Any]) -> pd.DataFrame:
    return _read_csv_artifact(context, "shap_importance.csv")


def _render_shap_table(context: dict[str, Any], shap_df: pd.DataFrame, *, title: str, key_suffix: str, source_name: str) -> None:
    st.markdown(f"#### {title}")
    if shap_df.empty:
        return
    feature_col = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
    value_col = next((c for c in ["mean_abs_shap", "importance", "gain", "weight"] if c in shap_df.columns), None)
    show_all = st.toggle(f"Show all SHAP features · {title}", value=True, key=f"performance_shap_show_all_{context.get('model_id')}_{key_suffix}")
    top_df = shap_df if show_all else shap_df.head(25)
    if value_col:
        chart_n = min(50, len(shap_df))
        chart_df = shap_df.head(chart_n)[[feature_col, value_col]].set_index(feature_col)
        st.bar_chart(chart_df)
        table_count = "all" if show_all else f"{len(top_df):,}"
        st.caption(f"Chart shows top {chart_n:,}; table shows {table_count} features.")
    st.dataframe(top_df, use_container_width=True, hide_index=True)


def _render_shap_importance(context: dict[str, Any]) -> None:
    st.markdown("### SHAP Analysis")
    overall = _read_shap_importance(context)
    if not _is_multiclass_model(context):
        _render_shap_table(context, overall, title="Overall SHAP Importance", key_suffix="overall", source_name="shap_importance.csv")
        return
    class_files = _discover_class_shap_files(context)
    tabs = st.tabs(["Overall"] + [str(label) for label in class_files])
    with tabs[0]:
        _render_shap_table(context, overall, title="Overall SHAP Importance", key_suffix="overall", source_name="shap_importance.csv")
    for tab, (label, filename) in zip(tabs[1:], class_files.items()):
        with tab:
            _render_shap_table(context, _read_csv_artifact(context, filename), title=f"{label} SHAP Importance", key_suffix=_safe_label_for_filename(label), source_name=filename)


def render_performance(registry, rows, row_by_id, *, existing_model_selector):
    st.markdown("## Performance")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_model_bar(context, registry)
    mlw._render_performance(context)
    render_multiclass_summary(context)
    _render_shap_importance(context)
