from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

import utils.model_lab_workflows as mlw


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _artifact_path(context: dict[str, Any], filename: str) -> Path:
    return Path(str(context.get("artifact_dir") or "")) / filename


def _is_multiclass_model(context: dict[str, Any]) -> bool:
    prediction = (context.get("config") or {}).get("prediction") or {}
    return str(prediction.get("format") or "").strip().lower() == "multiclass"


def _read_parquet_artifact(context: dict[str, Any], filename: str) -> pd.DataFrame:
    path = _artifact_path(context, filename)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _format_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["actual_rate", "predicted_rate", "mean_probability", "one_vs_rest_f1"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def render_multiclass_summary(context: dict[str, Any], *, compact: bool = False) -> None:
    """Render multiclass artifacts for models that emit class probability rows."""

    if not _is_multiclass_model(context):
        return

    class_metrics_path = _artifact_path(context, "class_metrics.parquet")
    confusion_path = _artifact_path(context, "confusion_matrix.parquet")
    class_metrics = _format_rate_columns(_read_parquet_artifact(context, "class_metrics.parquet"))
    confusion = _read_parquet_artifact(context, "confusion_matrix.parquet")

    st.markdown("### Multiclass Breakdown")
    if class_metrics.empty and confusion.empty:
        st.info(
            "No multiclass class artifacts found yet. Expected "
            f"`{class_metrics_path}` and `{confusion_path}` after training."
        )
        return

    if not class_metrics.empty:
        st.caption(f"Class metrics loaded from `{class_metrics_path}`")
        display_cols = [
            column
            for column in [
                "class_index",
                "class_label",
                "support",
                "predicted_count",
                "actual_rate",
                "predicted_rate",
                "mean_probability",
                "one_vs_rest_f1",
            ]
            if column in class_metrics.columns
        ]
        st.dataframe(class_metrics[display_cols], use_container_width=True, hide_index=True)

        chart_cols = [column for column in ["actual_rate", "predicted_rate", "mean_probability", "one_vs_rest_f1"] if column in class_metrics.columns]
        if chart_cols and "class_label" in class_metrics.columns and not compact:
            st.bar_chart(class_metrics.set_index("class_label")[chart_cols])

    if compact:
        return

    st.markdown("### Confusion Matrix")
    if confusion.empty:
        st.info(f"No confusion matrix artifact found yet: `{confusion_path}`")
        return

    st.caption(f"Confusion matrix loaded from `{confusion_path}`. Rows are actual classes; columns are predicted classes.")
    st.dataframe(confusion, use_container_width=True, hide_index=True)

    if "actual_class" in confusion.columns:
        numeric = confusion.set_index("actual_class")
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        st.dataframe(
            numeric.style.background_gradient(axis=None),
            use_container_width=True,
        )


def _read_shap_importance(context: dict[str, Any]) -> pd.DataFrame:
    """Load SHAP importance for the selected model artifact directory."""

    artifact_dir = Path(str(context.get("artifact_dir") or ""))
    path = artifact_dir / "shap_importance.csv"
    if not artifact_dir or not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "mean_abs_shap" in df.columns:
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    elif "importance" in df.columns:
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def _render_shap_importance(context: dict[str, Any]) -> None:
    """Render SHAP feature importance inside the Performance workspace."""

    st.markdown("### SHAP Analysis")
    shap_df = _read_shap_importance(context)
    shap_path = Path(str(context.get("artifact_dir") or "")) / "shap_importance.csv"

    if shap_df.empty:
        st.info(f"No SHAP artifact found yet: `{shap_path}`")
        return

    feature_col = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
    value_col = None
    for candidate in ["mean_abs_shap", "importance", "gain", "weight"]:
        if candidate in shap_df.columns:
            value_col = candidate
            break

    st.caption(f"Loaded `{shap_path}` · {len(shap_df):,} features")

    top_n = st.slider(
        "Top SHAP features",
        min_value=10,
        max_value=min(100, max(10, len(shap_df))),
        value=min(25, len(shap_df)),
        step=5,
        key=f"performance_shap_top_n_{context.get('model_id')}",
    )
    top_df = shap_df.head(top_n).copy()

    if value_col:
        chart_df = top_df[[feature_col, value_col]].set_index(feature_col)
        st.bar_chart(chart_df)

    st.dataframe(top_df, use_container_width=True, hide_index=True)


def render_performance(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    st.markdown("## Performance")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_model_bar(context, registry)
    mlw._render_performance(context)
    render_multiclass_summary(context)
    _render_shap_importance(context)
