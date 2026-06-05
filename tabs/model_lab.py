import pandas as pd
import streamlit as st

from utils.ui.sections import page_header

from utils.model_lab_artifacts import (
    get_live_audit_artifact_status,
    get_model_artifact_status,
    integer_text,
    load_best_threshold,
    load_feature_columns,
    load_live_feature_audit,
    load_live_match_audit,
    load_model_predictions,
    load_model_quality_summary,
    load_production_config,
    load_shap_importance,
    number_text,
    percent_text,
    quality_metric_value,
)


def render_model_lab():
    page_header(
        "Model Lab",
        "Read-only diagnostics for production model quality, feature importance, and live prediction audits.",
        kicker="Research Workspace",
    )

    render_model_artifact_status()
    render_model_quality()
    render_feature_importance()
    render_live_prediction_audit()


def render_model_artifact_status():
    with st.expander("📦 Model Artifact Status", expanded=True):
        config, config_error = load_production_config()
        feature_columns, feature_error = load_feature_columns()
        best_threshold, threshold_error = load_best_threshold()
        artifact_status = get_model_artifact_status()

        if config_error:
            st.warning(config_error)

        if feature_error:
            st.warning(feature_error)

        if threshold_error:
            st.warning(threshold_error)

        cols = st.columns(5)
        cols[0].metric("Model Version", config.get("version", "—"))
        cols[1].metric("Model Type", config.get("model_type", "—"))
        cols[2].metric("Train End", config.get("train_end_date", "—"))
        cols[3].metric("Feature Count", len(feature_columns) or config.get("feature_count", "—"))
        cols[4].metric("Best Threshold", number_text(best_threshold, decimals=2))

        st.caption(f"Created at: {config.get('created_at', '—')}")

        display_cols = ["artifact", "path", "exists", "size", "status"]
        st.dataframe(
            artifact_status[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        missing = artifact_status[~artifact_status["exists"]]

        if missing.empty:
            st.success("All expected production model artifacts are present.")
        else:
            st.error(f"Missing model artifacts: {len(missing)}")

        if feature_columns:
            with st.expander("Feature Column Registry Preview", expanded=False):
                feature_df = pd.DataFrame(
                    {
                        "feature_index": range(1, len(feature_columns) + 1),
                        "feature": feature_columns,
                    }
                )
                st.dataframe(feature_df, use_container_width=True, hide_index=True)


def render_model_quality():
    with st.expander("📊 Model Quality", expanded=True):
        quality_df, quality_error = load_model_quality_summary()
        config, config_error = load_production_config()

        if quality_error:
            st.warning(quality_error)
            return

        if config_error:
            st.warning(config_error)

        if quality_df.empty:
            st.info("No model quality summary found.")
            return

        calibrated_accuracy = quality_metric_value(quality_df, "calibrated_accuracy")
        calibrated_roc_auc = quality_metric_value(quality_df, "calibrated_roc_auc")
        calibrated_log_loss = quality_metric_value(quality_df, "calibrated_log_loss")
        best_threshold = quality_metric_value(quality_df, "best_calibrated_threshold")
        train_fights = quality_metric_value(quality_df, "train_fights")
        test_fights = quality_metric_value(quality_df, "test_fights")

        cols = st.columns(6)
        cols[0].metric("Cal. Accuracy", percent_text(calibrated_accuracy))
        cols[1].metric("Cal. ROC-AUC", number_text(calibrated_roc_auc))
        cols[2].metric("Cal. Log Loss", number_text(calibrated_log_loss))
        cols[3].metric("Best Threshold", number_text(best_threshold, decimals=2))
        cols[4].metric("Train Fights", integer_text(train_fights))
        cols[5].metric("Test Fights", integer_text(test_fights))

        comparison_rows = [
            {
                "metric": "Accuracy",
                "raw": quality_metric_value(quality_df, "raw_accuracy"),
                "calibrated": quality_metric_value(quality_df, "calibrated_accuracy"),
            },
            {
                "metric": "Log Loss",
                "raw": quality_metric_value(quality_df, "raw_log_loss"),
                "calibrated": quality_metric_value(quality_df, "calibrated_log_loss"),
            },
            {
                "metric": "ROC-AUC",
                "raw": quality_metric_value(quality_df, "raw_roc_auc"),
                "calibrated": quality_metric_value(quality_df, "calibrated_roc_auc"),
            },
        ]

        st.subheader("Raw vs Calibrated")
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

        red_bias_rows = [
            {
                "metric": "Actual Red Win Rate",
                "value": quality_metric_value(quality_df, "actual_red_win_rate_percent"),
            },
            {
                "metric": "Raw Red Pick Rate",
                "value": quality_metric_value(quality_df, "raw_red_pick_rate_percent"),
            },
            {
                "metric": "Calibrated Red Pick Rate",
                "value": quality_metric_value(quality_df, "calibrated_red_pick_rate_percent"),
            },
        ]

        st.subheader("Red-Side Bias / Pick Rate")
        st.dataframe(
            pd.DataFrame(red_bias_rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Full Quality Summary", expanded=False):
            st.dataframe(quality_df, use_container_width=True, hide_index=True)


def render_feature_importance():
    with st.expander("🧬 Feature Importance", expanded=True):
        shap_df, shap_error = load_shap_importance()

        if shap_error:
            st.warning(shap_error)
            return

        if shap_df.empty:
            st.info("No SHAP importance artifact found.")
            return

        top_n = st.slider("Top features to display", min_value=5, max_value=50, value=20, step=5)
        top_features = shap_df.head(top_n).copy()

        st.subheader(f"Top {top_n} Features by Mean Absolute SHAP")
        chart_df = top_features.set_index("feature")[["mean_abs_shap"]].sort_values(
            "mean_abs_shap",
            ascending=True,
        )
        st.bar_chart(chart_df)

        st.subheader("Feature Importance Table")
        st.dataframe(shap_df, use_container_width=True, hide_index=True)


def render_live_prediction_audit():
    with st.expander("🧾 Live Prediction Audit", expanded=True):
        artifact_status = get_live_audit_artifact_status()
        predictions_df, predictions_error = load_model_predictions()
        feature_audit_df, feature_error = load_live_feature_audit()
        match_audit_df, match_error = load_live_match_audit()

        st.subheader("Audit Artifact Status")
        st.dataframe(
            artifact_status[["artifact", "path", "exists", "size", "status"]],
            use_container_width=True,
            hide_index=True,
        )

        for error in [predictions_error, feature_error, match_error]:
            if error:
                st.warning(error)

        if predictions_df.empty:
            st.info("No live model prediction artifact found yet.")
            return

        latest_run_id = get_latest_value(predictions_df, "prediction_run_id")
        latest_timestamp = get_latest_value(predictions_df, "prediction_timestamp")
        model_version = get_latest_value(predictions_df, "model_version")

        passes_model_quality = safe_sum_bool(predictions_df, "passes_model_data_quality")
        feature_validation_passes = safe_sum_bool(feature_audit_df, "passes_feature_validation")
        feature_validation_failures = (
            len(feature_audit_df) - feature_validation_passes
            if not feature_audit_df.empty and "passes_feature_validation" in feature_audit_df.columns
            else None
        )

        cols = st.columns(6)
        cols[0].metric("Prediction Rows", len(predictions_df))
        cols[1].metric("Run ID", latest_run_id or "—")
        cols[2].metric("Model Version", model_version or "—")
        cols[3].metric("Data Quality Pass", integer_text(passes_model_quality))
        cols[4].metric("Feature Pass", integer_text(feature_validation_passes))
        cols[5].metric("Feature Fail", integer_text(feature_validation_failures))

        st.caption(f"Latest prediction timestamp: {latest_timestamp or '—'}")

        if not match_audit_df.empty:
            st.subheader("Match Quality")
            match_rows = []

            for column_name in ["red_feature_match", "blue_feature_match"]:
                if column_name not in match_audit_df.columns:
                    continue

                counts = match_audit_df[column_name].value_counts(dropna=False)

                for match_type, count in counts.items():
                    match_rows.append(
                        {
                            "side": column_name.replace("_feature_match", ""),
                            "match_type": match_type,
                            "count": int(count),
                        }
                    )

            if match_rows:
                st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)

        if not feature_audit_df.empty:
            st.subheader("Feature Audit Summary")
            summary_rows = build_feature_audit_summary(feature_audit_df)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            if "passes_feature_validation" in feature_audit_df.columns:
                problem_fights = feature_audit_df[
                    feature_audit_df["passes_feature_validation"] == False
                ].copy()

                st.subheader("Problem Fights")

                if problem_fights.empty:
                    st.success("All fights passed feature validation.")
                else:
                    display_cols = [
                        "event_name",
                        "red_fighter",
                        "blue_fighter",
                        "red_feature_match",
                        "blue_feature_match",
                        "missing_feature_count",
                        "nonzero_feature_count",
                        "zero_feature_pct",
                        "passes_feature_validation",
                    ]
                    display_cols = [col for col in display_cols if col in problem_fights.columns]
                    st.dataframe(problem_fights[display_cols], use_container_width=True, hide_index=True)

        with st.expander("Model Prediction Rows", expanded=False):
            st.dataframe(predictions_df, use_container_width=True, hide_index=True)


def get_latest_value(df, column_name):
    if df.empty or column_name not in df.columns:
        return None

    values = df[column_name].dropna()

    if values.empty:
        return None

    return values.iloc[-1]


def safe_sum_bool(df, column_name):
    if df.empty or column_name not in df.columns:
        return None

    return int(df[column_name].fillna(False).astype(bool).sum())


def safe_mean(df, column_name):
    if df.empty or column_name not in df.columns:
        return None

    return pd.to_numeric(df[column_name], errors="coerce").mean()


def safe_max(df, column_name):
    if df.empty or column_name not in df.columns:
        return None

    return pd.to_numeric(df[column_name], errors="coerce").max()


def build_feature_audit_summary(feature_audit_df):
    return [
        {
            "metric": "Rows",
            "value": len(feature_audit_df),
        },
        {
            "metric": "Expected Feature Count",
            "value": safe_max(feature_audit_df, "feature_count_expected"),
        },
        {
            "metric": "Actual Feature Count",
            "value": safe_max(feature_audit_df, "feature_count_actual"),
        },
        {
            "metric": "Max Missing Feature Count",
            "value": safe_max(feature_audit_df, "missing_feature_count"),
        },
        {
            "metric": "Avg Nonzero Feature Count",
            "value": safe_mean(feature_audit_df, "nonzero_feature_count"),
        },
        {
            "metric": "Avg Zero Feature %",
            "value": safe_mean(feature_audit_df, "zero_feature_pct"),
        },
        {
            "metric": "Max Zero Feature %",
            "value": safe_max(feature_audit_df, "zero_feature_pct"),
        },
        {
            "metric": "Feature Validation Passes",
            "value": safe_sum_bool(feature_audit_df, "passes_feature_validation"),
        },
    ]
