from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import yaml

from pipeline.modeling.model_loader import load_model_bundle
from pipeline.modeling.model_config import load_model_config
from pipeline.modeling.run_live_model_forensics import try_shap, unwrap_estimator
from tabs.model_lab_sections import backtest as base


DOG_EDGE_THRESHOLD = 0.10
MAX_SHAP_ROWS_PER_GROUP = 75
TOP_N_FEATURES = 30
DOG_ARCHETYPE_FEATURES = [
    "avg_opponent_elo_diff",
    "ewm_avg_opponent_elo_diff",
    "best_win_elo_diff",
    "ewm_best_win_elo_diff",
    "win_pct_diff",
    "ewm_win_pct_diff",
    "striking_edge",
    "grappling_edge",
    "submission_mismatch_diff",
    "wrestling_mismatch_diff",
    "striking_edge_x_avg_opp_elo",
    "striking_edge_x_ewm_avg_opp_elo",
    "grappling_edge_x_avg_opp_elo",
    "grappling_edge_x_ewm_avg_opp_elo",
    "wrestling_mismatch_x_avg_opp_elo",
    "submission_mismatch_x_avg_opp_elo",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_paths(config_payload: dict[str, Any], summary: dict[str, Any]) -> tuple[Path | None, Path | None]:
    model_config = config_payload.get("model_config_path") or summary.get("model_config_path")
    feature_view = config_payload.get("feature_view_path") or summary.get("feature_view_path")
    return (Path(model_config) if model_config else None, Path(feature_view) if feature_view else None)


def _load_inputs(artifact_dir: Path, config_payload: dict[str, Any], summary: dict[str, Any]):
    bets = base._load_bet_level_table(artifact_dir)
    model_config_path, feature_view_path = _resolve_paths(config_payload, summary)
    if bets.empty:
        return None, None, None, None, "No bet-level table found."
    if model_config_path is None or not model_config_path.exists():
        return bets, None, None, None, f"Model config not found: {model_config_path}"
    if feature_view_path is None or not feature_view_path.exists():
        return bets, model_config_path, None, None, f"Feature view not found: {feature_view_path}"

    try:
        model_config = load_model_config(model_config_path, require_prediction=True)
        bundle = load_model_bundle(model_config, prefer_calibrated=True)
    except Exception as exc:
        return bets, model_config_path, feature_view_path, None, f"Could not load model bundle: {exc}"

    return bets, model_config_path, feature_view_path, bundle, None


def _prepare_dog_groups(bets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = bets.copy()
    for column in ["american_odds", "edge"]:
        if column in temp.columns:
            temp[column] = pd.to_numeric(temp[column], errors="coerce")
    if "won" in temp.columns:
        temp["won_bool"] = temp["won"].astype("boolean")
    else:
        temp["won_bool"] = pd.NA

    required = {"american_odds", "edge", "won_bool", "fight_id"}
    if not required.issubset(temp.columns):
        return pd.DataFrame(), pd.DataFrame()

    dogs = temp[(temp["american_odds"] > 0) & (temp["edge"] >= DOG_EDGE_THRESHOLD)].copy()
    losing = dogs[dogs["won_bool"] == False].copy()
    winning = dogs[dogs["won_bool"] == True].copy()
    return losing, winning


def _match_feature_rows(feature_view: pd.DataFrame, bets: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if feature_view.empty or bets.empty or "fight_id" not in feature_view.columns or "fight_id" not in bets.columns:
        return pd.DataFrame(columns=feature_columns)

    fight_ids = bets["fight_id"].astype(str).dropna().unique().tolist()
    rows = feature_view[feature_view["fight_id"].astype(str).isin(fight_ids)].copy()
    missing = [column for column in feature_columns if column not in rows.columns]
    if missing:
        st.warning(f"Feature view is missing {len(missing)} model features. First missing: {missing[:10]}")
        return pd.DataFrame(columns=feature_columns)

    if "date" in rows.columns:
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
        rows = rows.sort_values(["fight_id", "date"]).drop_duplicates("fight_id", keep="last")
    else:
        rows = rows.drop_duplicates("fight_id", keep="last")

    return rows[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _aggregate_shap(estimator: Any, x: pd.DataFrame, label: str) -> pd.DataFrame:
    if x.empty:
        return pd.DataFrame(columns=["feature", f"{label}_mean_abs_shap", f"{label}_mean_shap"])

    sample = x.head(MAX_SHAP_ROWS_PER_GROUP).copy()
    frames = []
    progress = st.progress(0, text=f"Running SHAP for {label}...")
    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        shap_df = try_shap(estimator, row.to_frame().T)
        if not shap_df.empty and "abs_shap_value" in shap_df.columns:
            frames.append(shap_df[["feature", "shap_value", "abs_shap_value"]].copy())
        progress.progress(idx / len(sample), text=f"Running SHAP for {label}: {idx}/{len(sample)}")
    progress.empty()

    if not frames:
        return pd.DataFrame(columns=["feature", f"{label}_mean_abs_shap", f"{label}_mean_shap"])

    all_values = pd.concat(frames, ignore_index=True)
    grouped = all_values.groupby("feature", dropna=False).agg(
        mean_abs_shap=("abs_shap_value", "mean"),
        mean_shap=("shap_value", "mean"),
    ).reset_index()
    return grouped.rename(
        columns={
            "mean_abs_shap": f"{label}_mean_abs_shap",
            "mean_shap": f"{label}_mean_shap",
        }
    )


def _feature_value_summary(x_losing: pd.DataFrame, x_winning: pd.DataFrame) -> pd.DataFrame:
    if x_losing.empty and x_winning.empty:
        return pd.DataFrame()
    rows = []
    features = list(dict.fromkeys(list(x_losing.columns) + list(x_winning.columns)))
    for feature in features:
        losing_mean = pd.to_numeric(x_losing.get(feature), errors="coerce").mean() if feature in x_losing else pd.NA
        winning_mean = pd.to_numeric(x_winning.get(feature), errors="coerce").mean() if feature in x_winning else pd.NA
        rows.append({
            "feature": feature,
            "losing_dog_avg": losing_mean,
            "winning_dog_avg": winning_mean,
            "avg_delta_losing_minus_winning": losing_mean - winning_mean if pd.notna(losing_mean) and pd.notna(winning_mean) else pd.NA,
        })
    return pd.DataFrame(rows)


def _feature_separation_summary(values: pd.DataFrame) -> pd.DataFrame:
    if values.empty or "avg_delta_losing_minus_winning" not in values.columns:
        return pd.DataFrame()
    out = values.copy()
    out["abs_avg_delta"] = pd.to_numeric(out["avg_delta_losing_minus_winning"], errors="coerce").abs()
    out = out.sort_values("abs_avg_delta", ascending=False).reset_index(drop=True)
    return out[["feature", "losing_dog_avg", "winning_dog_avg", "avg_delta_losing_minus_winning", "abs_avg_delta"]]


def _dog_archetype_summary(values: pd.DataFrame) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame()
    out = values[values["feature"].isin(DOG_ARCHETYPE_FEATURES)].copy()
    if out.empty:
        return pd.DataFrame()
    out["abs_avg_delta"] = pd.to_numeric(out["avg_delta_losing_minus_winning"], errors="coerce").abs()
    return out[["feature", "losing_dog_avg", "winning_dog_avg", "avg_delta_losing_minus_winning", "abs_avg_delta"]].sort_values(
        "abs_avg_delta",
        ascending=False,
    )


def _format_shap_table(df: pd.DataFrame):
    if df.empty:
        return df
    formatters = {column: "{:.5f}" for column in df.columns if column != "feature"}
    return df.style.format(formatters)


def render_dog_audit(artifact_dir: Path, summary: dict[str, Any], config_payload: dict[str, Any]) -> None:
    st.markdown("##### Losing vs Winning Underdog SHAP Audit")
    st.caption("Filters underdogs with edge ≥ 10%, then compares losing dogs against winning dogs.")

    bets, model_config_path, feature_view_path, bundle, error = _load_inputs(artifact_dir, config_payload, summary)
    if error:
        st.warning(error)
        return
    assert bets is not None and feature_view_path is not None and bundle is not None

    losing_dogs, winning_dogs = _prepare_dog_groups(bets)
    c1, c2, c3 = st.columns(3)
    c1.metric("Losing high-edge dogs", f"{len(losing_dogs):,}")
    c2.metric("Winning high-edge dogs", f"{len(winning_dogs):,}")
    c3.metric("SHAP sample cap/group", f"{MAX_SHAP_ROWS_PER_GROUP:,}")

    if losing_dogs.empty or winning_dogs.empty:
        st.info("Dog Audit needs both losing and winning high-edge underdogs in the latest backtest.")
        return

    try:
        feature_view = pd.read_parquet(feature_view_path)
    except Exception as exc:
        st.warning(f"Could not read feature view: {exc}")
        return

    x_losing = _match_feature_rows(feature_view, losing_dogs, bundle.feature_columns)
    x_winning = _match_feature_rows(feature_view, winning_dogs, bundle.feature_columns)
    if x_losing.empty or x_winning.empty:
        st.warning("Could not reconstruct feature rows for both dog groups.")
        return

    st.caption(f"Model config: `{model_config_path}` · Feature view: `{feature_view_path}` · Model artifact: `{bundle.model_artifact_path}`")

    estimator = unwrap_estimator(bundle.model)
    with st.spinner("Computing grouped SHAP values..."):
        losing_shap = _aggregate_shap(estimator, x_losing, "losing_dogs")
        winning_shap = _aggregate_shap(estimator, x_winning, "winning_dogs")

    if losing_shap.empty or winning_shap.empty:
        st.warning("SHAP values could not be generated. Check that `shap` is installed and the model supports TreeExplainer.")
        return

    merged = losing_shap.merge(winning_shap, on="feature", how="outer").fillna(0.0)
    merged["abs_shap_delta_losing_minus_winning"] = merged["losing_dogs_mean_abs_shap"] - merged["winning_dogs_mean_abs_shap"]
    merged = merged.sort_values("abs_shap_delta_losing_minus_winning", ascending=False).reset_index(drop=True)

    st.markdown("###### Features More Important in Losing Dogs")
    st.dataframe(_format_shap_table(merged.head(TOP_N_FEATURES)), use_container_width=True, hide_index=True)

    values = _feature_value_summary(x_losing, x_winning)
    if not values.empty:
        values = values.merge(merged[["feature", "abs_shap_delta_losing_minus_winning"]], on="feature", how="left")
        importance_sorted_values = values.sort_values("abs_shap_delta_losing_minus_winning", ascending=False).head(TOP_N_FEATURES)
        st.markdown("###### Feature Value Comparison")
        st.dataframe(_format_shap_table(importance_sorted_values), use_container_width=True, hide_index=True)

        archetype = _dog_archetype_summary(values)
        if not archetype.empty:
            st.markdown("###### Dog Archetype Summary")
            st.caption("Focused comparison of competition-quality, matchup, and new quality-adjusted features.")
            st.dataframe(_format_shap_table(archetype), use_container_width=True, hide_index=True)

        separation = _feature_separation_summary(values).head(TOP_N_FEATURES)
        st.markdown("###### Top Feature Separation")
        st.caption("Sorted by absolute average difference between losing high-edge dogs and winning high-edge dogs.")
        st.dataframe(_format_shap_table(separation), use_container_width=True, hide_index=True)

    with st.expander("Audit assumptions", expanded=False):
        st.write(
            "SHAP is computed against the red-side model probability. For blue-side dog bets, absolute SHAP is still useful for identifying influential features, but signed SHAP should be interpreted cautiously."
        )
