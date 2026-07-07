from __future__ import annotations

import base64
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yaml

from utils.github_actions import GITHUB_API_BASE, get_github_config, github_headers, trigger_workflow


MODEL_REGISTRY_PATH = Path("configs/models/model_registry.yaml")

FEATURE_VIEW_CONFIG_BY_SOURCE = {
    "moneyline_feature_view": "configs/feature_views/moneyline_base.yaml",
    "prop_goes_distance_feature_view": "configs/feature_views/prop_goes_distance.yaml",
}

WORKFLOWS = {
    "feature_view": "run-build-feature-view-v2.yml",
    "training": "run-train-model-v2.yml",
    "prediction": "run-prediction-v2.yml",
    "ensemble_prediction": "run-ensemble-prediction-v2.yml",
    "betting_outcomes": "run-betting-outcomes-v2.yml",
}

STATUS_OPTIONS = ["draft", "production", "archived"]
EDITABLE_STATUSES = {"draft"}
READ_ONLY_STATUSES = {"production", "archived"}


class ModelLabWorkflowError(RuntimeError):
    """Raised when Model Lab cannot resolve registry-driven workflow inputs."""


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_yaml_file(path_text: str) -> dict[str, Any]:
    """Load a local YAML file as a dictionary."""

    path = Path(path_text)
    if not path.exists():
        raise ModelLabWorkflowError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    if not isinstance(payload, dict):
        raise ModelLabWorkflowError(f"YAML file must contain a mapping: {path}")

    return payload


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    """Load the model registry used by V2 workflow selection."""

    return load_yaml_file(str(path))


def get_registered_model_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all registered model rows for the Model Lab selector."""

    rows: list[dict[str, Any]] = []
    for model_id, entry in (registry.get("models", {}) or {}).items():
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "model_id": str(model_id),
                "display_name": entry.get("display_name") or str(model_id),
                "model_family": entry.get("model_family", ""),
                "market_key": entry.get("market_key", "moneyline"),
                "algorithm": entry.get("algorithm", ""),
                "status": entry.get("status", "draft"),
                "config_path": entry.get("config_path", ""),
                "artifact_dir": entry.get("artifact_dir", ""),
                "dashboard_selectable": bool(entry.get("dashboard_selectable", False)),
            }
        )
    return rows


def _model_label(row: dict[str, Any]) -> str:
    status = row.get("status") or "unknown"
    family = row.get("model_family") or "unknown"
    marker = "Champion" if status == "production" else status.title()
    return f"{row['model_id']} — {row.get('display_name', row['model_id'])} ({family}, {marker})"


def _first_feature_source(model_config: dict[str, Any]) -> str | None:
    sources = model_config.get("feature_sources") or []
    if not sources:
        return None
    return str(sources[0])


def _feature_view_config_path(model_config: dict[str, Any]) -> Path | None:
    explicit_path = (
        model_config.get("feature_view_config_path")
        or (model_config.get("feature_view") or {}).get("config_path")
    )
    if explicit_path:
        path = Path(str(explicit_path))
        return path if path.exists() else None

    feature_source = _first_feature_source(model_config)
    mapped_path = FEATURE_VIEW_CONFIG_BY_SOURCE.get(feature_source or "")
    if not mapped_path:
        return None

    path = Path(mapped_path)
    return path if path.exists() else None


def _feature_view_output_path(
    *,
    model_config: dict[str, Any],
    feature_view_config_path: Path | None,
) -> Path | None:
    if feature_view_config_path is not None:
        feature_config = load_yaml_file(str(feature_view_config_path))
        output_path = (feature_config.get("output") or {}).get("feature_view_path")
        if output_path:
            return Path(str(output_path))

    model_data_path = (model_config.get("data") or {}).get("rolling_features_path")
    return Path(str(model_data_path)) if model_data_path else None


def resolve_model_workflow_context(
    *,
    registry: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Resolve all workflow inputs for the selected registered model."""

    models = registry.get("models", {}) or {}
    if model_id not in models:
        raise ModelLabWorkflowError(f"Model is not registered: {model_id}")

    entry = models[model_id]
    config_path = Path(str(entry.get("config_path") or ""))
    if not config_path.exists():
        raise ModelLabWorkflowError(f"Model config path does not exist: {config_path}")

    model_config = load_yaml_file(str(config_path))
    feature_view_config_path = _feature_view_config_path(model_config)
    feature_view_output_path = _feature_view_output_path(
        model_config=model_config,
        feature_view_config_path=feature_view_config_path,
    )

    model_family = str(entry.get("model_family") or model_config.get("model_family") or "")
    artifact_dir = str(
        entry.get("artifact_dir")
        or (model_config.get("artifacts") or {}).get("output_dir")
        or ""
    )

    return {
        "model_id": model_id,
        "display_name": entry.get("display_name") or model_id,
        "description": entry.get("description", ""),
        "status": entry.get("status", "draft"),
        "dashboard_selectable": bool(entry.get("dashboard_selectable", False)),
        "model_family": model_family,
        "market_key": entry.get("market_key") or model_config.get("market_key") or "moneyline",
        "algorithm": entry.get("algorithm") or model_config.get("algorithm") or "",
        "config_path": str(config_path),
        "artifact_dir": artifact_dir,
        "feature_source": _first_feature_source(model_config) or "",
        "feature_view_config_path": str(feature_view_config_path) if feature_view_config_path else "",
        "feature_view_output_path": str(feature_view_output_path) if feature_view_output_path else "",
        "config": model_config,
        "registry_entry": entry,
    }


@st.cache_data(show_spinner=False)
def _read_json_file(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _read_parquet_file(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# GitHub persistence helpers
# -----------------------------------------------------------------------------


def _github_file_url(owner: str, repo: str, path: str) -> str:
    return f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"


def _github_read_file(path: str) -> tuple[bool, str, str | None]:
    owner, repo, token, branch = get_github_config()
    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets.", None

    response = requests.get(
        _github_file_url(owner, repo, path),
        headers=github_headers(token),
        params={"ref": branch},
        timeout=20,
    )
    if response.status_code == 404:
        return True, "", None
    if response.status_code != 200:
        return False, f"GitHub API error {response.status_code}: {response.text}", None

    payload = response.json()
    encoded = payload.get("content", "")
    text = base64.b64decode(encoded).decode("utf-8") if encoded else ""
    return True, text, payload.get("sha")


def _github_write_file(path: str, content: str, message: str) -> tuple[bool, str]:
    owner, repo, token, branch = get_github_config()
    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets."

    ok, _, sha = _github_read_file(path)
    if not ok:
        return False, "Could not inspect existing GitHub file before save."

    body: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha

    response = requests.put(
        _github_file_url(owner, repo, path),
        headers=github_headers(token),
        json=body,
        timeout=20,
    )
    if response.status_code in {200, 201}:
        return True, f"Saved {path} to GitHub."
    return False, f"GitHub API error {response.status_code}: {response.text}"


def _yaml_dump(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


# -----------------------------------------------------------------------------
# Feature bundle resolution
# -----------------------------------------------------------------------------


BUNDLE_LABELS = {
    "core_state": "Core State",
    "ewm_state": "EWM State",
    "recent_form": "Recent Form",
    "finish_profile": "Finish Profile",
    "durability": "Durability",
    "fight_context": "Fight Context",
    "physical": "Reach & Physical",
    "experience": "Age & Experience",
    "engineered": "Engineered Matchup",
}


def _available_feature_columns(context: dict[str, Any]) -> list[str]:
    feature_view_path = context.get("feature_view_output_path") or (context["config"].get("data") or {}).get("rolling_features_path")
    df = _read_parquet_file(str(feature_view_path)) if feature_view_path else pd.DataFrame()
    if df.empty:
        return list((context["config"].get("features") or {}).get("feature_columns") or [])

    excluded = {
        "event_id", "event_name", "fight_id", "date", "fight_date", "method", "winner",
        "target", "target_goes_distance", "r_id", "b_id", "r_name", "b_name",
        "red_fighter", "blue_fighter", "division", "referee", "location",
    }
    return [column for column in df.columns if column not in excluded and not str(column).startswith(("r_pre_", "b_pre_", "r_ewm_", "b_ewm_", "r_recent_form_", "b_recent_form_"))]


def _bundle_for_feature(feature: str) -> str:
    name = str(feature)
    if name in {"title_fight", "total_rounds", "division"}:
        return "fight_context"
    if name.startswith("ewm_"):
        return "ewm_state"
    if name.startswith("recent_form_"):
        return "recent_form"
    if any(token in name for token in ["finish", "ko_rate", "sub_win", "decision_win", "decision_loss", "avg_fight_time"]):
        return "finish_profile"
    if any(token in name for token in ["kd_absorbed", "sapm", "str_def", "td_def", "finish_loss", "chin"]):
        return "durability"
    if any(token in name for token in ["height", "reach", "weight"]):
        return "physical"
    if any(token in name for token in ["age", "fights", "wins", "losses", "streak", "days_since"]):
        return "experience"
    if any(token in name for token in ["edge", "volatility", "mismatch", "combo", "ratio", "pressure", "aggression"]):
        return "engineered"
    return "core_state"


def _bundle_map(features: list[str]) -> dict[str, list[str]]:
    bundles: dict[str, list[str]] = {key: [] for key in BUNDLE_LABELS}
    for feature in features:
        bundles.setdefault(_bundle_for_feature(feature), []).append(feature)
    return {key: sorted(values) for key, values in bundles.items() if values}


def _infer_selected_bundles(config: dict[str, Any], features: list[str]) -> list[str]:
    feature_config = config.get("features") or {}
    explicit = feature_config.get("selected_bundles") or feature_config.get("bundles")
    if isinstance(explicit, list) and explicit:
        return [str(value) for value in explicit]
    current = set(feature_config.get("feature_columns") or [])
    bundles = _bundle_map(features)
    selected = []
    for bundle, cols in bundles.items():
        if cols and current.intersection(cols):
            selected.append(bundle)
    return selected or list(bundles.keys())


def _resolve_features_from_bundles(
    *,
    available_features: list[str],
    selected_bundles: list[str],
    include_overrides: list[str],
    exclude_overrides: list[str],
) -> list[str]:
    bundles = _bundle_map(available_features)
    selected: list[str] = []
    for bundle in selected_bundles:
        selected.extend(bundles.get(bundle, []))
    selected.extend(include_overrides)
    exclude = set(exclude_overrides)
    return list(dict.fromkeys([feature for feature in selected if feature and feature not in exclude]))


def _csv_to_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").replace("\n", ",").split(",") if item.strip()]


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _artifact_path(context: dict[str, Any], filename: str) -> str:
    return str(Path(context.get("artifact_dir") or "") / filename)


def _metrics_payload(context: dict[str, Any]) -> dict[str, Any]:
    return _read_json_file(_artifact_path(context, "metrics.json"))


def _model_card(context: dict[str, Any]) -> dict[str, Any]:
    path = Path(_artifact_path(context, "model_card.yaml"))
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _final_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics.get("final_metrics") or metrics.get("metrics") or {}


def _fmt_pct(value, decimals: int = 1) -> str:
    try:
        value = float(value)
    except Exception:
        return "—"
    if abs(value) <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


def _fmt_num(value, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def _metric_value(final: dict[str, Any], *names: str):
    for name in names:
        if name in final:
            return final[name]
    return None


# -----------------------------------------------------------------------------
# Styling and cards
# -----------------------------------------------------------------------------


def _inject_css() -> None:
    st.html(
        """
        <style>
        .mlab-hero { display:flex; justify-content:space-between; align-items:flex-start; margin:.1rem 0 .8rem; }
        .mlab-title { color:#f5f7fb; font-size:2rem; font-weight:950; letter-spacing:-.04em; text-transform:uppercase; }
        .mlab-subtitle { color:#dbe7f5; font-size:.95rem; margin-top:.25rem; }
        .mlab-grid { display:grid; gap:.65rem; }
        .mlab-kpis { grid-template-columns:repeat(8, minmax(0,1fr)); margin:.75rem 0; }
        .mlab-card { background:linear-gradient(180deg, rgba(17,31,49,.95), rgba(9,19,32,.98)); border:1px solid rgba(43,60,82,.95); border-radius:9px; box-shadow:0 20px 42px rgba(0,0,0,.24); }
        .mlab-kpi { min-height:86px; padding:.85rem .65rem; text-align:center; }
        .mlab-label { color:#dbe7f5; text-transform:uppercase; font-size:.67rem; font-weight:900; letter-spacing:.035em; }
        .mlab-value { color:#f5f7fb; font-size:1.45rem; font-weight:950; margin-top:.32rem; }
        .mlab-caption { color:#9fb0c4; font-size:.72rem; margin-top:.2rem; }
        .mlab-green { color:#31df63 !important; } .mlab-blue { color:#3b82f6 !important; } .mlab-purple { color:#a855f7 !important; } .mlab-red { color:#ff5555 !important; }
        .mlab-section { padding:.9rem 1rem 1rem; }
        .mlab-section-title { color:#f5f7fb; text-transform:uppercase; font-size:.8rem; font-weight:950; margin-bottom:.65rem; }
        .mlab-model-bar { display:flex; justify-content:space-between; gap:1rem; padding:.9rem 1rem; align-items:center; }
        .mlab-model-name { color:#f5f7fb; font-size:1.25rem; font-weight:950; }
        .mlab-pill { display:inline-block; padding:.2rem .48rem; border-radius:5px; font-size:.68rem; font-weight:900; text-transform:uppercase; }
        .mlab-prod { color:#d1fae5; background:rgba(34,197,94,.28); } .mlab-draft { color:#dbeafe; background:rgba(37,99,235,.3); } .mlab-archived { color:#e5e7eb; background:rgba(107,114,128,.38); }
        .mlab-two { display:grid; grid-template-columns:1.02fr 1.52fr; gap:.65rem; margin-top:.65rem; }
        .mlab-three { display:grid; grid-template-columns:1fr 1fr 1.15fr; gap:.65rem; margin-top:.65rem; }
        .mlab-actions { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:.65rem; }
        @media (max-width: 1200px) { .mlab-kpis { grid-template-columns:repeat(2,1fr); } .mlab-two, .mlab-three, .mlab-actions { grid-template-columns:1fr; } }
        </style>
        """
    )


def _kpi(label: str, value: str, caption: str = "", color_class: str = "") -> str:
    return (
        "<div class='mlab-card mlab-kpi'>"
        f"<div class='mlab-label'>{label}</div>"
        f"<div class='mlab-value {color_class}'>{value}</div>"
        f"<div class='mlab-caption'>{caption}</div>"
        "</div>"
    )


def _render_header() -> None:
    now = datetime.now(timezone.utc).strftime("%b %-d, %Y %I:%M %p UTC")
    st.html(
        "<div class='mlab-hero'>"
        "<div><div class='mlab-title'>Model Lab</div>"
        "<div class='mlab-subtitle'>Build, tune, compare, and promote predictive models.</div></div>"
        f"<div class='mlab-caption'>Last Loaded: {now}</div>"
        "</div>"
    )


def _render_kpis(context: dict[str, Any]) -> None:
    metrics = _metrics_payload(context)
    final = _final_metrics(metrics)
    card = _model_card(context)
    feature_count = len((context["config"].get("features") or {}).get("feature_columns") or [])
    status = context.get("status", "draft")
    cards = [
        _kpi("Accuracy", _fmt_pct(_metric_value(final, "accuracy")), "Test split", "mlab-green"),
        _kpi("ROC AUC", _fmt_num(_metric_value(final, "roc_auc")), "Higher is better", "mlab-blue"),
        _kpi("Log Loss", _fmt_num(_metric_value(final, "log_loss")), "Lower is better", "mlab-blue"),
        _kpi("Brier Score", _fmt_num(_metric_value(final, "brier_score")), "Lower is better", "mlab-blue"),
        _kpi("Best Threshold", _fmt_num(metrics.get("best_threshold") or card.get("best_threshold"), 2), "From sweep", ""),
        _kpi("Feature Count", str(feature_count), "Configured inputs", ""),
        _kpi("Model Status", status.title(), "Registry status", "mlab-green" if status == "production" else "mlab-blue" if status == "draft" else ""),
        _kpi("Trained", "Yes" if metrics else "No", "Metrics artifact", "mlab-green" if metrics else "mlab-red"),
    ]
    st.html("<div class='mlab-grid mlab-kpis'>" + "".join(cards) + "</div>")


# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------


def _render_model_bar(context: dict[str, Any], registry: dict[str, Any]) -> None:
    status = str(context.get("status") or "draft").lower()
    pill_class = "mlab-prod" if status == "production" else "mlab-draft" if status == "draft" else "mlab-archived"
    active = registry.get("active_models", {}).get(context["model_family"], {}).get("primary") == context["model_id"]
    active_text = " · Active Primary" if active else ""
    st.html(
        "<div class='mlab-card mlab-model-bar'>"
        "<div>"
        f"<div class='mlab-model-name'>{context['model_id']} <span class='mlab-pill {pill_class}'>{status}</span></div>"
        f"<div class='mlab-caption'>Family: {context['model_family']} · Market: {context['market_key']} · Artifact: {context['artifact_dir']}{active_text}</div>"
        "</div>"
        "<div class='mlab-caption'>Production models are immutable. Clone to draft before tuning.</div>"
        "</div>"
    )


def _render_registry_table(rows: list[dict[str, Any]]) -> None:
    with st.expander("Model Registry", expanded=False):
        display = pd.DataFrame(rows)
        if not display.empty:
            st.dataframe(
                display[["model_id", "display_name", "model_family", "market_key", "status", "artifact_dir", "config_path"]],
                use_container_width=True,
                hide_index=True,
            )


def _render_lifecycle_controls(context: dict[str, Any], registry: dict[str, Any]) -> None:
    status = str(context.get("status") or "draft").lower()
    st.markdown("#### Lifecycle")
    if status == "production":
        st.info("Production configs are read-only. Clone this model to draft before editing.")
        default_clone = f"{context['model_id']}_draft"
        clone_id = st.text_input("Draft model ID", value=default_clone, key="mlab_clone_id")
        if st.button("Clone to Draft", use_container_width=True, key="mlab_clone_button"):
            ok, msg = _clone_model_to_draft(context=context, registry=registry, clone_id=clone_id)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()
    elif status == "draft":
        st.success("Draft model is editable.")
        new_status = st.selectbox("Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(status), key="mlab_status_select")
        set_active = st.toggle("Make active primary for this family on save", value=False, key="mlab_set_active_primary")
        if st.button("Update Registry Status", use_container_width=True, key="mlab_update_status"):
            updated = deepcopy(registry)
            updated["models"][context["model_id"]]["status"] = new_status
            if set_active and new_status == "production":
                updated.setdefault("active_models", {}).setdefault(context["model_family"], {})["primary"] = context["model_id"]
            ok, msg = _save_registry(updated)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()
    else:
        st.warning("Archived models are read-only. Clone from another model if you want a new experiment.")


def _render_config_editor(context: dict[str, Any], registry: dict[str, Any]) -> None:
    config = deepcopy(context["config"])
    editable = context["status"] in EDITABLE_STATUSES
    feature_config = config.setdefault("features", {})
    split = config.setdefault("split", {})
    calibration = config.setdefault("calibration", {})
    params = config.setdefault("params", {})
    probability = config.setdefault("prediction", {}).setdefault("probability", {})

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Configuration</div>")
    if not editable:
        st.caption("Read-only because this model is not draft.")

    display_name = st.text_input("Display Name", value=str(context.get("display_name") or context["model_id"]), disabled=not editable, key="mlab_display_name")
    description = st.text_area("Description", value=str(context.get("description") or ""), disabled=not editable, height=72, key="mlab_description")

    c1, c2 = st.columns(2)
    with c1:
        train_end = st.text_input("Train End Date", value=str(split.get("train_end_date", "2022-12-31")), disabled=not editable, key="mlab_train_end")
        cal_end = st.text_input("Calibration End Date", value=str(split.get("calibration_end_date", "2023-12-31")), disabled=not editable, key="mlab_cal_end")
        calibration_enabled = st.toggle("Calibration Enabled", value=bool(calibration.get("enabled", True)), disabled=not editable, key="mlab_cal_enabled")
        calibration_method = st.selectbox("Calibration Method", ["isotonic", "sigmoid", "none"], index=["isotonic", "sigmoid", "none"].index(str(calibration.get("method", "isotonic"))) if str(calibration.get("method", "isotonic")) in ["isotonic", "sigmoid", "none"] else 0, disabled=not editable, key="mlab_cal_method")
    with c2:
        clip_low = st.number_input("Probability Clip Low", value=float(probability.get("clip_low", 0.02)), step=0.01, min_value=0.0, max_value=0.49, disabled=not editable, key="mlab_clip_low")
        clip_high = st.number_input("Probability Clip High", value=float(probability.get("clip_high", 0.98)), step=0.01, min_value=0.51, max_value=1.0, disabled=not editable, key="mlab_clip_high")
        expected_count = st.number_input("Expected Feature Count", value=int(feature_config.get("expected_feature_count", len(feature_config.get("feature_columns") or []))), step=1, min_value=1, disabled=not editable, key="mlab_expected_count")
        dashboard_selectable = st.toggle("Dashboard Selectable", value=bool(context.get("dashboard_selectable", False)), disabled=not editable, key="mlab_dashboard_selectable")

    st.markdown("##### XGBoost Parameters")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        n_estimators = st.number_input("N Estimators", value=int(params.get("n_estimators", 500)), step=50, min_value=50, disabled=not editable, key="mlab_n_estimators")
    with p2:
        max_depth = st.number_input("Max Depth", value=int(params.get("max_depth", 4)), step=1, min_value=1, max_value=12, disabled=not editable, key="mlab_max_depth")
    with p3:
        learning_rate = st.number_input("Learning Rate", value=float(params.get("learning_rate", 0.03)), step=0.01, min_value=0.001, max_value=1.0, disabled=not editable, key="mlab_learning_rate")
    with p4:
        subsample = st.number_input("Subsample", value=float(params.get("subsample", 0.8)), step=0.05, min_value=0.1, max_value=1.0, disabled=not editable, key="mlab_subsample")
    with p5:
        colsample = st.number_input("Colsample", value=float(params.get("colsample_bytree", 0.8)), step=0.05, min_value=0.1, max_value=1.0, disabled=not editable, key="mlab_colsample")

    st.html("</div></div>")

    return {
        "display_name": display_name,
        "description": description,
        "dashboard_selectable": dashboard_selectable,
        "train_end_date": train_end,
        "calibration_end_date": cal_end,
        "calibration_enabled": calibration_enabled,
        "calibration_method": calibration_method,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "expected_feature_count": int(expected_count),
        "params": {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
            "random_state": int(params.get("random_state", 42)),
            "eval_metric": params.get("eval_metric", "logloss"),
        },
    }


def _render_feature_bundle_editor(context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    editable = context["status"] in EDITABLE_STATUSES
    current_features = list((config.get("features") or {}).get("feature_columns") or [])
    available_features = _available_feature_columns(context)
    bundle_map = _bundle_map(available_features)
    default_bundles = _infer_selected_bundles(config, available_features)

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Feature Bundles</div>")
    selected_bundles = st.multiselect(
        "Selected Bundles",
        options=list(bundle_map.keys()),
        default=[bundle for bundle in default_bundles if bundle in bundle_map],
        format_func=lambda key: f"{BUNDLE_LABELS.get(key, key)} ({len(bundle_map.get(key, []))})",
        disabled=not editable,
        key="mlab_selected_bundles",
    )
    include_text = st.text_area("Include Feature Overrides", value="", help="Comma or newline separated feature names.", disabled=not editable, height=68, key="mlab_include_features")
    exclude_text = st.text_area("Exclude Feature Overrides", value="", help="Comma or newline separated feature names.", disabled=not editable, height=68, key="mlab_exclude_features")

    resolved = _resolve_features_from_bundles(
        available_features=available_features,
        selected_bundles=selected_bundles,
        include_overrides=_csv_to_list(include_text),
        exclude_overrides=_csv_to_list(exclude_text),
    )
    if not resolved and current_features:
        resolved = current_features

    st.caption(f"Resolved features: {len(resolved):,} · Available candidate columns: {len(available_features):,}")
    st.dataframe(pd.DataFrame({"feature": resolved[:200]}), use_container_width=True, hide_index=True, height=240)
    if len(resolved) > 200:
        st.caption(f"Showing first 200 of {len(resolved):,} resolved features.")
    st.html("</div></div>")

    return {
        "selected_bundles": selected_bundles,
        "include_features": _csv_to_list(include_text),
        "exclude_features": _csv_to_list(exclude_text),
        "resolved_features": resolved,
    }


def _render_performance(context: dict[str, Any]) -> None:
    metrics = _metrics_payload(context)
    final = _final_metrics(metrics)
    threshold_sweep = _read_parquet_file(_artifact_path(context, "threshold_sweep.parquet"))
    confidence = _read_parquet_file(_artifact_path(context, "confidence_buckets.parquet"))

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Performance Overview</div>")
    tabs = st.tabs(["Overview", "Calibration", "Thresholds", "Feature Importance"])
    with tabs[0]:
        if final:
            st.dataframe(pd.DataFrame([final]).T.rename(columns={0: "value"}), use_container_width=True)
        else:
            st.info("No metrics.json artifact found yet. Train this model to populate performance metrics.")
    with tabs[1]:
        if confidence.empty:
            st.info("No confidence_buckets.parquet artifact found yet.")
        else:
            st.dataframe(confidence, use_container_width=True, hide_index=True)
            numeric_cols = [col for col in confidence.columns if pd.api.types.is_numeric_dtype(confidence[col])]
            if numeric_cols:
                st.bar_chart(confidence[numeric_cols])
    with tabs[2]:
        if threshold_sweep.empty:
            st.info("No threshold_sweep.parquet artifact found yet.")
        else:
            st.dataframe(threshold_sweep, use_container_width=True, hide_index=True)
            x_col = "threshold" if "threshold" in threshold_sweep.columns else threshold_sweep.columns[0]
            y_candidates = [col for col in ["accuracy", "roc_auc", "f1", "log_loss", "brier_score"] if col in threshold_sweep.columns]
            if y_candidates:
                st.line_chart(threshold_sweep.set_index(x_col)[y_candidates])
    with tabs[3]:
        st.info("Feature importance display will read SHAP/gain artifacts when the training runner emits them for V2 models.")
    st.html("</div></div>")


def _comparison_rows(context: dict[str, Any], challenger: dict[str, Any] | None) -> pd.DataFrame:
    champion_metrics = _final_metrics(_metrics_payload(context))
    challenger_metrics = _final_metrics(_metrics_payload(challenger)) if challenger else {}
    rows = []
    for label, key, kind in [
        ("Accuracy", "accuracy", "pct"),
        ("ROC AUC", "roc_auc", "num"),
        ("Log Loss", "log_loss", "num"),
        ("Brier Score", "brier_score", "num"),
    ]:
        a = champion_metrics.get(key)
        b = challenger_metrics.get(key)
        delta = None
        try:
            delta = float(b) - float(a)
        except Exception:
            pass
        rows.append({"Metric": label, "Selected": a, "Comparison": b, "Delta": delta})
    return pd.DataFrame(rows)


def _render_comparison(context: dict[str, Any], registry: dict[str, Any], selected_model_id: str) -> None:
    model_ids = [model_id for model_id in (registry.get("models") or {}) if model_id != selected_model_id]
    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Model Comparison</div>")
    if not model_ids:
        st.info("No other models are registered for comparison yet.")
        st.html("</div></div>")
        return
    comparison_id = st.selectbox("Compare Against", model_ids, key="mlab_compare_model")
    challenger = None
    try:
        challenger = resolve_model_workflow_context(registry=registry, model_id=comparison_id)
    except Exception as exc:
        st.warning(f"Could not load comparison model: {exc}")
    table = _comparison_rows(context, challenger)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.html("</div></div>")


def _dispatch_button(label: str, workflow_file: str, inputs: dict[str, str], disabled: bool, key: str) -> None:
    if st.button(label, disabled=disabled, use_container_width=True, key=key):
        ok, message = trigger_workflow(workflow_file, inputs=inputs)
        st.success(message) if ok else st.error(message)


def _render_actions(context: dict[str, Any]) -> None:
    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Actions & Workflows</div>")
    feature_inputs = {
        "config_path": context.get("feature_view_config_path", ""),
        "output_path": context.get("feature_view_output_path", ""),
    }
    training_inputs = {
        "config_path": context["config_path"],
        "artifact_dir": context["artifact_dir"],
    }
    prediction_inputs = {
        "model_family": context["model_family"],
        "model_id": context["model_id"],
    }
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _dispatch_button("Build Feature View", WORKFLOWS["feature_view"], feature_inputs, not bool(feature_inputs["config_path"] and feature_inputs["output_path"]), f"mlab_build_{context['model_id']}")
    with c2:
        _dispatch_button("Train Model", WORKFLOWS["training"], training_inputs, not bool(training_inputs["config_path"] and training_inputs["artifact_dir"]), f"mlab_train_{context['model_id']}")
    with c3:
        _dispatch_button("Run Predictions", WORKFLOWS["prediction"], prediction_inputs, not bool(prediction_inputs["model_family"] and prediction_inputs["model_id"]), f"mlab_predict_{context['model_id']}")
    with c4:
        model_mode = st.selectbox("Betting Mode", ["production", "all", "single"], key="mlab_betting_mode")
        _dispatch_button("Run Outcomes", WORKFLOWS["betting_outcomes"], {"model_mode": model_mode}, False, f"mlab_outcomes_{context['model_id']}")
    st.html("</div></div>")


# -----------------------------------------------------------------------------
# Save / clone operations
# -----------------------------------------------------------------------------


def _apply_config_updates(
    context: dict[str, Any],
    form_values: dict[str, Any],
    feature_values: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(context["config"])
    config["model_id"] = context["model_id"]
    config["model_family"] = context["model_family"]
    config["market_key"] = context["market_key"]
    config["algorithm"] = context["algorithm"] or config.get("algorithm", "xgboost")
    config.setdefault("split", {})["train_end_date"] = str(form_values["train_end_date"])
    config.setdefault("split", {})["calibration_end_date"] = str(form_values["calibration_end_date"])
    config.setdefault("calibration", {})["enabled"] = bool(form_values["calibration_enabled"])
    config.setdefault("calibration", {})["method"] = str(form_values["calibration_method"])
    config.setdefault("prediction", {}).setdefault("probability", {})["clip_low"] = float(form_values["clip_low"])
    config.setdefault("prediction", {}).setdefault("probability", {})["clip_high"] = float(form_values["clip_high"])
    config["params"] = form_values["params"]

    features = config.setdefault("features", {})
    features["selection_mode"] = "explicit"
    features["allow_unsafe_features"] = bool(features.get("allow_unsafe_features", False))
    features["selected_bundles"] = feature_values["selected_bundles"]
    features["include_features"] = feature_values["include_features"]
    features["exclude_features"] = feature_values["exclude_features"]
    features["feature_columns"] = feature_values["resolved_features"]
    features["expected_feature_count"] = len(feature_values["resolved_features"])
    return config


def _save_registry(registry: dict[str, Any]) -> tuple[bool, str]:
    return _github_write_file(str(MODEL_REGISTRY_PATH), _yaml_dump(registry), "Update model registry from Model Lab")


def _save_draft_model(
    *,
    context: dict[str, Any],
    registry: dict[str, Any],
    form_values: dict[str, Any],
    feature_values: dict[str, Any],
) -> tuple[bool, str]:
    if context["status"] not in EDITABLE_STATUSES:
        return False, "Only draft models can be edited. Clone production models to draft first."
    resolved_features = feature_values.get("resolved_features") or []
    if not resolved_features:
        return False, "Feature bundle selection resolved to zero features."

    updated_config = _apply_config_updates(context, form_values, feature_values)
    updated_registry = deepcopy(registry)
    entry = updated_registry.setdefault("models", {}).setdefault(context["model_id"], {})
    entry["display_name"] = form_values["display_name"]
    entry["description"] = form_values["description"]
    entry["model_family"] = context["model_family"]
    entry["market_key"] = context["market_key"]
    entry["algorithm"] = context["algorithm"] or updated_config.get("algorithm", "xgboost")
    entry["config_path"] = context["config_path"]
    entry["artifact_dir"] = context["artifact_dir"]
    entry["status"] = context["status"]
    entry["dashboard_selectable"] = bool(form_values["dashboard_selectable"])
    entry["outcome_architecture"] = True

    ok, msg = _github_write_file(context["config_path"], _yaml_dump(updated_config), f"Update model config {context['model_id']}")
    if not ok:
        return ok, msg
    ok, msg = _save_registry(updated_registry)
    if not ok:
        return ok, msg
    return True, "Draft model config and registry entry saved."


def _clone_model_to_draft(*, context: dict[str, Any], registry: dict[str, Any], clone_id: str) -> tuple[bool, str]:
    clone_id = str(clone_id or "").strip()
    if not clone_id:
        return False, "Enter a draft model ID."
    if clone_id in (registry.get("models") or {}):
        return False, f"Model already exists: {clone_id}"

    clone_config = deepcopy(context["config"])
    clone_config["model_id"] = clone_id
    clone_config["artifact_name"] = clone_id
    clone_config.setdefault("artifacts", {})["output_dir"] = f"models/{context['model_family']}/{clone_id}"
    clone_config["status"] = "draft"
    clone_config_path = f"configs/models/{clone_id}.yaml"
    clone_artifact_dir = clone_config["artifacts"]["output_dir"]

    updated_registry = deepcopy(registry)
    source_entry = deepcopy(context.get("registry_entry") or {})
    source_entry.update(
        {
            "display_name": f"{context.get('display_name', context['model_id'])} Draft",
            "description": f"Draft clone of {context['model_id']} created from Model Lab.",
            "model_family": context["model_family"],
            "market_key": context["market_key"],
            "algorithm": context["algorithm"] or clone_config.get("algorithm", "xgboost"),
            "config_path": clone_config_path,
            "artifact_dir": clone_artifact_dir,
            "status": "draft",
            "dashboard_selectable": False,
            "outcome_architecture": True,
        }
    )
    updated_registry.setdefault("models", {})[clone_id] = source_entry

    ok, msg = _github_write_file(clone_config_path, _yaml_dump(clone_config), f"Create draft model config {clone_id}")
    if not ok:
        return ok, msg
    ok, msg = _save_registry(updated_registry)
    if not ok:
        return ok, msg
    return True, f"Created draft model {clone_id}."


# -----------------------------------------------------------------------------
# Main render
# -----------------------------------------------------------------------------


def render_model_workflow_launcher() -> None:
    """Render Model Lab V2 control center."""

    _inject_css()
    _render_header()

    try:
        registry = load_model_registry()
        model_rows = get_registered_model_rows(registry)
    except Exception as exc:
        st.error(f"Unable to load model registry: {exc}")
        return

    if not model_rows:
        st.info("No models are registered in configs/models/model_registry.yaml.")
        return

    model_ids = [row["model_id"] for row in model_rows]
    row_by_id = {row["model_id"]: row for row in model_rows}
    selected_model_id = st.selectbox(
        "Selected Model",
        model_ids,
        format_func=lambda model_id: _model_label(row_by_id[model_id]),
        key="model_lab_registered_model_id",
    )

    try:
        context = resolve_model_workflow_context(registry=registry, model_id=selected_model_id)
    except Exception as exc:
        st.error(f"Unable to resolve selected model: {exc}")
        return

    _render_kpis(context)
    _render_model_bar(context, registry)
    _render_registry_table(model_rows)

    top_left, top_right = st.columns([1.05, 1.45], gap="medium")
    with top_left:
        _render_lifecycle_controls(context, registry)
        form_values = _render_config_editor(context, registry)
    with top_right:
        feature_values = _render_feature_bundle_editor(context)
        _render_performance(context)

    if context["status"] in EDITABLE_STATUSES:
        if st.button("Save Draft Configuration", type="primary", use_container_width=True, key="mlab_save_draft_config"):
            ok, msg = _save_draft_model(
                context=context,
                registry=registry,
                form_values=form_values,
                feature_values=feature_values,
            )
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()

    bottom_left, bottom_right = st.columns([1.05, 1.35], gap="medium")
    with bottom_left:
        _render_comparison(context, registry, selected_model_id)
    with bottom_right:
        _render_actions(context)
