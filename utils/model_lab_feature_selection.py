from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

import utils.model_lab_workflows as mlw


FEATURE_BUNDLE_REGISTRY_PATH = Path("configs/features/feature_bundles.yaml")
UNSAFE_FEATURE_PREFIXES = ("r_pre_", "b_pre_", "R_", "B_", "r_", "b_")
UNSAFE_FEATURE_NAMES = {
    "winner",
    "winner_id",
    "winner_is_red",
    "winner_is_blue",
    "red_won",
    "blue_won",
    "target",
    "label",
    "outcome",
    "result",
    "fight_result",
}
FEATURE_PANEL_HEIGHT = 520
FEATURE_CHECKBOX_COLUMNS = 2


def _safe_key(value) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "feature"


@st.cache_data(show_spinner=False)
def _load_yaml_registry(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_feature_bundle_registry(path_text: str = str(FEATURE_BUNDLE_REGISTRY_PATH)) -> dict[str, Any]:
    return _load_yaml_registry(path_text)


def _bundle_label(bundle_id: str, bundle: dict[str, Any] | None = None) -> str:
    bundle = bundle or {}
    if bundle.get("label"):
        return str(bundle["label"])
    if bundle.get("description"):
        return str(bundle["description"])
    return mlw.BUNDLE_LABELS.get(bundle_id, bundle_id)


def _is_safe_model_lab_feature(feature: str) -> bool:
    name = str(feature or "")
    normalized = name.strip().lower()
    if name.startswith(UNSAFE_FEATURE_PREFIXES):
        return False
    if normalized in UNSAFE_FEATURE_NAMES:
        return False
    if normalized.startswith("winner_is"):
        return False
    return True


def _registry_bundle_map(available: list[str]) -> tuple[dict[str, list[str]], dict[str, str], dict[str, set[str]]]:
    available_set = set(available)
    registry = _load_feature_bundle_registry()
    bundles = registry.get("bundles", {}) or {}
    resolved: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    availability: dict[str, set[str]] = {}

    for bundle_id, bundle in bundles.items():
        if not isinstance(bundle, dict):
            continue
        candidates = [str(item) for item in bundle.get("candidate_columns", []) or []]
        safe_candidates = [feature for feature in candidates if _is_safe_model_lab_feature(feature)]
        if safe_candidates:
            bundle_key = str(bundle_id)
            resolved[bundle_key] = sorted(dict.fromkeys(safe_candidates))
            labels[bundle_key] = _bundle_label(bundle_key, bundle)
            availability[bundle_key] = {feature for feature in safe_candidates if feature in available_set}

    return resolved, labels, availability


def _model_lab_bundle_map(available: list[str]) -> tuple[dict[str, list[str]], dict[str, str], dict[str, set[str]]]:
    registry_bundles, registry_labels, availability = _registry_bundle_map(available)
    legacy_bundles = mlw._bundle_map(available)

    combined: dict[str, list[str]] = dict(registry_bundles)
    labels: dict[str, str] = dict(registry_labels)

    for bundle_id, features in legacy_bundles.items():
        if bundle_id not in combined:
            combined[bundle_id] = features
            labels[bundle_id] = mlw.BUNDLE_LABELS.get(bundle_id, bundle_id)
            availability[bundle_id] = set(features)

    return combined, labels, availability


def _infer_selected_bundles(config: dict[str, Any], available: list[str], bundle_map: dict[str, list[str]]) -> list[str]:
    feature_config = config.get("features") or {}
    explicit = feature_config.get("selected_bundles") or feature_config.get("bundles")
    if isinstance(explicit, list) and explicit:
        return [str(value) for value in explicit if str(value) in bundle_map]

    current = set(feature_config.get("feature_columns") or [])
    selected = []
    for bundle, cols in bundle_map.items():
        if cols and current.intersection(cols):
            selected.append(bundle)
    return selected or list(bundle_map.keys())


def _inject_feature_selector_css() -> None:
    st.markdown(
        """
        <style>
        .feature-two-pane-caption { color: #9fb0c4; font-size: .78rem; margin: -.15rem 0 .5rem 0; }
        .feature-pane-title { color: #f8fbff; font-size: .95rem; font-weight: 900; margin-bottom: .15rem; }
        .feature-pane-subtitle { color: #38bdf8; font-size: .86rem; font-weight: 800; margin-bottom: .55rem; }
        .feature-selection-scroll-note { color: #8fa4bd; font-size: .72rem; margin-bottom: .55rem; }
        [data-testid="stCheckbox"] label p { font-size: .82rem !important; color: #f8fbff !important; font-weight: 650 !important; overflow-wrap: anywhere !important; }
        [data-testid="stCheckbox"] { min-height: 1.28rem !important; margin-bottom: .08rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-pane-title) { border-color: rgba(43,60,82,.95) !important; border-radius: 12px !important; background: linear-gradient(180deg, rgba(9, 22, 40, .96), rgba(5, 15, 28, .98)) !important; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-pane-title) ::-webkit-scrollbar { width: 9px; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-pane-title) ::-webkit-scrollbar-track { background: rgba(5, 15, 28, .75); border-radius: 999px; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-pane-title) ::-webkit-scrollbar-thumb { background: rgba(76, 112, 154, .85); border-radius: 999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _selection_state(context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    editable = context.get("is_new_model") or context["status"] in mlw.EDITABLE_STATUSES
    model_key = _safe_key(context.get("model_id") or "new_model")
    feature_config = config.get("features") or {}
    current = {feature for feature in set(feature_config.get("feature_columns") or []) if _is_safe_model_lab_feature(feature)}
    available = [feature for feature in mlw._available_feature_columns(context) if _is_safe_model_lab_feature(feature)]
    available_set = set(available)
    bundle_map, bundle_labels, availability = _model_lab_bundle_map(available)
    saved = list(feature_config.get("selected_bundles") or [])
    if not saved:
        saved = _infer_selected_bundles(config, available, bundle_map)
    return {
        "config": config,
        "editable": editable,
        "model_key": model_key,
        "current": current,
        "available": available,
        "available_set": available_set,
        "bundle_map": bundle_map,
        "bundle_labels": bundle_labels,
        "availability": availability,
        "saved": saved,
        "saved_set": set(saved),
    }


def _default_selected_bundles(state: dict[str, Any]) -> list[str]:
    bundle_map = state["bundle_map"]
    return [bundle for bundle in state["saved"] if bundle in bundle_map]


def _render_bundle_selector(state: dict[str, Any]) -> list[str]:
    editable = state["editable"]
    model_key = state["model_key"]
    bundle_map = state["bundle_map"]
    bundle_labels = state["bundle_labels"]
    default_selected = set(_default_selected_bundles(state))
    selected: list[str] = []

    st.markdown('<div class="feature-pane-title">Bundles</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-two-pane-caption">Select bundles to include in this model.</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-selection-scroll-note">Scroll inside this panel for more bundles.</div>', unsafe_allow_html=True)
    for bundle_id, features in bundle_map.items():
        label = f"{bundle_labels.get(bundle_id, bundle_id)} ({len(features)} features)"
        value = st.checkbox(
            label,
            value=bundle_id in default_selected,
            disabled=not editable,
            key=f"mlab_bundle_{model_key}_{_safe_key(bundle_id)}",
        )
        if value:
            selected.append(bundle_id)
    return selected


def _collect_selected_feature_rows(state: dict[str, Any], selected: list[str]) -> list[dict[str, Any]]:
    bundle_map = state["bundle_map"]
    availability = state["availability"]
    current = state["current"]
    available_set = state["available_set"]
    saved_set = state["saved_set"]
    seen_features: set[str] = set()
    rows: list[dict[str, Any]] = []

    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        available_in_bundle = availability.get(bundle, set())
        for feature in features:
            if feature in seen_features:
                continue
            seen_features.add(feature)
            rows.append(
                {
                    "feature": feature,
                    "default": feature in current if bundle in saved_set else True,
                    "checkbox_label": feature if feature in available_set else f"{feature} ⚠",
                    "help_text": None if feature in available_in_bundle else "Configured bundle feature, but not currently present in the feature view. Training may require rebuilding/updating the feature view.",
                    "registry_only": feature not in available_set,
                }
            )
    return rows


def _render_feature_selector(state: dict[str, Any], selected: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    editable = state["editable"]
    model_key = state["model_key"]
    bundle_labels = state["bundle_labels"]

    st.markdown('<div class="feature-pane-title">Features in Selected Bundles</div>', unsafe_allow_html=True)
    if selected:
        bundle_names = ", ".join(bundle_labels.get(bundle, bundle) for bundle in selected[:3])
        suffix = "..." if len(selected) > 3 else ""
        st.markdown(f'<div class="feature-pane-subtitle">{bundle_names}{suffix}</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-selection-scroll-note">Features are shown in two columns. Scroll inside this panel for more.</div>', unsafe_allow_html=True)
    else:
        st.info("Select at least one bundle.")
        return [], [], [], []

    included: list[str] = []
    removed: list[str] = []
    registry_only: list[str] = []
    rows = _collect_selected_feature_rows(state, selected)
    universe = [row["feature"] for row in rows]
    feature_columns = st.columns(FEATURE_CHECKBOX_COLUMNS, gap="medium")

    for index, row in enumerate(rows):
        with feature_columns[index % FEATURE_CHECKBOX_COLUMNS]:
            feature = row["feature"]
            key = f"mlab_feature_{model_key}_{_safe_key(feature)}"
            value = st.checkbox(row["checkbox_label"], value=bool(row["default"]), disabled=not editable, key=key, help=row["help_text"])
            if value:
                included.append(feature)
                if row["registry_only"]:
                    registry_only.append(feature)
            else:
                removed.append(feature)

    return list(dict.fromkeys(included)), list(dict.fromkeys(removed)), list(dict.fromkeys(universe)), list(dict.fromkeys(registry_only))


def render_feature_checklist(context):
    _inject_feature_selector_css()

    state = _selection_state(context)
    st.markdown("#### 5. Feature Selection")

    bundle_col, feature_col = st.columns([0.56, 1.0], gap="large")
    with bundle_col:
        with st.container(border=True, height=FEATURE_PANEL_HEIGHT):
            selected = _render_bundle_selector(state)
    with feature_col:
        with st.container(border=True, height=FEATURE_PANEL_HEIGHT):
            resolved, removed, universe, registry_only = _render_feature_selector(state, selected)

    if registry_only:
        st.warning("Some selected features are configured but not in the current feature view yet: " + ", ".join(registry_only))

    return {"selected_bundles": selected, "include_features": [], "exclude_features": removed, "resolved_features": resolved}
