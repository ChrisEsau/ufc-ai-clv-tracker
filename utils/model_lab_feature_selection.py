from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

import utils.feature_registry as fr
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


def _registry_defined_features() -> set[str]:
    registry = fr.load_feature_registry()
    definitions = registry.get("feature_definitions", {}) or {}
    return {str(feature_id) for feature_id in definitions.keys()}


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
    registry_defined = _registry_defined_features()
    registry = _load_feature_bundle_registry()
    bundles = registry.get("bundles", {}) or {}
    resolved: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    availability: dict[str, set[str]] = {}

    for bundle_id, bundle in bundles.items():
        if not isinstance(bundle, dict):
            continue
        candidates = [str(item) for item in bundle.get("candidate_columns", []) or []]
        safe_candidates = [
            feature
            for feature in candidates
            if _is_safe_model_lab_feature(feature)
            and (feature in available_set or feature in registry_defined)
        ]
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
        .feature-two-pane-caption {
            color: #9fb0c4;
            font-size: .78rem;
            margin: -.15rem 0 .5rem 0;
        }
        .feature-pane-title {
            color: #f8fbff;
            font-size: .95rem;
            font-weight: 900;
            margin-bottom: .15rem;
        }
        .feature-pane-subtitle {
            color: #38bdf8;
            font-size: .86rem;
            font-weight: 800;
            margin-bottom: .55rem;
        }
        .feature-pane-divider {
            border-left: 1px solid rgba(43,60,82,.95);
            padding-left: 1rem;
        }
        [data-testid="stCheckbox"] label p {
            font-size: .82rem !important;
            color: #f8fbff !important;
            font-weight: 650 !important;
        }
        [data-testid="stCheckbox"] {
            min-height: 1.28rem !important;
            margin-bottom: .08rem !important;
        }
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


def _render_feature_selector(state: dict[str, Any], selected: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    editable = state["editable"]
    model_key = state["model_key"]
    bundle_map = state["bundle_map"]
    bundle_labels = state["bundle_labels"]
    availability = state["availability"]
    current = state["current"]
    available_set = state["available_set"]
    saved_set = state["saved_set"]

    st.markdown('<div class="feature-pane-divider">', unsafe_allow_html=True)
    st.markdown('<div class="feature-pane-title">Features in Selected Bundles</div>', unsafe_allow_html=True)
    if selected:
        bundle_names = ", ".join(bundle_labels.get(bundle, bundle) for bundle in selected[:3])
        suffix = "..." if len(selected) > 3 else ""
        st.markdown(f'<div class="feature-pane-subtitle">{bundle_names}{suffix}</div>', unsafe_allow_html=True)
    else:
        st.info("Select at least one bundle.")
        st.markdown('</div>', unsafe_allow_html=True)
        return [], [], [], []

    included: list[str] = []
    removed: list[str] = []
    universe: list[str] = []
    registry_only: list[str] = []
    seen_features: set[str] = set()

    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        available_in_bundle = availability.get(bundle, set())
        for feature in features:
            if feature in seen_features:
                continue
            seen_features.add(feature)
            universe.append(feature)
            default = feature in current if bundle in saved_set else True
            key = f"mlab_feature_{model_key}_{_safe_key(feature)}"
            checkbox_label = feature if feature in available_set else f"{feature} ⚠"
            help_text = None if feature in available_in_bundle else "Registered feature, but not currently present in the feature view. Training may require rebuilding/updating the feature view."
            value = st.checkbox(checkbox_label, value=default, disabled=not editable, key=key, help=help_text)
            if value:
                included.append(feature)
                if feature not in available_set:
                    registry_only.append(feature)
            else:
                removed.append(feature)

    st.markdown('</div>', unsafe_allow_html=True)
    return (
        list(dict.fromkeys(included)),
        list(dict.fromkeys(removed)),
        list(dict.fromkeys(universe)),
        list(dict.fromkeys(registry_only)),
    )


def render_feature_checklist(context):
    _inject_feature_selector_css()

    state = _selection_state(context)
    st.markdown("#### 5. Feature Selection")

    bundle_col, feature_col = st.columns([0.56, 1.0], gap="large")
    with bundle_col:
        selected = _render_bundle_selector(state)
    with feature_col:
        resolved, removed, universe, registry_only = _render_feature_selector(state, selected)

    if registry_only:
        st.warning(
            "Some selected features are registered but not in the current feature view yet: "
            + ", ".join(registry_only)
        )

    return {"selected_bundles": selected, "include_features": [], "exclude_features": removed, "resolved_features": resolved}
