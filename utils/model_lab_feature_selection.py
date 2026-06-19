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
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: linear-gradient(180deg, #2563eb, #1d4ed8) !important;
            background-color: #2563eb !important;
            border: 1px solid #60a5fa !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 1px rgba(59,130,246,.22) inset !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            opacity: 1 !important;
        }
        [data-testid="stMultiSelect"] input,
        [data-testid="stMultiSelect"] [data-baseweb="input"],
        [data-testid="stMultiSelect"] [data-baseweb="input"] > div {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] [data-baseweb="menu"],
        [data-baseweb="popover"] [role="listbox"] {
            background: rgba(7,16,28,.98) !important;
            background-color: rgba(7,16,28,.98) !important;
            color: #ffffff !important;
        }
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="popover"] [role="option"] *,
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] li * {
            color: #ffffff !important;
            background-color: rgba(7,16,28,.98) !important;
            opacity: 1 !important;
        }
        details[data-testid="stExpander"] summary,
        details[data-testid="stExpander"] summary * {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        details[data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(9,19,32,.95), rgba(7,16,28,.98)) !important;
            border: 1px solid rgba(51,75,108,.95) !important;
            border-radius: 10px !important;
        }
        .feature-summary-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto auto;
            gap: .65rem;
            align-items: center;
            padding: .5rem .65rem;
            margin-bottom: .35rem;
            border: 1px solid rgba(43,60,82,.95);
            border-radius: 9px;
            background: rgba(7,17,31,.66);
        }
        .feature-summary-name {
            color: #f8fbff;
            font-weight: 760;
            font-size: .84rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .feature-summary-count {
            color: #9fb0c4;
            font-size: .76rem;
            white-space: nowrap;
        }
        .feature-summary-pill {
            color: #6ee7b7;
            border: 1px solid rgba(110,231,183,.35);
            background: rgba(6,78,59,.22);
            padding: .1rem .45rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .feature-editor-shell {
            border: 1px solid rgba(59,130,246,.35);
            border-radius: 12px;
            padding: .9rem;
            margin-top: .75rem;
            background: linear-gradient(180deg, rgba(10,31,55,.84), rgba(5,15,28,.92));
            box-shadow: 0 18px 40px rgba(0,0,0,.28);
        }
        .feature-editor-title {
            color: #f8fbff;
            font-size: 1rem;
            font-weight: 900;
            margin-bottom: .15rem;
        }
        .feature-editor-caption {
            color: #9fb0c4;
            font-size: .78rem;
            margin-bottom: .7rem;
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


def _default_selected_bundles(state: dict[str, Any], selected_key: str) -> list[str]:
    bundle_map = state["bundle_map"]
    session_value = st.session_state.get(selected_key)
    if isinstance(session_value, list):
        return [bundle for bundle in session_value if bundle in bundle_map]
    return [bundle for bundle in state["saved"] if bundle in bundle_map]


def _resolve_features_without_editor(state: dict[str, Any], selected: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    bundle_map = state["bundle_map"]
    current = state["current"]
    saved_set = state["saved_set"]
    available_set = state["available_set"]

    included: list[str] = []
    removed: list[str] = []
    universe: list[str] = []
    registry_only: list[str] = []
    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        universe.extend(features)
        for feature in features:
            default = feature in current if bundle in saved_set else True
            if default:
                included.append(feature)
                if feature not in available_set:
                    registry_only.append(feature)
            else:
                removed.append(feature)
    return (
        list(dict.fromkeys(included)),
        list(dict.fromkeys(removed)),
        list(dict.fromkeys(universe)),
        list(dict.fromkeys(registry_only)),
    )


def _render_feature_summary(state: dict[str, Any], selected: list[str], resolved: list[str], removed: list[str], universe: list[str]) -> None:
    bundle_map = state["bundle_map"]
    bundle_labels = state["bundle_labels"]
    availability = state["availability"]

    if not selected:
        st.info("No feature bundles selected.")
        return

    for bundle in selected:
        features = bundle_map.get(bundle, [])
        available_count = len(availability.get(bundle, set()))
        included_count = len([feature for feature in features if feature in set(resolved)])
        label = bundle_labels.get(bundle, bundle)
        st.markdown(
            f"""
            <div class="feature-summary-row">
                <div class="feature-summary-name">{label}</div>
                <div class="feature-summary-count">{available_count}/{len(features)} available</div>
                <div class="feature-summary-pill">{included_count} included</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bundles", len(selected))
    c2.metric("Available", len(list(dict.fromkeys(universe))))
    c3.metric("Included", len(resolved))
    c4.metric("Excluded", len(removed))


def _render_feature_editor(state: dict[str, Any], selected_key: str) -> tuple[list[str], list[str], list[str], list[str]]:
    editable = state["editable"]
    model_key = state["model_key"]
    bundle_map = state["bundle_map"]
    bundle_labels = state["bundle_labels"]
    availability = state["availability"]
    current = state["current"]
    available_set = state["available_set"]
    saved_set = state["saved_set"]

    st.markdown(
        """
        <div class="feature-editor-shell">
            <div class="feature-editor-title">Manage Feature Bundles</div>
            <div class="feature-editor-caption">Select bundles, then optionally uncheck individual features inside each bundle.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.multiselect(
        "Selected Bundles",
        list(bundle_map.keys()),
        default=_default_selected_bundles(state, selected_key),
        format_func=lambda b: f"{bundle_labels.get(b, b)} ({len(bundle_map.get(b, []))})",
        disabled=not editable,
        key=selected_key,
    )

    included = []
    removed = []
    universe = []
    registry_only = []
    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        available_in_bundle = availability.get(bundle, set())
        universe.extend(features)
        label = bundle_labels.get(bundle, bundle)
        with st.expander(f"{label} ({len(features)} features)", expanded=False):
            cols = st.columns(3)
            for i, feature in enumerate(features):
                default = feature in current if bundle in saved_set else True
                key = f"mlab_feature_{model_key}_{_safe_key(bundle)}_{_safe_key(feature)}"
                checkbox_label = feature if feature in available_set else f"{feature} ⚠ registry-only"
                help_text = None if feature in available_in_bundle else "Registered feature, but not currently present in the feature view. Training may require rebuilding/updating the feature view."
                with cols[i % 3]:
                    value = st.checkbox(checkbox_label, value=default, disabled=not editable, key=key, help=help_text)
                if value:
                    included.append(feature)
                    if feature not in available_set:
                        registry_only.append(feature)
                else:
                    removed.append(feature)

    return (
        selected,
        list(dict.fromkeys(included)),
        list(dict.fromkeys(removed)),
        list(dict.fromkeys(universe)),
        list(dict.fromkeys(registry_only)),
    )


def render_feature_checklist(context):
    _inject_feature_selector_css()

    state = _selection_state(context)
    model_key = state["model_key"]
    selected_key = f"mlab_selected_bundles_{model_key}"
    editor_key = f"mlab_manage_bundles_open_{model_key}"

    st.markdown("#### Feature Selection")
    top_col1, top_col2 = st.columns([1.0, 0.34])
    with top_col1:
        st.caption("Model-specific bundle summary. Open Manage Bundles for detailed feature editing.")
    with top_col2:
        if st.button("Manage Bundles", use_container_width=True, key=f"mlab_manage_bundles_button_{model_key}"):
            st.session_state[editor_key] = not bool(st.session_state.get(editor_key, False))
            st.rerun()

    editor_open = bool(st.session_state.get(editor_key, False))
    selected = _default_selected_bundles(state, selected_key)

    if editor_open:
        selected, resolved, removed, universe, registry_only = _render_feature_editor(state, selected_key)
        done_col1, done_col2 = st.columns([1.0, 0.22])
        with done_col2:
            if st.button("Done", use_container_width=True, key=f"mlab_manage_bundles_done_{model_key}"):
                st.session_state[editor_key] = False
                st.rerun()
    else:
        resolved, removed, universe, registry_only = _resolve_features_without_editor(state, selected)

    _render_feature_summary(state, selected, resolved, removed, universe)

    if registry_only:
        st.warning(
            "Some selected features are registered but not in the current feature view yet: "
            + ", ".join(registry_only)
        )

    return {"selected_bundles": selected, "include_features": [], "exclude_features": removed, "resolved_features": resolved}
