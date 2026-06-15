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


def _safe_key(value) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "feature"


@st.cache_data(show_spinner=False)
def _load_feature_bundle_registry(path_text: str = str(FEATURE_BUNDLE_REGISTRY_PATH)) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _bundle_label(bundle_id: str, bundle: dict[str, Any] | None = None) -> str:
    bundle = bundle or {}
    if bundle.get("label"):
        return str(bundle["label"])
    if bundle.get("description"):
        return str(bundle["description"])
    return mlw.BUNDLE_LABELS.get(bundle_id, bundle_id)


def _is_safe_model_lab_feature(feature: str) -> bool:
    """Mirror trainer safety and hide target/outcome leakage columns from UI bundles."""

    name = str(feature or "")
    normalized = name.strip().lower()
    if name.startswith(UNSAFE_FEATURE_PREFIXES):
        return False
    if normalized in UNSAFE_FEATURE_NAMES:
        return False
    if normalized.startswith("winner_is"):
        return False
    return True


def _registry_bundle_map(available: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Resolve feature_bundles.yaml against available feature-view columns.

    Custom bundles created in the Feature workspace live in feature_bundles.yaml.
    Only bundle columns that exist in the current feature view and pass safety
    filtering are exposed in Model Lab Config.
    """

    available_set = set(available)
    registry = _load_feature_bundle_registry()
    bundles = registry.get("bundles", {}) or {}
    resolved: dict[str, list[str]] = {}
    labels: dict[str, str] = {}

    for bundle_id, bundle in bundles.items():
        if not isinstance(bundle, dict):
            continue
        candidates = [str(item) for item in bundle.get("candidate_columns", []) or []]
        safe_candidates = [
            feature
            for feature in candidates
            if feature in available_set and _is_safe_model_lab_feature(feature)
        ]
        if safe_candidates:
            resolved[str(bundle_id)] = sorted(dict.fromkeys(safe_candidates))
            labels[str(bundle_id)] = _bundle_label(str(bundle_id), bundle)

    return resolved, labels


def _model_lab_bundle_map(available: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return registry-defined bundles plus legacy inferred bundles as fallback."""

    registry_bundles, registry_labels = _registry_bundle_map(available)
    legacy_bundles = mlw._bundle_map(available)

    combined: dict[str, list[str]] = dict(registry_bundles)
    labels: dict[str, str] = dict(registry_labels)

    # Keep old heuristic bundles available during migration, but do not let them
    # override real bundle registry definitions with the same ID.
    for bundle_id, features in legacy_bundles.items():
        if bundle_id not in combined:
            combined[bundle_id] = features
            labels[bundle_id] = mlw.BUNDLE_LABELS.get(bundle_id, bundle_id)

    return combined, labels


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

        /* Remove the dark search-input overlay that appears over the first tag. */
        [data-testid="stMultiSelect"] input {
            background: transparent !important;
            background-color: transparent !important;
            color: #ffffff !important;
            caret-color: #ffffff !important;
            box-shadow: none !important;
        }

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

        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"]:hover *,
        [data-baseweb="popover"] [role="option"][aria-selected="true"],
        [data-baseweb="popover"] [role="option"][aria-selected="true"] *,
        [data-baseweb="popover"] [aria-selected="true"],
        [data-baseweb="popover"] [aria-selected="true"] * {
            background: rgba(17,31,49,.98) !important;
            background-color: rgba(17,31,49,.98) !important;
            color: #ffffff !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_feature_checklist(context):
    _inject_feature_selector_css()

    config = context["config"]
    editable = context.get("is_new_model") or context["status"] in mlw.EDITABLE_STATUSES
    model_key = _safe_key(context.get("model_id") or "new_model")
    feature_config = config.get("features") or {}
    current = {feature for feature in set(feature_config.get("feature_columns") or []) if _is_safe_model_lab_feature(feature)}
    available = [feature for feature in mlw._available_feature_columns(context) if _is_safe_model_lab_feature(feature)]
    bundle_map, bundle_labels = _model_lab_bundle_map(available)
    saved = list(feature_config.get("selected_bundles") or [])
    if not saved:
        saved = _infer_selected_bundles(config, available, bundle_map)
    saved_set = set(saved)

    st.markdown("#### Feature Selection")
    selected = st.multiselect(
        "Selected Bundles",
        list(bundle_map.keys()),
        default=[b for b in saved if b in bundle_map],
        format_func=lambda b: f"{bundle_labels.get(b, b)} ({len(bundle_map.get(b, []))})",
        disabled=not editable,
        key=f"mlab_selected_bundles_{model_key}",
    )

    included = []
    removed = []
    universe = []
    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        universe.extend(features)
        label = bundle_labels.get(bundle, bundle)
        with st.expander(f"{label} ({len(features)} features)", expanded=False):
            cols = st.columns(3)
            for i, feature in enumerate(features):
                default = feature in current if bundle in saved_set else True
                key = f"mlab_feature_{model_key}_{_safe_key(bundle)}_{_safe_key(feature)}"
                with cols[i % 3]:
                    value = st.checkbox(feature, value=default, disabled=not editable, key=key)
                if value:
                    included.append(feature)
                else:
                    removed.append(feature)

    resolved = list(dict.fromkeys(included))
    removed = list(dict.fromkeys(removed))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bundles", len(selected))
    c2.metric("Available", len(list(dict.fromkeys(universe))))
    c3.metric("Included", len(resolved))
    c4.metric("Unchecked", len(removed))
    return {"selected_bundles": selected, "include_features": [], "exclude_features": removed, "resolved_features": resolved}
