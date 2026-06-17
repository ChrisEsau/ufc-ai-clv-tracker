from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import streamlit as st

import utils.model_lab_workflows as mlw

ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _primary_model_id(registry: dict[str, Any], family: str) -> str:
    return str(((registry.get("active_models") or {}).get(family) or {}).get("primary") or "")


def _is_primary(registry: dict[str, Any], context: dict[str, Any]) -> bool:
    return _primary_model_id(registry, str(context.get("model_family") or "")) == str(context.get("model_id") or "")


def _promote(context: dict[str, Any], registry: dict[str, Any]) -> tuple[bool, str]:
    model_id = str(context.get("model_id") or "")
    family = str(context.get("model_family") or "")
    status = str(context.get("status") or "").lower()
    if status != "draft":
        return False, "Only draft models can be promoted."

    updated = deepcopy(registry)
    models = updated.setdefault("models", {})
    if model_id not in models:
        return False, f"Model is not registered: {model_id}"

    old_primary = _primary_model_id(updated, family)
    if old_primary and old_primary in models and old_primary != model_id:
        models[old_primary]["status"] = "draft"
        models[old_primary]["dashboard_selectable"] = False

    models[model_id]["status"] = "production"
    models[model_id]["dashboard_selectable"] = True
    updated.setdefault("active_models", {}).setdefault(family, {})["primary"] = model_id

    ok, msg = mlw._save_registry(updated)
    if not ok:
        return ok, msg
    return True, f"Promoted {model_id} to production."


def _demote(context: dict[str, Any], registry: dict[str, Any]) -> tuple[bool, str]:
    model_id = str(context.get("model_id") or "")
    status = str(context.get("status") or "").lower()
    if status != "production":
        return False, "Only production models can be demoted."
    if _is_primary(registry, context):
        return False, "Cannot demote the active primary model. Promote another model first."

    updated = deepcopy(registry)
    models = updated.setdefault("models", {})
    if model_id not in models:
        return False, f"Model is not registered: {model_id}"
    models[model_id]["status"] = "draft"
    models[model_id]["dashboard_selectable"] = False

    ok, msg = mlw._save_registry(updated)
    if not ok:
        return ok, msg
    return True, f"Demoted {model_id} to draft."


def render_lifecycle(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    st.markdown("## Lifecycle")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_model_bar(context, registry)

    model_id = str(context.get("model_id") or "")
    family = str(context.get("model_family") or "")
    status = str(context.get("status") or "").lower()
    primary = _primary_model_id(registry, family)
    is_primary = primary == model_id

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Lifecycle</div>")
    st.caption("Status changes update the model registry only. Model artifacts are not moved or deleted.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Selected Model", model_id or "—")
    with c2:
        st.metric("Status", status.title() if status else "—")
    with c3:
        st.metric("Active Primary", "Yes" if is_primary else "No")

    st.write(f"Current `{family}` primary: `{primary or 'not configured'}`")

    if status == "draft":
        confirm = st.checkbox("Confirm promotion", key=f"mlab_confirm_promote_{model_id}")
        if st.button("Promote to Production", type="primary", disabled=not confirm, use_container_width=True, key=f"mlab_promote_{model_id}"):
            ok, msg = _promote(context, registry)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()
    elif status == "production":
        if is_primary:
            st.warning("This is the active primary model. Promote another model before demoting it.")
        else:
            confirm = st.checkbox("Confirm demotion", key=f"mlab_confirm_demote_{model_id}")
            if st.button("Demote to Draft", disabled=not confirm, use_container_width=True, key=f"mlab_demote_{model_id}"):
                ok, msg = _demote(context, registry)
                st.success(msg) if ok else st.error(msg)
                if ok:
                    st.cache_data.clear()
                    st.rerun()
    else:
        st.info("Only draft models can be promoted. Only production models can be demoted.")

    st.html("</div></div>")
