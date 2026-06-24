from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import streamlit as st

import utils.model_lab_workflows as mlw

ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _market_key(context: dict[str, Any]) -> str:
    family = str(context.get("model_family") or "").strip().lower()
    market = str(context.get("market_key") or "").strip().lower()
    if market:
        return market
    return "moneyline" if family == "moneyline" else ""


def _active_model_id(registry: dict[str, Any], family: str, market_key: str = "") -> str:
    family = str(family or "").strip().lower()
    market_key = str(market_key or "").strip().lower()
    active_family = ((registry.get("active_models") or {}).get(family) or {})
    if market_key:
        market_model = active_family.get(market_key)
        if market_model:
            return str(market_model)
        if family == "moneyline" and market_key == "moneyline":
            return str(active_family.get("primary") or "")
        return ""
    return str(active_family.get("primary") or "")


def _primary_model_id(registry: dict[str, Any], family: str) -> str:
    return _active_model_id(registry, family)


def _is_primary(registry: dict[str, Any], context: dict[str, Any]) -> bool:
    family = str(context.get("model_family") or "")
    market = _market_key(context)
    return _active_model_id(registry, family, market) == str(context.get("model_id") or "")


def _production_models_for_scope(
    registry: dict[str, Any],
    *,
    family: str,
    market_key: str,
) -> list[str]:
    """Return production model IDs in the same family/market scope."""

    matches: list[str] = []
    for model_id, entry in (registry.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        entry_family = str(entry.get("model_family") or "").strip().lower()
        entry_market = str(entry.get("market_key") or "").strip().lower()
        entry_status = str(entry.get("status") or "").strip().lower()
        if entry_family == family and entry_market == market_key and entry_status == "production":
            matches.append(str(model_id))
    return matches


def _set_active_model(updated: dict[str, Any], *, family: str, market_key: str, model_id: str) -> None:
    family = str(family or "").strip().lower()
    market_key = str(market_key or "").strip().lower()
    active_family = updated.setdefault("active_models", {}).setdefault(family, {})
    if market_key:
        active_family[market_key] = model_id

    # Moneyline has a true single primary. Props do not; keep prop.primary only as a
    # compatibility fallback and do not rewrite it every time a new prop market is promoted.
    if family == "moneyline" or not active_family.get("primary"):
        active_family["primary"] = model_id


def _promote(context: dict[str, Any], registry: dict[str, Any]) -> tuple[bool, str]:
    model_id = str(context.get("model_id") or "")
    family = str(context.get("model_family") or "").strip().lower()
    market = _market_key(context)
    status = str(context.get("status") or "").lower()
    if status != "draft":
        return False, "Only draft models can be promoted."
    if not family:
        return False, "Model family is required for promotion."
    if family == "prop" and not market:
        return False, "Prop model promotion requires a market_key."

    updated = deepcopy(registry)
    models = updated.setdefault("models", {})
    if model_id not in models:
        return False, f"Model is not registered: {model_id}"

    same_market_production = _production_models_for_scope(updated, family=family, market_key=market)
    old_active = _active_model_id(updated, family, market)
    if old_active and old_active not in same_market_production:
        same_market_production.append(old_active)

    for old_model_id in same_market_production:
        if old_model_id in models and old_model_id != model_id:
            models[old_model_id]["status"] = "draft"
            models[old_model_id]["dashboard_selectable"] = False

    models[model_id]["status"] = "production"
    models[model_id]["dashboard_selectable"] = True
    models[model_id]["model_family"] = family
    models[model_id]["market_key"] = market
    _set_active_model(updated, family=family, market_key=market, model_id=model_id)

    ok, msg = mlw._save_registry(updated)
    if not ok:
        return ok, msg
    market_label = f" for {market}" if market else ""
    return True, f"Promoted {model_id} to production{market_label}."


def _demote(context: dict[str, Any], registry: dict[str, Any]) -> tuple[bool, str]:
    model_id = str(context.get("model_id") or "")
    market = _market_key(context)
    status = str(context.get("status") or "").lower()
    if status != "production":
        return False, "Only production models can be demoted."
    if _is_primary(registry, context):
        market_label = f" for {market}" if market else ""
        return False, f"Cannot demote the active production model{market_label}. Promote another model first."

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
    family = str(context.get("model_family") or "").strip().lower()
    market = _market_key(context)
    status = str(context.get("status") or "").lower()
    active_model = _active_model_id(registry, family, market)
    is_primary = active_model == model_id

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Lifecycle</div>")
    st.caption("Status changes update the model registry only. Model artifacts are not moved or deleted.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Selected Model", model_id or "—")
    with c2:
        st.metric("Status", status.title() if status else "—")
    with c3:
        st.metric("Active for Market", "Yes" if is_primary else "No")

    scope_label = f"{family}/{market}" if market else family
    st.write(f"Current `{scope_label}` production model: `{active_model or 'not configured'}`")

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
            st.warning("This is the active production model for this market. Promote another model for the same market before demoting it.")
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
