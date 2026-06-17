from __future__ import annotations

from typing import Any, Callable

import streamlit as st

import utils.model_lab_workflows as mlw
from tabs.model_lab_sections.lifecycle import render_lifecycle
from utils.workflow_status import launch_workflow_with_status, workflow_status_label


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _dispatch_button_with_status(
    label: str,
    workflow_file: str,
    inputs: dict[str, str],
    disabled: bool,
    key: str,
) -> None:
    status = workflow_status_label(workflow_file, key, idle_label="Ready")
    st.caption(f"Status: {status}")
    if st.button(label, disabled=disabled, use_container_width=True, key=key):
        ok, message = launch_workflow_with_status(workflow_file, key, inputs=inputs)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)


def _render_actions_with_status(context: dict[str, Any]) -> None:
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

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _dispatch_button_with_status(
            "Build Fighter State",
            "run-build-fighter-state-v2.yml",
            {},
            False,
            f"mlab_fighter_state_{context['model_id']}",
        )
    with c2:
        _dispatch_button_with_status(
            "Build Feature View",
            mlw.WORKFLOWS["feature_view"],
            feature_inputs,
            not bool(feature_inputs["config_path"] and feature_inputs["output_path"]),
            f"mlab_build_{context['model_id']}",
        )
    with c3:
        _dispatch_button_with_status(
            "Train Model",
            mlw.WORKFLOWS["training"],
            training_inputs,
            not bool(training_inputs["config_path"] and training_inputs["artifact_dir"]),
            f"mlab_train_{context['model_id']}",
        )
    with c4:
        _dispatch_button_with_status(
            "Run Predictions",
            mlw.WORKFLOWS["prediction"],
            prediction_inputs,
            not bool(prediction_inputs["model_family"] and prediction_inputs["model_id"]),
            f"mlab_predict_{context['model_id']}",
        )
    with c5:
        model_mode = st.selectbox("Betting Mode", ["production", "all", "single"], key="mlab_betting_mode")
        _dispatch_button_with_status(
            "Run Outcomes",
            mlw.WORKFLOWS["betting_outcomes"],
            {"model_mode": model_mode},
            False,
            f"mlab_outcomes_{context['model_id']}",
        )

    st.html("</div></div>")


def render_actions(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    st.markdown("## Actions")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_model_bar(context, registry)
    _render_actions_with_status(context)
    st.divider()
    render_lifecycle(
        registry,
        rows,
        row_by_id,
        existing_model_selector=existing_model_selector,
    )
