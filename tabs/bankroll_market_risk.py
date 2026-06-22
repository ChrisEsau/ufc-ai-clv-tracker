from __future__ import annotations

import json

import streamlit as st

import tabs.bankroll as base
from pipeline.common.risk_settings import MarketRiskFilter, RiskSettings
from utils.ui.market_risk_controls import (
    market_filters_summary_frame,
    market_filters_to_payload,
    render_market_risk_controls,
)


def _risk_settings_workflow_inputs(settings: RiskSettings) -> dict[str, str]:
    """Build workflow_dispatch inputs with per-market filter payloads."""

    return {
        "settings_json": json.dumps(
            {
                "starting_bankroll": settings.starting_bankroll,
                "kelly_fraction": settings.kelly_fraction,
                "max_stake_pct": settings.max_stake_pct,
                "max_event_exposure_pct": settings.max_event_exposure_pct,
                "market_filters": market_filters_to_payload(settings.market_filters),
            },
            default=str,
        )
    }


def _risk_settings_html(settings: RiskSettings) -> str:
    """Render global staking summary plus market-specific filter table."""

    global_items = [
        ("Kelly Fraction", f"{settings.kelly_fraction:.2f}"),
        ("Max Stake Per Bet", base._pct(settings.max_stake_pct)),
        ("Max Event Exposure", base._pct(settings.max_event_exposure_pct)),
        ("Starting Bankroll", base._money(settings.starting_bankroll)),
    ]
    global_html = (
        "<div class='bankroll-settings'>"
        + "".join(
            f"<div class='bankroll-setting'><div class='bankroll-setting-label'>{base._escape(label)}</div>"
            f"<div class='bankroll-setting-value'>{base._escape(value)}</div></div>"
            for label, value in global_items
        )
        + "</div>"
    )

    summary = market_filters_summary_frame(settings)
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{base._escape(row.get('Market'))}</td>"
            f"<td class='bankroll-right'>{base._escape(row.get('Min Edge'))}</td>"
            f"<td class='bankroll-right'>{base._escape(row.get('Min Confidence'))}</td>"
            f"<td class='bankroll-right'>{base._escape(row.get('Odds Range'))}</td>"
            "</tr>"
        )
    market_html = (
        "<div class='bankroll-chart-wrap' style='padding-top:0;'>"
        "<table class='bankroll-table'><thead><tr>"
        "<th>Market</th><th class='bankroll-right'>Min Edge</th>"
        "<th class='bankroll-right'>Min Confidence</th><th class='bankroll-right'>Odds Range</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return global_html + market_html


def _render_risk_settings_form(settings: RiskSettings, in_dialog: bool = False) -> None:
    st.caption(
        "Update global bankroll/staking settings and market-specific qualification filters. "
        "Saving launches a GitHub workflow so the settings persist beyond this Streamlit session."
    )
    with st.form("bankroll_market_risk_settings_dialog_form"):
        st.markdown("#### Global Bankroll / Staking")
        starting_bankroll = st.number_input(
            "Starting bankroll",
            min_value=0.0,
            value=float(settings.starting_bankroll),
            step=100.0,
        )
        kelly_options = [0.25, 0.50]
        kelly_fraction = st.radio(
            "Kelly fraction",
            kelly_options,
            index=kelly_options.index(0.25) if float(settings.kelly_fraction) == 0.25 else kelly_options.index(0.50),
            format_func=lambda value: f"{value:.2f}",
            horizontal=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            max_stake_pct = st.number_input(
                "Max stake per bet (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(settings.max_stake_pct * 100),
                step=0.25,
            )
        with col2:
            max_event_exposure_pct = st.number_input(
                "Max event exposure (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(settings.max_event_exposure_pct * 100),
                step=0.5,
            )

        market_filters = render_market_risk_controls(settings)
        submitted = st.form_submit_button("Save Risk Settings", use_container_width=True)

    if submitted:
        for market_key, market_filter in market_filters.items():
            if market_filter.min_odds > market_filter.max_odds:
                st.error(f"{market_key}: minimum odds cannot be greater than maximum odds.")
                return

        moneyline_filter = market_filters.get("moneyline", MarketRiskFilter())
        updated = RiskSettings(
            starting_bankroll=float(starting_bankroll),
            kelly_fraction=float(kelly_fraction),
            max_stake_pct=float(max_stake_pct) / 100,
            max_event_exposure_pct=float(max_event_exposure_pct) / 100,
            min_edge=moneyline_filter.min_edge,
            min_confidence=moneyline_filter.min_confidence,
            min_odds=moneyline_filter.min_odds,
            max_odds=moneyline_filter.max_odds,
            market_filters=market_filters,
        )
        ok, msg = base._dispatch_risk_settings(updated)
        if ok:
            st.success("Risk settings workflow launched. Refresh after it completes to load the committed settings.")
            st.cache_data.clear()
        else:
            st.error(f"Could not launch risk settings workflow: {msg}")
            return
        if in_dialog:
            st.session_state["bankroll_dialog"] = None
        st.rerun()


# Patch only the risk-settings behavior on the existing Bankroll tab.
base._risk_settings_workflow_inputs = _risk_settings_workflow_inputs
base._risk_settings_html = _risk_settings_html
base._render_risk_settings_form = _render_risk_settings_form

render_bankroll = base.render_bankroll
