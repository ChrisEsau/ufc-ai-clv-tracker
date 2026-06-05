import pandas as pd
import streamlit as st
import plotly.express as px

from utils.bankroll_artifacts import (
    BankrollSettings,
    bankroll_summary,
    derive_open_bets,
    exposure_by_event,
    load_bankroll_settings,
    load_bet_ledger,
    performance_by_event,
    save_bankroll_settings,
    settle_bet,
)
from utils.panels import render_section_header
from utils.ui_components import american, money, pct, render_metric
from utils.ui.sections import page_header
from utils.ui.charts import apply_plotly_theme


RESULT_OPTIONS = ["Open", "Win", "Loss", "Push", "Void"]


def _format_roi(value):
    if pd.isna(value):
        return "0.0%"
    return f"{float(value) * 100:.1f}%"


def _display_money_columns(df, columns):
    display = df.copy()
    for column in columns:
        if column in display.columns:
            display[f"{column}_display"] = display[column].apply(money)
    return display


def render_bankroll_summary(ledger, settings):
    render_section_header("Bankroll Summary")
    summary = bankroll_summary(ledger=ledger, settings=settings)

    cols = st.columns(6)
    with cols[0]:
        render_metric("Starting", money(summary["starting_bankroll"]))
    with cols[1]:
        render_metric("Current", money(summary["current_bankroll"]), accent="green" if summary["total_profit"] >= 0 else "red")
    with cols[2]:
        render_metric("Available", money(summary["available_bankroll"]))
    with cols[3]:
        render_metric("Open Risk", money(summary["open_risk"]), accent="amber" if summary["open_risk"] else "neutral")
    with cols[4]:
        render_metric("Total Profit", money(summary["total_profit"]), accent="green" if summary["total_profit"] >= 0 else "red")
    with cols[5]:
        render_metric("ROI", _format_roi(summary["roi"]))


def render_open_exposure(ledger):
    render_section_header("Open Exposure")
    open_bets = derive_open_bets(ledger)

    if open_bets.empty:
        st.info("No open bets are currently tracked in the bankroll ledger.")
        return

    open_display = _display_money_columns(open_bets, ["stake"])
    open_cols = [
        "event_name",
        "fighter",
        "opponent",
        "market_type",
        "odds_taken",
        "stake_display",
        "model_probability",
        "edge",
        "ev",
        "placed_timestamp",
    ]
    open_cols = [column for column in open_cols if column in open_display.columns]
    st.dataframe(open_display[open_cols], use_container_width=True, hide_index=True)

    event_exposure = exposure_by_event(open_bets)
    if not event_exposure.empty:
        event_display = _display_money_columns(event_exposure, ["open_risk", "potential_profit"])
        st.markdown("**Exposure by event**")
        st.dataframe(
            event_display[["event_name", "open_bets", "open_risk_display", "potential_profit_display"]],
            use_container_width=True,
            hide_index=True,
        )


def render_ledger(ledger):
    render_section_header("Official Bet Ledger")

    if ledger.empty:
        st.info("No official bets have been added to the ledger yet.")
        return

    with st.expander("Ledger filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            result_values = sorted(ledger["result"].dropna().unique().tolist()) if "result" in ledger.columns else []
            selected_results = st.multiselect("Result", result_values, default=result_values)
        with c2:
            events = sorted(ledger["event_name"].dropna().unique().tolist()) if "event_name" in ledger.columns else []
            selected_event = st.selectbox("Event", ["All Events"] + events)
        with c3:
            markets = sorted(ledger["market_type"].dropna().unique().tolist()) if "market_type" in ledger.columns else []
            selected_market = st.selectbox("Market", ["All Markets"] + markets)

    filtered = ledger.copy()
    if selected_results and "result" in filtered.columns:
        filtered = filtered[filtered["result"].isin(selected_results)]
    if selected_event != "All Events" and "event_name" in filtered.columns:
        filtered = filtered[filtered["event_name"] == selected_event]
    if selected_market != "All Markets" and "market_type" in filtered.columns:
        filtered = filtered[filtered["market_type"] == selected_market]

    display = _display_money_columns(filtered, ["stake", "profit_loss"])
    if "odds_taken" in display.columns:
        display["odds_display"] = display["odds_taken"].apply(american)
    for column in ["model_probability", "implied_probability", "edge", "ev", "clv"]:
        if column in display.columns:
            display[f"{column}_display"] = display[column].apply(pct)

    ledger_cols = [
        "result",
        "event_name",
        "fighter",
        "market_type",
        "odds_display",
        "stake_display",
        "profit_loss_display",
        "model_probability_display",
        "edge_display",
        "ev_display",
        "clv_display",
        "placed_timestamp",
        "settled_timestamp",
        "bet_id",
    ]
    ledger_cols = [column for column in ledger_cols if column in display.columns]
    st.dataframe(display[ledger_cols], use_container_width=True, hide_index=True)


def render_performance(ledger):
    render_section_header("Performance Charts")

    if ledger.empty:
        st.info("Performance charts will populate after bets are added and settled.")
        return

    event_perf = performance_by_event(ledger)
    if event_perf.empty:
        st.info("No settled bets are available yet for performance analytics.")
        return

    chart_col, table_col = st.columns([1.1, 1])
    with chart_col:
        chart = event_perf.copy()
        chart["bar_color"] = chart["profit_loss"].apply(lambda value: "Profit" if value >= 0 else "Loss")
        fig = px.bar(
            chart,
            x="event_name",
            y="profit_loss",
            color="bar_color",
            color_discrete_map={"Profit": "#35d96b", "Loss": "#ef4444"},
            labels={"event_name": "Event", "profit_loss": "Profit / Loss"},
        )
        st.plotly_chart(apply_plotly_theme(fig, height=320), use_container_width=True)

    display = _display_money_columns(event_perf, ["profit_loss", "stake"])
    display["roi_display"] = display["roi"].apply(_format_roi)
    with table_col:
        st.dataframe(
            display[["event_name", "bets", "profit_loss_display", "stake_display", "roi_display"]],
            use_container_width=True,
            hide_index=True,
        )


def render_clv_quality(ledger):
    render_section_header("CLV & Bet Quality")

    if ledger.empty or "clv" not in ledger.columns:
        st.info("CLV quality metrics will populate after closing-line data is recorded on ledger bets.")
        return

    clv = pd.to_numeric(ledger["clv"], errors="coerce").dropna()
    if clv.empty:
        st.info("No CLV values are recorded yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Avg CLV", _format_roi(clv.mean()))
    cols[1].metric("Median CLV", _format_roi(clv.median()))
    cols[2].metric("Positive CLV", _format_roi((clv > 0).mean()))
    cols[3].metric("CLV Samples", len(clv))


def render_risk_settings(settings):
    render_section_header("Risk Settings")

    with st.form("bankroll_settings_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            starting_bankroll = st.number_input("Starting bankroll", min_value=0.0, value=float(settings.starting_bankroll), step=100.0)
            kelly_fraction = st.number_input("Kelly fraction", min_value=0.0, max_value=2.0, value=float(settings.kelly_fraction), step=0.05)
        with c2:
            max_stake_pct = st.number_input("Max stake %", min_value=0.0, max_value=100.0, value=float(settings.max_stake_pct * 100), step=0.25)
            max_event_exposure_pct = st.number_input("Max event exposure %", min_value=0.0, max_value=100.0, value=float(settings.max_event_exposure_pct * 100), step=0.5)
        with c3:
            min_edge = st.number_input("EV/edge threshold", min_value=-100.0, max_value=100.0, value=float(settings.min_edge), step=0.01)
            min_confidence = st.number_input("Confidence threshold", min_value=0.0, max_value=100.0, value=float(settings.min_confidence), step=1.0)
        with c4:
            min_odds = st.number_input("Min odds", min_value=-1000, max_value=1000, value=int(settings.min_odds), step=5)
            max_odds = st.number_input("Max odds", min_value=-1000, max_value=2000, value=int(settings.max_odds), step=5)

        submitted = st.form_submit_button("Save Risk Settings", use_container_width=True)

    if submitted:
        updated = BankrollSettings(
            starting_bankroll=float(starting_bankroll),
            kelly_fraction=float(kelly_fraction),
            max_stake_pct=float(max_stake_pct) / 100,
            max_event_exposure_pct=float(max_event_exposure_pct) / 100,
            min_edge=float(min_edge),
            min_confidence=float(min_confidence),
            min_odds=int(min_odds),
            max_odds=int(max_odds),
        )
        save_bankroll_settings(updated)
        st.success("Saved bankroll risk settings.")
        st.cache_data.clear()
        st.rerun()


def render_settlement_controls(ledger):
    render_section_header("Settle Open Bet")
    open_bets = derive_open_bets(ledger)
    if open_bets.empty:
        st.info("No open bets are available to settle.")
        return

    open_bets = open_bets.copy()
    open_bets["label"] = open_bets.apply(
        lambda row: f"{row.get('event_name', '')} — {row.get('fighter', '')} {american(row.get('odds_taken'))} ({money(row.get('stake'))})",
        axis=1,
    )
    selected = st.selectbox("Open bet", open_bets.to_dict("records"), format_func=lambda row: row["label"])

    with st.form("settle_open_bet_form"):
        result = st.selectbox("Result", RESULT_OPTIONS[1:])
        closing_odds = st.number_input("Closing odds", min_value=-2000, max_value=2000, value=0, step=5)
        clv = st.number_input("CLV", min_value=-10.0, max_value=10.0, value=0.0, step=0.01, format="%.3f")
        notes = st.text_input("Settlement notes", value=str(selected.get("notes", "") or ""))
        submitted = st.form_submit_button("Settle Bet", use_container_width=True)

    if submitted:
        ok = settle_bet(
            selected["bet_id"],
            result=result,
            closing_odds=None if closing_odds == 0 else closing_odds,
            clv=clv,
            notes=notes,
        )
        if ok:
            st.success("Bet settled and bankroll ledger updated.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Could not find the selected bet in the ledger.")


def render_bankroll():
    page_header(
        "Bankroll",
        "Track performance, manage open risk, settle wagers, and maintain bankroll settings.",
        kicker="Financial Control Center",
    )

    settings = load_bankroll_settings()
    ledger = load_bet_ledger()

    render_bankroll_summary(ledger, settings)
    render_open_exposure(ledger)
    render_ledger(ledger)
    render_performance(ledger)
    render_clv_quality(ledger)
    render_settlement_controls(ledger)
    render_risk_settings(settings)
