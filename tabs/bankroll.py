from __future__ import annotations

import hashlib
import html
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.common.paths import BETTING_BOARD_PATH, LIVE_CARD_PATH, MASTER_PATH
from utils.bankroll_artifacts import (
    BankrollSettings,
    american_profit,
    bankroll_summary,
    derive_open_bets,
    exposure_by_event,
    is_open_result,
    load_bankroll_settings,
    load_bet_ledger,
    normalize_ledger,
    performance_by_event,
    save_bankroll_settings,
    save_bet_ledger,
    settle_bet,
)
from utils.data_loader import load_parquet
from utils.github_actions import trigger_workflow
from utils.ui.charts import apply_plotly_theme

RESULT_OPTIONS = ["Win", "Loss", "Push", "Void"]
BANKROLL_STATUS_WORKFLOW = "run-bankroll-status.yml"
CENTRAL_TZ = ZoneInfo("America/Chicago")


def _escape(value) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def _money(value) -> str:
    if value is None or pd.isna(value):
        return "$0.00"
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _signed_money(value) -> str:
    if value is None or pd.isna(value):
        return "$0.00"
    value = float(value)
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _pct(value, decimals: int = 1, already_pct: bool = False, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "0.0%"
    value = float(value if already_pct else value * 100)
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _american(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    odds = int(round(float(value)))
    return f"+{odds}" if odds > 0 else str(odds)


def _as_float(value, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def _last_updated(ledger: pd.DataFrame) -> str:
    candidates = []
    for column in ["settled_timestamp", "placed_timestamp"]:
        if column in ledger.columns:
            parsed = pd.to_datetime(ledger[column], utc=True, errors="coerce").dropna()
            if not parsed.empty:
                candidates.append(parsed.max())
    if not candidates:
        return "Last Updated: No ledger activity"
    latest = max(candidates).tz_convert(CENTRAL_TZ)
    return f"Last Updated: {latest.strftime('%b %-d, %Y %I:%M %p %Z')}"


def _ledger_csv(ledger: pd.DataFrame) -> str:
    if ledger.empty:
        return ""
    return ledger.to_csv(index=False)


def _ledger_summary(ledger: pd.DataFrame, settings: BankrollSettings) -> dict:
    summary = bankroll_summary(ledger=ledger, settings=settings)
    settled = ledger[~ledger["result"].apply(is_open_result)].copy() if not ledger.empty else pd.DataFrame()
    wins = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.lower() == "win").sum()) if not settled.empty else 0
    losses = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.lower() == "loss").sum()) if not settled.empty else 0
    pushes = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.lower().isin(["push", "void"])).sum()) if not settled.empty else 0
    decisions = wins + losses
    win_rate = wins / decisions if decisions else 0.0
    summary.update(
        {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_rate,
            "total_stake": float(pd.to_numeric(ledger.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not ledger.empty else 0.0,
        }
    )
    return summary


def _potential_profit(row: pd.Series) -> float:
    return max(american_profit(row.get("stake"), row.get("odds_taken"), "Win"), 0.0)


def _open_exposure_by_fighter(open_bets: pd.DataFrame) -> pd.DataFrame:
    if open_bets.empty:
        return pd.DataFrame(columns=["fighter", "stake", "potential_profit", "open_bets"])
    work = open_bets.copy()
    work["stake"] = pd.to_numeric(work["stake"], errors="coerce").fillna(0.0)
    work["potential_profit"] = work.apply(_potential_profit, axis=1)
    return (
        work.groupby("fighter", dropna=False)
        .agg(open_bets=("bet_id", "count"), stake=("stake", "sum"), potential_profit=("potential_profit", "sum"))
        .reset_index()
        .sort_values("stake", ascending=False)
    )


def _bankroll_points(ledger: pd.DataFrame, settings: BankrollSettings) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["date", "bankroll"])
    settled = ledger[~ledger["result"].apply(is_open_result)].copy()
    if settled.empty:
        return pd.DataFrame(columns=["date", "bankroll"])
    settled["settled_timestamp"] = pd.to_datetime(settled["settled_timestamp"], utc=True, errors="coerce")
    settled["placed_timestamp"] = pd.to_datetime(settled["placed_timestamp"], utc=True, errors="coerce")
    settled["date"] = settled["settled_timestamp"].fillna(settled["placed_timestamp"])
    settled = settled.dropna(subset=["date"]).sort_values("date")
    if settled.empty:
        return pd.DataFrame(columns=["date", "bankroll"])
    settled["profit_loss"] = pd.to_numeric(settled["profit_loss"], errors="coerce").fillna(0.0)
    settled["bankroll"] = settings.starting_bankroll + settled["profit_loss"].cumsum()
    start = pd.DataFrame([{"date": settled["date"].min() - pd.Timedelta(days=1), "bankroll": settings.starting_bankroll}])
    return pd.concat([start, settled[["date", "bankroll"]]], ignore_index=True)


def _recent_month_profit(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["month", "profit_loss"])
    settled = ledger[~ledger["result"].apply(is_open_result)].copy()
    if settled.empty:
        return pd.DataFrame(columns=["month", "profit_loss"])
    settled["date"] = pd.to_datetime(settled["settled_timestamp"], utc=True, errors="coerce").fillna(
        pd.to_datetime(settled["placed_timestamp"], utc=True, errors="coerce")
    )
    settled = settled.dropna(subset=["date"])
    settled["month"] = settled["date"].dt.to_period("M").dt.to_timestamp()
    settled["profit_loss"] = pd.to_numeric(settled["profit_loss"], errors="coerce").fillna(0.0)
    return settled.groupby("month", as_index=False)["profit_loss"].sum().tail(14)


def _inject_css() -> None:
    st.html(
        """
        <style>
        .bankroll-title { color:#f5f7fb; font-size:1.95rem; line-height:1; font-weight:900; letter-spacing:-.04em; }
        .bankroll-subtitle { color:#dbe7f5; font-size:.98rem; margin-top:.35rem; }
        .bankroll-actions { color:#f5f7fb; font-size:.82rem; font-weight:700; text-align:right; margin-bottom:.35rem; }
        .bankroll-kpis { display:grid; grid-template-columns:repeat(7, minmax(0,1fr)); gap:.65rem; margin:.85rem 0 1rem; }
        .bankroll-card { background:linear-gradient(180deg, rgba(17,31,49,.94), rgba(12,24,39,.98)); border:1px solid rgba(43,60,82,.96); border-radius:7px; box-shadow:0 20px 40px rgba(0,0,0,.22); }
        .bankroll-kpi { min-height:84px; padding:.95rem .75rem .8rem; text-align:center; }
        .bankroll-kpi-label { color:#dbe7f5; text-transform:uppercase; font-size:.68rem; font-weight:850; }
        .bankroll-kpi-value { color:#f5f7fb; font-size:1.45rem; line-height:1.15; font-weight:900; margin-top:.35rem; }
        .bankroll-kpi-caption { color:#f5f7fb; font-size:.76rem; margin-top:.28rem; }
        .bankroll-green { color:#31df63 !important; } .bankroll-red { color:#ff4949 !important; } .bankroll-blue { color:#3b82f6 !important; } .bankroll-amber { color:#fb923c !important; }
        .bankroll-card-title { color:#f5f7fb; text-transform:uppercase; font-size:.82rem; font-weight:900; margin:1rem 1rem .4rem; }
        .bankroll-chart-wrap { padding:.35rem 1rem 1rem; }
        .bankroll-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.78rem; }
        .bankroll-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.66rem; padding:.55rem .42rem; border-bottom:1px solid rgba(43,60,82,.9); }
        .bankroll-table td { padding:.58rem .42rem; border-bottom:1px solid rgba(43,60,82,.68); }
        .bankroll-right { text-align:right !important; } .bankroll-center { text-align:center !important; }
        .bankroll-badge { display:inline-block; border-radius:5px; padding:.18rem .45rem; font-weight:800; font-size:.7rem; }
        .bankroll-badge.win { color:#31df63; background:rgba(49,223,99,.13); border:1px solid rgba(49,223,99,.3); }
        .bankroll-badge.loss { color:#ff4949; background:rgba(255,73,73,.13); border:1px solid rgba(255,73,73,.3); }
        .bankroll-badge.open { color:#facc15; background:rgba(250,204,21,.12); border:1px solid rgba(250,204,21,.28); }
        .bankroll-panel-grid { display:grid; grid-template-columns:1.18fr 1fr; gap:.7rem; margin-top:.7rem; }
        .bankroll-small-grid { display:grid; grid-template-columns:1.08fr 1fr 1.12fr 1.18fr 1fr; gap:.7rem; margin-top:.7rem; }
        .bankroll-risk-grid { display:grid; grid-template-columns:2.7fr .9fr; gap:.7rem; margin-top:.7rem; }
        .bankroll-settings { display:grid; grid-template-columns:repeat(7, minmax(0,1fr)); gap:.7rem; padding:.9rem 1rem 1rem; }
        .bankroll-setting { border-right:1px solid rgba(43,60,82,.75); min-height:48px; }
        .bankroll-setting:last-child { border-right:0; }
        .bankroll-setting-label { color:#dbe7f5; font-size:.68rem; }
        .bankroll-setting-value { color:#f5f7fb; font-size:1rem; margin-top:.25rem; }
        .bankroll-health { display:flex; gap:.9rem; align-items:center; padding:1rem; color:#f5f7fb; }
        .bankroll-health-icon { color:#31df63; font-size:2rem; }
        @media (max-width:1200px) { .bankroll-kpis, .bankroll-small-grid, .bankroll-settings { grid-template-columns:repeat(2, minmax(0,1fr)); } .bankroll-panel-grid, .bankroll-risk-grid { grid-template-columns:1fr; } }
        </style>
        """
    )


def _kpi(label: str, value: str, caption: str = "", color_class: str = "") -> str:
    return (
        '<div class="bankroll-card bankroll-kpi">'
        f'<div class="bankroll-kpi-label">{_escape(label)}</div>'
        f'<div class="bankroll-kpi-value {color_class}">{_escape(value)}</div>'
        f'<div class="bankroll-kpi-caption">{_escape(caption)}</div>'
        '</div>'
    )


def _render_header(ledger: pd.DataFrame) -> None:
    left, right = st.columns([1.8, 1])
    with left:
        st.html('<div class="bankroll-title">BANKROLL</div><div class="bankroll-subtitle">Track performance, manage risk, and grow your bankroll</div>')
    with right:
        st.markdown(f"<div class='bankroll-actions'>{_escape(_last_updated(ledger))}</div>", unsafe_allow_html=True)
        st.download_button(
            "⇩  Export Report",
            data=_ledger_csv(ledger),
            file_name="ufc_bet_ledger.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=ledger.empty,
        )


def _render_kpis(summary: dict, settings: BankrollSettings) -> None:
    total_profit = float(summary.get("total_profit", 0.0))
    roi = total_profit / settings.starting_bankroll if settings.starting_bankroll else 0.0
    cards = [
        _kpi("Starting Bankroll", _money(summary.get("starting_bankroll")), "Configured starting point"),
        _kpi("Current Bankroll", _money(summary.get("current_bankroll")), _pct(roi), "bankroll-green" if total_profit >= 0 else "bankroll-red"),
        _kpi("Total Profit / Loss", _signed_money(total_profit), "All-time", "bankroll-green" if total_profit >= 0 else "bankroll-red"),
        _kpi("Open Risk", _money(summary.get("open_risk")), f"{summary.get('open_bets', 0)} open bets", "bankroll-amber" if summary.get("open_risk", 0) else ""),
        _kpi("Available Bankroll", _money(summary.get("available_bankroll")), "For new bets", "bankroll-blue"),
        _kpi("Total Stake", _money(summary.get("total_stake")), "All-time"),
        _kpi("Win Rate", _pct(summary.get("win_rate")), f"{summary.get('wins', 0)}W - {summary.get('losses', 0)}L - {summary.get('pushes', 0)}P", "bankroll-green"),
    ]
    st.html('<div class="bankroll-kpis">' + ''.join(cards) + '</div>')


def _bankroll_figure(points: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["date"],
                y=points["bankroll"],
                mode="lines+markers",
                name="Bankroll",
                line=dict(color="#31df63", width=3),
                fill="tozeroy",
                fillcolor="rgba(49,223,99,.12)",
            )
        )
    fig.update_layout(height=300, margin=dict(l=35, r=10, t=10, b=25), yaxis_tickprefix="$", showlegend=False)
    return apply_plotly_theme(fig)


def _open_exposure_figure(open_bets: pd.DataFrame) -> go.Figure:
    total_stake = float(pd.to_numeric(open_bets.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not open_bets.empty else 0.0
    potential = float(open_bets.apply(_potential_profit, axis=1).sum()) if not open_bets.empty else 0.0
    values = [total_stake, potential]
    fig = go.Figure(data=[go.Pie(labels=["Open Stake", "Potential Profit"], values=values, hole=.62, marker=dict(colors=["#2f7de1", "#31df63"]), textinfo="none")])
    fig.update_layout(height=230, margin=dict(l=5, r=5, t=5, b=5), showlegend=False, annotations=[dict(text=f"<b>{len(open_bets)}</b><br>Open Bets", x=.5, y=.5, showarrow=False, font=dict(color="#f5f7fb", size=15))])
    return apply_plotly_theme(fig)


def _profit_by_event_figure(ledger: pd.DataFrame) -> go.Figure:
    perf = performance_by_event(ledger)
    fig = go.Figure()
    if not perf.empty:
        perf = perf.sort_values("profit_loss").tail(6)
        colors = ["#31df63" if value >= 0 else "#ff4949" for value in perf["profit_loss"]]
        fig.add_trace(go.Bar(x=perf["profit_loss"], y=perf["event_name"], orientation="h", marker_color=colors, text=[_signed_money(v) for v in perf["profit_loss"]], textposition="outside"))
    fig.update_layout(height=255, margin=dict(l=5, r=45, t=5, b=25), showlegend=False, xaxis_tickprefix="$")
    return apply_plotly_theme(fig)


def _monthly_profit_figure(ledger: pd.DataFrame) -> go.Figure:
    monthly = _recent_month_profit(ledger)
    fig = go.Figure()
    if not monthly.empty:
        colors = ["#31df63" if value >= 0 else "#ff4949" for value in monthly["profit_loss"]]
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["profit_loss"], marker_color=colors, name="Profit/Loss"))
    fig.update_layout(height=255, margin=dict(l=35, r=10, t=5, b=25), showlegend=False, yaxis_tickprefix="$")
    return apply_plotly_theme(fig)


def _roi_by_odds_figure(ledger: pd.DataFrame) -> go.Figure:
    settled = ledger[~ledger["result"].apply(is_open_result)].copy() if not ledger.empty else pd.DataFrame()
    buckets = ["Favorites", "Slight Dogs", "Medium Dogs", "Large Dogs"]
    values = [0, 0, 0, 0]
    if not settled.empty:
        odds = pd.to_numeric(settled["odds_taken"], errors="coerce")
        masks = [odds.between(-250, -110), odds.between(100, 200), odds.between(201, 400), odds >= 401]
        for idx, mask in enumerate(masks):
            subset = settled[mask]
            stake = pd.to_numeric(subset.get("stake"), errors="coerce").fillna(0).sum()
            profit = pd.to_numeric(subset.get("profit_loss"), errors="coerce").fillna(0).sum()
            values[idx] = max(profit / stake * 100, 0) if stake else 0
    fig = go.Figure(data=[go.Pie(labels=buckets, values=values, hole=.62, marker=dict(colors=["#31df63", "#2f7de1", "#a855f7", "#ff4949"]), textinfo="none")])
    fig.update_layout(height=255, margin=dict(l=5, r=5, t=5, b=5), showlegend=True, legend=dict(font=dict(color="#f5f7fb")), annotations=[dict(text="<b>ROI</b><br>By Odds", x=.5, y=.5, showarrow=False, font=dict(color="#f5f7fb", size=14))])
    return apply_plotly_theme(fig)


def _win_rate_figure(ledger: pd.DataFrame) -> go.Figure:
    settled = ledger[~ledger["result"].apply(is_open_result)].copy() if not ledger.empty else pd.DataFrame()
    labels = ["Win", "Loss", "Push/Void"]
    values = [0, 0, 0]
    if not settled.empty:
        result = settled["result"].astype(str).str.lower()
        values = [int((result == "win").sum()), int((result == "loss").sum()), int(result.isin(["push", "void"]).sum())]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.62, marker=dict(colors=["#31df63", "#ff4949", "#facc15"]), textinfo="none")])
    decisions = values[0] + values[1]
    win_rate = values[0] / decisions * 100 if decisions else 0
    fig.update_layout(height=255, margin=dict(l=5, r=5, t=5, b=5), showlegend=True, legend=dict(font=dict(color="#f5f7fb")), annotations=[dict(text=f"<b>{win_rate:.1f}%</b><br>Win Rate", x=.5, y=.5, showarrow=False, font=dict(color="#f5f7fb", size=15))])
    return apply_plotly_theme(fig)


def _status_badge(result: str) -> str:
    clean = str(result or "Open").strip().lower()
    css = "win" if clean == "win" else "loss" if clean == "loss" else "open"
    return f'<span class="bankroll-badge {css}">{_escape(str(result or "Open").title())}</span>'


def _ledger_table(ledger: pd.DataFrame, limit: int = 5) -> str:
    if ledger.empty:
        return "<table class='bankroll-table'><tbody><tr><td>No official bets have been added to the ledger yet.</td></tr></tbody></table>"
    display = ledger.copy().head(limit)
    rows = []
    for _, row in display.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_escape(pd.to_datetime(row.get('placed_timestamp'), errors='coerce').strftime('%b %-d, %Y') if pd.notna(pd.to_datetime(row.get('placed_timestamp'), errors='coerce')) else '—')}</td>"
            f"<td>{_escape(row.get('event_name'))}</td>"
            f"<td>{_escape(row.get('fighter'))} vs {_escape(row.get('opponent'))}</td>"
            f"<td>{_escape(row.get('fighter'))}</td>"
            f"<td>{_escape(row.get('market_type'))}</td>"
            f"<td class='bankroll-right'>{_american(row.get('odds_taken'))}</td>"
            f"<td class='bankroll-right'>{_money(row.get('stake'))}</td>"
            f"<td class='bankroll-center'>{_status_badge(row.get('result'))}</td>"
            f"<td class='bankroll-right {'bankroll-green' if _as_float(row.get('profit_loss')) >= 0 else 'bankroll-red'}'>{_signed_money(row.get('profit_loss'))}</td>"
            f"<td class='bankroll-right'>{_american(row.get('closing_odds'))}</td>"
            f"<td class='bankroll-right {'bankroll-green' if _as_float(row.get('clv')) >= 0 else 'bankroll-red'}'>{_pct(row.get('clv'), signed=True)}</td>"
            f"<td class='bankroll-right'>{_pct(row.get('model_probability'))}</td>"
            f"<td class='bankroll-right'>{_signed_money(_as_float(row.get('ev')) * _as_float(row.get('stake')) if abs(_as_float(row.get('ev'))) < 10 else _as_float(row.get('ev')))}</td>"
            "</tr>"
        )
    return (
        "<table class='bankroll-table'><thead><tr>"
        "<th>Date</th><th>Event</th><th>Fight</th><th>Pick</th><th>Bet Type</th><th class='bankroll-right'>Odds Taken</th><th class='bankroll-right'>Stake</th><th class='bankroll-center'>Result</th><th class='bankroll-right'>Profit / Loss</th><th class='bankroll-right'>Closing Odds</th><th class='bankroll-right'>CLV</th><th class='bankroll-right'>Model Prob</th><th class='bankroll-right'>EV</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table><div class='bankroll-kpi-caption' style='text-align:center; padding:.75rem;'>View Full Bet Ledger →</div>"
    )


def _exposure_table(open_bets: pd.DataFrame, by: str = "event") -> str:
    if by == "fighter":
        exposure = _open_exposure_by_fighter(open_bets)
        label_col = "fighter"
        label = "Fighter"
    else:
        exposure = exposure_by_event(open_bets)
        label_col = "event_name"
        label = "Event"
        if "open_risk" in exposure.columns:
            exposure = exposure.rename(columns={"open_risk": "stake"})
    if exposure.empty:
        return f"<table class='bankroll-table'><tbody><tr><td>No open exposure by {label.lower()}.</td></tr></tbody></table>"
    rows = []
    for _, row in exposure.head(5).iterrows():
        rows.append(f"<tr><td>{_escape(row.get(label_col))}</td><td class='bankroll-right'>{_money(row.get('stake'))}</td><td class='bankroll-right bankroll-green'>{_money(row.get('potential_profit'))}</td></tr>")
    return f"<table class='bankroll-table'><thead><tr><th>{label}</th><th class='bankroll-right'>Stake</th><th class='bankroll-right'>Potential Profit</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _render_open_exposure(open_bets: pd.DataFrame) -> None:
    total_stake = float(pd.to_numeric(open_bets.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not open_bets.empty else 0.0
    potential = float(open_bets.apply(_potential_profit, axis=1).sum()) if not open_bets.empty else 0.0
    left, middle, right = st.columns([.8, .85, 1.25])
    with left:
        st.plotly_chart(_open_exposure_figure(open_bets), use_container_width=True, config={"displayModeBar": False})
    with middle:
        st.html(
            f"<div class='bankroll-chart-wrap'><table class='bankroll-table'><tbody>"
            f"<tr><td>Total Stake Pending</td><td class='bankroll-right'>{_money(total_stake)}</td></tr>"
            f"<tr><td>Potential Profit</td><td class='bankroll-right bankroll-green'>{_money(potential)}</td></tr>"
            f"<tr><td>Worst Case Risk</td><td class='bankroll-right bankroll-red'>-{_money(total_stake)}</td></tr>"
            "</tbody></table></div>"
        )
    with right:
        st.html("<div class='bankroll-card-title' style='margin-top:0;'>Exposure Breakdown</div>" + _exposure_table(open_bets, "event"))


def _clv_summary_html(ledger: pd.DataFrame) -> str:
    clv = pd.to_numeric(ledger.get("clv", pd.Series(dtype=float)), errors="coerce").dropna() if not ledger.empty else pd.Series(dtype=float)
    if clv.empty:
        rows = [("Beat Closing Line %", "0.0%", "bankroll-green"), ("Average CLV", "+0.0%", "bankroll-green"), ("Median CLV", "+0.0%", "bankroll-green"), ("Profit w/ CLV > 0", "$0.00", "bankroll-green"), ("Profit w/ CLV < 0", "$0.00", "bankroll-red")]
    else:
        profit = pd.to_numeric(ledger.get("profit_loss"), errors="coerce").fillna(0)
        rows = [
            ("Beat Closing Line %", _pct((clv > 0).mean()), "bankroll-green"),
            ("Average CLV", _pct(clv.mean(), signed=True), "bankroll-green" if clv.mean() >= 0 else "bankroll-red"),
            ("Median CLV", _pct(clv.median(), signed=True), "bankroll-green" if clv.median() >= 0 else "bankroll-red"),
            ("Profit w/ CLV > 0", _money(profit[clv.reindex(profit.index, fill_value=False) > 0].sum()), "bankroll-green"),
            ("Profit w/ CLV < 0", _money(profit[clv.reindex(profit.index, fill_value=False) < 0].sum()), "bankroll-red"),
        ]
    body = "".join(f"<tr><td>{label}</td><td class='bankroll-right {css}'>{value}</td></tr>" for label, value, css in rows)
    return f"<table class='bankroll-table'><tbody>{body}</tbody></table>"


def _risk_settings_html(settings: BankrollSettings) -> str:
    items = [
        ("Kelly Fraction", f"{settings.kelly_fraction:.2f}"),
        ("Max Stake Per Bet", _pct(settings.max_stake_pct)),
        ("Max Event Exposure", _pct(settings.max_event_exposure_pct)),
        ("EV Threshold", _pct(settings.min_edge, signed=True)),
        ("Min Confidence", _pct(settings.min_confidence, already_pct=True)),
        ("Odds Range", f"{settings.min_odds} to +{settings.max_odds}"),
        ("Starting Bankroll", _money(settings.starting_bankroll)),
    ]
    return "<div class='bankroll-settings'>" + "".join(f"<div class='bankroll-setting'><div class='bankroll-setting-label'>{_escape(label)}</div><div class='bankroll-setting-value'>{_escape(value)}</div></div>" for label, value in items) + "</div>"


def _load_fight_options() -> pd.DataFrame:
    frames = []
    for path in [BETTING_BOARD_PATH, LIVE_CARD_PATH]:
        df = load_parquet(path)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True, sort=False)
    keep = [col for col in ["event_name", "event_date", "fight_id", "red_fighter", "blue_fighter", "best_side", "best_prob", "best_edge", "best_ev", "best_american_odds", "model_pick", "model_confidence"] if col in rows.columns]
    rows = rows[keep].dropna(subset=["fight_id"]).drop_duplicates(subset=["fight_id", "event_name"]).copy()
    rows["label"] = rows.apply(lambda row: f"{row.get('event_name', 'Unknown Event')} — {row.get('red_fighter', 'Red')} vs {row.get('blue_fighter', 'Blue')}", axis=1)
    return rows.sort_values(["event_name", "label"])


def _manual_bet_id(row: dict) -> str:
    raw = "|".join(str(row.get(key, "")) for key in ["event_name", "fight_id", "fighter", "odds_taken", "stake", "placed_timestamp"])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _append_manual_bet(row: dict) -> None:
    ledger = load_bet_ledger()
    row["bet_id"] = _manual_bet_id(row)
    updated = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    save_bet_ledger(normalize_ledger(updated))


def _auto_settlement(open_bet: pd.Series) -> tuple[str | None, str]:
    master = load_parquet(MASTER_PATH)
    if master is None or master.empty:
        return None, "Master results are not available yet."
    work = master.copy()
    fight_id = str(open_bet.get("fight_id", ""))
    if fight_id and "fight_id" in work.columns:
        work = work[work["fight_id"].astype(str) == fight_id]
    if work.empty:
        fighter = str(open_bet.get("fighter", "")).lower()
        opponent = str(open_bet.get("opponent", "")).lower()
        name_cols = [col for col in ["r_name", "b_name", "red_fighter", "blue_fighter"] if col in master.columns]
        if name_cols:
            mask = False
            for col in name_cols:
                values = master[col].astype(str).str.lower()
                mask = mask | values.isin([fighter, opponent]) if not isinstance(mask, bool) else values.isin([fighter, opponent])
            work = master[mask]
    if work.empty or "winner" not in work.columns:
        return None, "No completed fight result matched this open bet."
    row = work.iloc[-1]
    winner = str(row.get("winner", "")).strip().lower()
    fighter = str(open_bet.get("fighter", "")).strip().lower()
    if not winner:
        return None, "Matched fight does not have a winner recorded yet."
    return ("Win" if winner == fighter else "Loss"), f"Matched master winner: {row.get('winner')}"


def _open_add_bet_dialog():
    if not hasattr(st, "dialog"):
        st.info("Your Streamlit version does not support popup dialogs. Add-bet controls are shown inline below.")
        return _render_add_bet_form(in_dialog=False)

    @st.dialog("Add New Bet")
    def dialog():
        _render_add_bet_form(in_dialog=True)

    dialog()


def _render_add_bet_form(in_dialog: bool = False) -> None:
    options = _load_fight_options()
    if options.empty:
        st.warning("No current fight options are available from live card or Betting Board artifacts.")
        return
    selected = st.selectbox("Fight", options.to_dict("records"), format_func=lambda row: row["label"], key="bankroll_add_fight")
    fighters = [fighter for fighter in [selected.get("red_fighter"), selected.get("blue_fighter")] if pd.notna(fighter) and str(fighter).strip()]
    default_idx = fighters.index(selected.get("best_side")) if selected.get("best_side") in fighters else 0
    fighter = st.selectbox("Pick", fighters, index=default_idx, key="bankroll_add_pick")
    stake = st.number_input("Amount Staked", min_value=0.0, value=float(selected.get("recommended_stake", 0) or 0), step=25.0, key="bankroll_add_stake")
    default_odds = int(selected.get("best_american_odds") or 0) if pd.notna(selected.get("best_american_odds")) else 0
    odds = st.number_input("Odds Taken", min_value=-2000, max_value=3000, value=default_odds, step=5, key="bankroll_add_odds")
    submitted = st.button("Add Bet", use_container_width=True, key="bankroll_add_submit")
    if submitted:
        if stake <= 0 or odds == 0:
            st.error("Enter a positive stake and non-zero odds.")
            return
        opponent = next((name for name in fighters if name != fighter), "")
        row = {
            "event_name": selected.get("event_name", ""),
            "event_date": selected.get("event_date", ""),
            "fight_id": selected.get("fight_id", ""),
            "fighter": fighter,
            "opponent": opponent,
            "market_type": "Moneyline",
            "odds_taken": odds,
            "stake": stake,
            "result": "Open",
            "profit_loss": 0.0,
            "model_probability": selected.get("best_prob", selected.get("model_confidence")),
            "edge": selected.get("best_edge"),
            "ev": selected.get("best_ev"),
            "clv": pd.NA,
            "closing_odds": pd.NA,
            "bet_status": "MANUAL",
            "placed_timestamp": datetime.now(timezone.utc).isoformat(),
            "settled_timestamp": "",
            "source_workflow": "Bankroll Manual Entry",
            "source_prediction_run_id": "",
            "notes": "Manual bankroll entry",
        }
        _append_manual_bet(row)
        st.success("Bet added to the bankroll ledger.")
        st.cache_data.clear()
        if in_dialog:
            st.session_state["bankroll_dialog"] = None
        st.rerun()


def _open_settle_dialog(ledger: pd.DataFrame):
    if not hasattr(st, "dialog"):
        st.info("Your Streamlit version does not support popup dialogs. Settlement controls are shown inline below.")
        return _render_settle_form(ledger, in_dialog=False)

    @st.dialog("Settle Bet")
    def dialog():
        _render_settle_form(ledger, in_dialog=True)

    dialog()


def _render_settle_form(ledger: pd.DataFrame, in_dialog: bool = False) -> None:
    open_bets = derive_open_bets(ledger)
    if open_bets.empty:
        st.info("No open bets are available to settle.")
        return
    open_bets = open_bets.copy()
    open_bets["label"] = open_bets.apply(lambda row: f"{row.get('event_name', '')} — {row.get('fighter', '')} {_american(row.get('odds_taken'))} ({_money(row.get('stake'))})", axis=1)
    selected = st.selectbox("Open bet", open_bets.to_dict("records"), format_func=lambda row: row["label"], key="bankroll_settle_bet")
    if st.button("Scrape / Auto-Fill Result", use_container_width=True, key="bankroll_auto_settle"):
        result, message = _auto_settlement(pd.Series(selected))
        st.session_state["bankroll_auto_result"] = result or ""
        st.info(message)
    default_result = st.session_state.get("bankroll_auto_result") or "Win"
    result = st.selectbox("Result", RESULT_OPTIONS, index=RESULT_OPTIONS.index(default_result) if default_result in RESULT_OPTIONS else 0, key="bankroll_settle_result")
    closing_odds = st.number_input("Closing odds", min_value=-2000, max_value=3000, value=0, step=5, key="bankroll_settle_closing")
    clv = st.number_input("CLV", min_value=-10.0, max_value=10.0, value=0.0, step=0.01, format="%.3f", key="bankroll_settle_clv")
    notes = st.text_input("Settlement notes", value=str(selected.get("notes", "") or ""), key="bankroll_settle_notes")
    if st.button("Settle Bet", use_container_width=True, key="bankroll_settle_submit"):
        ok = settle_bet(selected["bet_id"], result=result, closing_odds=None if closing_odds == 0 else closing_odds, clv=clv, notes=notes)
        if ok:
            st.success("Bet settled and bankroll ledger updated.")
            st.cache_data.clear()
            if in_dialog:
                st.session_state["bankroll_dialog"] = None
            st.rerun()
        else:
            st.error("Could not find the selected bet in the ledger.")



def _open_risk_settings_dialog():
    settings = load_bankroll_settings()
    if not hasattr(st, "dialog"):
        st.info("Your Streamlit version does not support popup dialogs. Risk settings are shown inline below.")
        return _render_risk_settings_form(settings, in_dialog=False)

    @st.dialog("Risk Settings")
    def dialog():
        _render_risk_settings_form(settings, in_dialog=True)

    dialog()


def _render_risk_settings_form(settings: BankrollSettings, in_dialog: bool = False) -> None:
    st.caption("Update the bankroll risk defaults stored in the settings artifact, then refresh bankroll status through the existing pipeline workflow.")
    with st.form("bankroll_risk_settings_dialog_form"):
        starting_bankroll = st.number_input(
            "Starting bankroll",
            min_value=0.0,
            value=float(settings.starting_bankroll),
            step=100.0,
        )
        kelly_fraction = st.number_input(
            "Kelly fraction",
            min_value=0.0,
            max_value=2.0,
            value=float(settings.kelly_fraction),
            step=0.05,
        )
        max_stake_pct = st.number_input(
            "Max stake per bet (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(settings.max_stake_pct * 100),
            step=0.25,
        )
        max_event_exposure_pct = st.number_input(
            "Max event exposure (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(settings.max_event_exposure_pct * 100),
            step=0.5,
        )
        min_edge = st.number_input(
            "EV / edge threshold",
            min_value=-100.0,
            max_value=100.0,
            value=float(settings.min_edge),
            step=0.01,
        )
        min_confidence = st.number_input(
            "Minimum confidence (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(settings.min_confidence),
            step=1.0,
        )
        odds_col1, odds_col2 = st.columns(2)
        with odds_col1:
            min_odds = st.number_input("Minimum odds", min_value=-1000, max_value=1000, value=int(settings.min_odds), step=5)
        with odds_col2:
            max_odds = st.number_input("Maximum odds", min_value=-1000, max_value=3000, value=int(settings.max_odds), step=5)
        run_workflow = st.checkbox("Run bankroll status workflow after save", value=True)
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
        if run_workflow:
            ok, msg = trigger_workflow(BANKROLL_STATUS_WORKFLOW)
            if ok:
                st.success("Risk settings saved and bankroll status workflow launched.")
            else:
                st.warning(f"Risk settings saved, but the workflow could not be launched: {msg}")
        else:
            st.success("Risk settings saved.")
        st.cache_data.clear()
        if in_dialog:
            st.session_state["bankroll_dialog"] = None
        st.rerun()

def _handle_dialogs(ledger: pd.DataFrame) -> None:
    dialog = st.session_state.get("bankroll_dialog")
    if dialog == "add":
        _open_add_bet_dialog()
    elif dialog == "settle":
        _open_settle_dialog(ledger)
    elif dialog == "risk":
        _open_risk_settings_dialog()


def render_bankroll():
    _inject_css()
    settings = load_bankroll_settings()
    ledger = load_bet_ledger()
    summary = _ledger_summary(ledger, settings)
    open_bets = derive_open_bets(ledger)

    _handle_dialogs(ledger)
    _render_header(ledger)
    _render_kpis(summary, settings)

    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">Bankroll Over Time</div><div class="bankroll-chart-wrap">')
        st.plotly_chart(_bankroll_figure(_bankroll_points(ledger, settings)), use_container_width=True, config={"displayModeBar": False})
        st.html('</div></div>')
    with top_right:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">Open Exposure Summary</div><div class="bankroll-chart-wrap">')
        _render_open_exposure(open_bets)
        st.html('</div></div>')

    st.html('<div class="bankroll-card" style="margin-top:.7rem;"><div class="bankroll-card-title">Bet Ledger (All Time)</div><div class="bankroll-chart-wrap">' + _ledger_table(ledger) + '</div></div>')

    st.html('<div class="bankroll-small-grid">')
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">Performance Over Time</div><div class="bankroll-chart-wrap">')
        st.plotly_chart(_monthly_profit_figure(ledger), use_container_width=True, config={"displayModeBar": False})
        st.html('</div></div>')
    with c2:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">Profit By Event</div><div class="bankroll-chart-wrap">')
        st.plotly_chart(_profit_by_event_figure(ledger), use_container_width=True, config={"displayModeBar": False})
        st.html('</div></div>')
    with c3:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">ROI By Odds Range</div><div class="bankroll-chart-wrap">')
        st.plotly_chart(_roi_by_odds_figure(ledger), use_container_width=True, config={"displayModeBar": False})
        st.html('</div></div>')
    with c4:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">Win Rate By Confidence</div><div class="bankroll-chart-wrap">')
        st.plotly_chart(_win_rate_figure(ledger), use_container_width=True, config={"displayModeBar": False})
        st.html('</div></div>')
    with c5:
        st.html('<div class="bankroll-card"><div class="bankroll-card-title">CLV Performance</div><div class="bankroll-chart-wrap">' + _clv_summary_html(ledger) + '</div></div>')
    st.html('</div>')

    bottom_left, bottom_right = st.columns([2.7, .9])
    with bottom_left:
        st.html('<div class="bankroll-card" style="margin-top:.7rem;"><div class="bankroll-card-title">Risk Settings Summary</div>' + _risk_settings_html(settings) + '</div>')
    with bottom_right:
        health = "Excellent" if summary.get("available_bankroll", 0) >= 0 else "At Risk"
        detail = "You are managing risk well and your bankroll is available for new bets." if health == "Excellent" else "Open risk exceeds current bankroll."
        st.html(f'<div class="bankroll-card" style="margin-top:.7rem;"><div class="bankroll-card-title">Bankroll Health</div><div class="bankroll-health"><div class="bankroll-health-icon">🛡</div><div><strong>{health}</strong><br><span class="bankroll-kpi-caption">{detail}</span></div></div></div>')
