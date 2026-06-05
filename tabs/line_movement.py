from __future__ import annotations

import html
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.common.paths import (
    BET_LEDGER_PATH,
    CLOSING_LINES_PATH,
    CLV_RESULTS_PATH,
    LINE_MOVEMENT_PATH,
    MARKET_SNAPSHOTS_PATH,
    NORMALIZED_MARKET_SNAPSHOTS_PATH,
)
from utils.data_loader import load_parquet
from utils.github_actions import trigger_workflow
from utils.ui.charts import apply_plotly_theme

MARKET_WORKFLOW = "run-market-update.yml"
CLV_WORKFLOW = "run-clv-tracker.yml"
CENTRAL_TZ = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class ClvSummary:
    total_bets: int
    beat_pct: float
    avg_clv: float
    median_clv: float
    positive_pct: float
    negative_pct: float
    total_clv: float


# -----------------------------------------------------------------------------
# Formatting and safe data helpers
# -----------------------------------------------------------------------------


def _escape(value) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def _as_float(value, default=None):
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def _fmt_pct(value, decimals: int = 1, signed: bool = False) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "—"
    prefix = "+" if signed and parsed > 0 else ""
    return f"{prefix}{parsed:.{decimals}f}%"


def _fmt_money(value, decimals: int = 0, signed: bool = False) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "$0"
    sign = "+" if signed and parsed > 0 else "-" if parsed < 0 else ""
    return f"{sign}${abs(parsed):,.{decimals}f}"


def _fmt_int(value) -> str:
    parsed = _as_float(value, 0)
    return f"{int(round(parsed)):,}"


def _fmt_american(value) -> str:
    parsed = _as_float(value)
    if parsed is None or parsed == 0:
        return "—"
    rounded = int(round(parsed))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def _fmt_date(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "—"
    return parsed.strftime("%b %-d, %Y")


def _read(path) -> pd.DataFrame:
    data = load_parquet(path)
    return pd.DataFrame() if data is None else data.copy()


def _prepare_clv_results(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows = raw.copy()
    for column in [
        "bet_id",
        "event_name",
        "fight_id",
        "fighter",
        "opponent",
        "market_type",
        "sportsbook",
        "odds_taken",
        "closing_odds",
        "clv_pct",
        "beat_closing_line",
        "stake",
        "result",
        "profit_loss",
        "roi",
        "model_probability",
        "confidence_tier",
        "odds_bucket",
        "placed_timestamp",
        "closing_timestamp",
        "closing_line_status",
    ]:
        if column not in rows.columns:
            rows[column] = pd.NA

    rows["sportsbook"] = rows["sportsbook"].replace("", pd.NA).fillna("Unknown Book")
    rows["market_type"] = rows["market_type"].replace("", pd.NA).fillna("Moneyline")
    rows["event_name"] = rows["event_name"].replace("", pd.NA).fillna("Unknown Event")
    rows["fighter"] = rows["fighter"].replace("", pd.NA).fillna("Unknown Pick")
    rows["opponent"] = rows["opponent"].replace("", pd.NA).fillna("Unknown Opponent")
    rows["fight"] = rows["fighter"].astype(str) + " vs " + rows["opponent"].astype(str)

    rows["odds_taken"] = pd.to_numeric(rows["odds_taken"], errors="coerce")
    rows["closing_odds"] = pd.to_numeric(rows["closing_odds"], errors="coerce")
    rows["clv_pct"] = pd.to_numeric(rows["clv_pct"], errors="coerce")
    rows["stake"] = pd.to_numeric(rows["stake"], errors="coerce").fillna(0.0)
    rows["profit_loss"] = pd.to_numeric(rows["profit_loss"], errors="coerce").fillna(0.0)
    rows["roi"] = pd.to_numeric(rows["roi"], errors="coerce")
    rows["model_probability"] = pd.to_numeric(rows["model_probability"], errors="coerce")
    rows["placed_timestamp"] = pd.to_datetime(rows["placed_timestamp"], utc=True, errors="coerce")
    rows["closing_timestamp"] = pd.to_datetime(rows["closing_timestamp"], utc=True, errors="coerce")
    rows["event_date"] = pd.to_datetime(rows.get("event_date"), utc=True, errors="coerce")
    rows["display_date"] = rows["placed_timestamp"].fillna(rows["event_date"]).fillna(rows["closing_timestamp"])
    beat_raw = rows["beat_closing_line"]
    if not pd.api.types.is_bool_dtype(beat_raw):
        beat_raw = beat_raw.astype(str).str.lower().map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    rows["beat_closing_line"] = beat_raw.fillna(rows["clv_pct"] >= 0).astype(bool)
    return rows


def _prepare_normalized(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows = raw.copy()
    for column in ["snapshot_timestamp", "commence_time"]:
        if column in rows.columns:
            rows[column] = pd.to_datetime(rows[column], utc=True, errors="coerce")
    for column in ["american_odds", "implied_prob"]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in ["sportsbook", "market_type", "event_name", "fighter_name", "opponent_name"]:
        if column not in rows.columns:
            rows[column] = pd.NA
    rows["sportsbook"] = rows["sportsbook"].replace("", pd.NA).fillna("Unknown Book")
    rows["market_type"] = rows["market_type"].replace("", pd.NA).fillna("Moneyline")
    rows["event_name"] = rows["event_name"].replace("", pd.NA).fillna("Unknown Event")
    return rows.dropna(subset=["snapshot_timestamp"]).copy()


def _prepare_line_movement(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows = raw.copy()
    for column in ["opening_timestamp", "latest_timestamp"]:
        if column in rows.columns:
            rows[column] = pd.to_datetime(rows[column], utc=True, errors="coerce")
    for column in ["opening_odds", "latest_odds", "odds_movement", "implied_prob_movement"]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _latest_artifact_timestamp(*frames: pd.DataFrame) -> str:
    candidates = []
    for frame in frames:
        if frame.empty:
            continue
        for column in ["placed_timestamp", "closing_timestamp", "latest_timestamp", "snapshot_timestamp", "opening_timestamp"]:
            if column in frame.columns:
                parsed = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
                if not parsed.empty:
                    candidates.append(parsed.max())
    if not candidates:
        return "Last Updated: No CLV artifacts yet"
    latest = max(candidates).tz_convert(CENTRAL_TZ)
    return f"Last Updated: {latest.strftime('%b %-d, %Y %I:%M %p %Z')}"


# -----------------------------------------------------------------------------
# Filters and summaries
# -----------------------------------------------------------------------------


def _date_bounds(rows: pd.DataFrame) -> tuple | None:
    if rows.empty:
        return None
    dates = pd.to_datetime(rows.get("display_date"), errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().date(), dates.max().date()


def _options(rows: pd.DataFrame, column: str, all_label: str) -> list[str]:
    if rows.empty or column not in rows.columns:
        return [all_label]
    values = sorted({str(value).strip() for value in rows[column].dropna() if str(value).strip()})
    return [all_label, *values]


def _reset_filter_state() -> None:
    for key in [
        "clv_filter_event",
        "clv_filter_market",
        "clv_filter_book",
        "clv_filter_odds_range",
        "clv_filter_clv_range",
        "clv_filter_date_range",
    ]:
        st.session_state.pop(key, None)


def _render_filters(rows: pd.DataFrame) -> pd.DataFrame:
    events = _options(rows, "event_name", "All Events")
    markets = _options(rows, "market_type", "All Bet Types")
    books = _options(rows, "sportsbook", "All Sportsbooks")
    bounds = _date_bounds(rows)

    with st.container():
        st.html('<div class="clv-filter-shell"><div class="clv-filter-title">CLV Filters</div></div>')
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.15, 1.25, 1.05, 1.15, 1.1, 1.1, .72])
        with c1:
            if bounds:
                date_range = st.date_input("Date Range", value=st.session_state.get("clv_filter_date_range", bounds), key="clv_filter_date_range")
            else:
                date_range = None
                st.caption("Date Range: no CLV dates")
        with c2:
            event = st.selectbox("Event", events, key="clv_filter_event")
        with c3:
            market = st.selectbox("Bet Type", markets, key="clv_filter_market")
        with c4:
            book = st.selectbox("Sportsbook", books, key="clv_filter_book")
        with c5:
            odds_range = st.slider("Odds Range", min_value=-500, max_value=500, value=st.session_state.get("clv_filter_odds_range", (-250, 400)), step=10, key="clv_filter_odds_range")
        with c6:
            clv_range = st.slider("CLV Range", min_value=-25, max_value=25, value=st.session_state.get("clv_filter_clv_range", (-25, 25)), step=1, key="clv_filter_clv_range")
        with c7:
            st.caption("Actions")
            if st.button("Reset", use_container_width=True):
                _reset_filter_state()
                st.rerun()

    filtered = rows.copy()
    if filtered.empty:
        return filtered
    if event != "All Events":
        filtered = filtered[filtered["event_name"].astype(str) == event]
    if market != "All Bet Types":
        filtered = filtered[filtered["market_type"].astype(str) == market]
    if book != "All Sportsbooks":
        filtered = filtered[filtered["sportsbook"].astype(str) == book]
    if odds_range:
        odds = pd.to_numeric(filtered["odds_taken"], errors="coerce")
        filtered = filtered[odds.between(odds_range[0], odds_range[1]) | odds.isna()]
    if clv_range:
        clv = pd.to_numeric(filtered["clv_pct"], errors="coerce")
        filtered = filtered[clv.between(clv_range[0], clv_range[1]) | clv.isna()]
    if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        dates = pd.to_datetime(filtered["display_date"], utc=True, errors="coerce").dt.tz_localize(None)
        filtered = filtered[dates.between(start, end) | dates.isna()]
    return filtered.reset_index(drop=True)


def _summary(rows: pd.DataFrame) -> ClvSummary:
    if rows.empty:
        return ClvSummary(0, 0, 0, 0, 0, 0, 0)
    clv = pd.to_numeric(rows.get("clv_pct"), errors="coerce").dropna()
    total_bets = len(rows)
    if clv.empty:
        return ClvSummary(total_bets, 0, 0, 0, 0, 0, 0)
    positive = int((clv >= 0).sum())
    negative = int((clv < 0).sum())
    return ClvSummary(
        total_bets=total_bets,
        beat_pct=positive / len(clv) * 100,
        avg_clv=float(clv.mean()),
        median_clv=float(clv.median()),
        positive_pct=positive / len(clv) * 100,
        negative_pct=negative / len(clv) * 100,
        total_clv=float(clv.sum()),
    )


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------


def _inject_css() -> None:
    st.html(
        """
        <style>
        .clv-hero { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin:.1rem 0 1rem; }
        .clv-title { color:#f5f7fb; font-size:2.05rem; line-height:1; font-weight:950; letter-spacing:-.045em; text-transform:uppercase; }
        .clv-subtitle { color:#dbe7f5; font-size:1rem; margin-top:.35rem; }
        .clv-actions { display:flex; align-items:center; justify-content:flex-end; gap:.65rem; color:#f5f7fb; font-size:.82rem; font-weight:800; }
        .clv-kpi-grid { display:grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap:.65rem; margin:.8rem 0 1rem; }
        .clv-card { background:linear-gradient(180deg, rgba(17,31,49,.95), rgba(10,20,34,.98)); border:1px solid rgba(43,60,82,.96); border-radius:10px; box-shadow:0 20px 42px rgba(0,0,0,.25); }
        .clv-kpi { min-height:92px; padding:1rem .7rem .85rem; text-align:center; }
        .clv-kpi-label { color:#dbe7f5; text-transform:uppercase; font-size:.68rem; font-weight:900; letter-spacing:.035em; }
        .clv-kpi-value { color:#f5f7fb; font-size:1.65rem; line-height:1.15; font-weight:950; margin-top:.35rem; }
        .clv-kpi-caption { color:#99a8bc; font-size:.74rem; margin-top:.25rem; }
        .clv-green { color:#31df63 !important; } .clv-red { color:#ff5b5b !important; } .clv-blue { color:#60a5fa !important; }
        .clv-filter-shell { background:rgba(17,31,49,.55); border:1px solid rgba(43,60,82,.75); border-radius:10px; padding:.7rem 1rem .25rem; margin:.25rem 0 .45rem; }
        .clv-filter-title { color:#f5f7fb; text-transform:uppercase; font-weight:900; font-size:.78rem; letter-spacing:.04em; }
        .clv-grid-two { display:grid; grid-template-columns:1.18fr 1fr; gap:.75rem; margin-top:.75rem; }
        .clv-grid-three { display:grid; grid-template-columns:1fr 1fr 1fr; gap:.75rem; margin-top:.75rem; }
        .clv-section-title { color:#f5f7fb; text-transform:uppercase; font-size:.82rem; font-weight:950; margin:1rem 1rem .18rem; }
        .clv-section-subtitle { color:#99a8bc; font-size:.76rem; margin:0 1rem .45rem; }
        .clv-body { padding:.45rem 1rem 1rem; }
        .clv-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.78rem; }
        .clv-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.65rem; letter-spacing:.035em; padding:.55rem .42rem; border-bottom:1px solid rgba(43,60,82,.9); }
        .clv-table td { padding:.62rem .42rem; border-bottom:1px solid rgba(43,60,82,.65); vertical-align:middle; }
        .clv-right { text-align:right !important; } .clv-center { text-align:center !important; }
        .clv-badge { display:inline-block; padding:.16rem .42rem; border-radius:999px; font-size:.68rem; font-weight:900; text-transform:uppercase; }
        .clv-badge-good { color:#082a13; background:#31df63; } .clv-badge-bad { color:#330909; background:#ff7777; } .clv-badge-neutral { color:#dbe7f5; background:rgba(148,163,184,.22); }
        .clv-detail-grid { display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:.55rem; margin-bottom:.65rem; }
        .clv-detail-tile { border:1px solid rgba(43,60,82,.8); border-radius:8px; padding:.65rem; background:rgba(10,20,34,.7); }
        .clv-detail-label { color:#99a8bc; font-size:.66rem; font-weight:900; text-transform:uppercase; }
        .clv-detail-value { color:#f5f7fb; font-size:1rem; font-weight:950; margin-top:.22rem; }
        .clv-info { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-top:.9rem; padding:.85rem 1rem; color:#dbeafe; background:rgba(37,99,235,.14); border:1px solid rgba(96,165,250,.32); border-radius:10px; font-size:.82rem; }
        .clv-empty { color:#aeb8c6; padding:.85rem 0; font-size:.84rem; }
        @media (max-width: 1100px) { .clv-kpi-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } .clv-grid-two, .clv-grid-three { grid-template-columns:1fr; } .clv-detail-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } }
        </style>
        """
    )


# -----------------------------------------------------------------------------
# Header and KPI rendering
# -----------------------------------------------------------------------------


def _render_header(last_updated: str) -> None:
    left, right = st.columns([1.6, 1], vertical_alignment="center")
    with left:
        st.html(
            "<div class='clv-hero'><div>"
            "<div class='clv-title'>Line Movement / CLV</div>"
            "<div class='clv-subtitle'>Track closing line value performance and market movement.</div>"
            "</div></div>"
        )
    with right:
        action_cols = st.columns([.62, .38], vertical_alignment="center")
        with action_cols[0]:
            st.html(f"<div class='clv-actions'>{_escape(last_updated)}</div>")
        with action_cols[1]:
            if st.button("↻ Refresh Data", type="primary", use_container_width=True):
                ok_market, msg_market = trigger_workflow(MARKET_WORKFLOW)
                ok_clv, msg_clv = trigger_workflow(CLV_WORKFLOW)
                if ok_market and ok_clv:
                    st.toast("Market + CLV workflows launched.", icon="↻")
                else:
                    st.warning(f"Unable to launch refresh workflows: {msg_market}; {msg_clv}")


def _kpi(label: str, value: str, caption: str = "", color_class: str = "") -> str:
    return (
        "<div class='clv-card clv-kpi'>"
        f"<div class='clv-kpi-label'>{_escape(label)}</div>"
        f"<div class='clv-kpi-value {color_class}'>{_escape(value)}</div>"
        f"<div class='clv-kpi-caption'>{_escape(caption)}</div>"
        "</div>"
    )


def _render_kpis(summary: ClvSummary) -> None:
    cards = [
        _kpi("Total Bets Official", _fmt_int(summary.total_bets), "Logged bankroll bets"),
        _kpi("Beat Closing Line %", _fmt_pct(summary.beat_pct), "Positive CLV rate", "clv-green" if summary.beat_pct >= 50 else "clv-red"),
        _kpi("Average CLV", _fmt_pct(summary.avg_clv, signed=True), "Mean edge captured", "clv-green" if summary.avg_clv >= 0 else "clv-red"),
        _kpi("Median CLV", _fmt_pct(summary.median_clv, signed=True), "Middle result", "clv-green" if summary.median_clv >= 0 else "clv-red"),
        _kpi("Positive CLV %", _fmt_pct(summary.positive_pct), "Bets above close", "clv-green"),
        _kpi("Negative CLV %", _fmt_pct(summary.negative_pct), "Bets below close", "clv-red" if summary.negative_pct else ""),
        _kpi("Total CLV", _fmt_pct(summary.total_clv, signed=True), "Cumulative CLV", "clv-green" if summary.total_clv >= 0 else "clv-red"),
    ]
    st.html("<div class='clv-kpi-grid'>" + "".join(cards) + "</div>")


# -----------------------------------------------------------------------------
# Tables and charts
# -----------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]], empty_text: str) -> str:
    if rows:
        body = "".join("<tr>" + "".join(cells) + "</tr>" for cells in rows)
    else:
        body = f"<tr><td colspan='{len(headers)}'><div class='clv-empty'>{_escape(empty_text)}</div></td></tr>"
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    return f"<table class='clv-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _results_table(rows: pd.DataFrame) -> str:
    table_rows = []
    if not rows.empty:
        recent = rows.sort_values("display_date", ascending=False, na_position="last").head(12)
        for _, row in recent.iterrows():
            clv = _as_float(row.get("clv_pct"), 0)
            result = str(row.get("result") or "Open")
            badge_class = "clv-badge-good" if clv >= 0 else "clv-badge-bad"
            table_rows.append(
                [
                    f"<td>{_escape(_fmt_date(row.get('display_date')))}</td>",
                    f"<td>{_escape(row.get('event_name'))}</td>",
                    f"<td>{_escape(row.get('fighter'))}<br><span style='color:#99a8bc'>vs {_escape(row.get('opponent'))}</span></td>",
                    f"<td>{_escape(row.get('market_type'))}</td>",
                    f"<td>{_escape(row.get('sportsbook'))}</td>",
                    f"<td class='clv-right'>{_fmt_american(row.get('odds_taken'))}</td>",
                    f"<td class='clv-right'>{_fmt_american(row.get('closing_odds'))}</td>",
                    f"<td class='clv-right {('clv-green' if clv >= 0 else 'clv-red')}'>{_fmt_pct(row.get('clv_pct'), signed=True)}</td>",
                    f"<td><span class='clv-badge {badge_class}'>{'Beat' if clv >= 0 else 'Missed'}</span></td>",
                    f"<td>{_escape(result)}</td>",
                ]
            )
    return _table(
        ["Date", "Event", "Pick", "Type", "Book", "Taken", "Close", "CLV", "Line", "Result"],
        table_rows,
        "No official bet CLV results yet. Log bets in Bankroll, run market snapshots, then run the CLV tracker.",
    )


def _group_table(rows: pd.DataFrame, group_col: str, label: str) -> str:
    table_rows = []
    if not rows.empty and group_col in rows.columns:
        for value, subset in rows.groupby(group_col, dropna=False):
            clv = pd.to_numeric(subset.get("clv_pct"), errors="coerce").dropna()
            beat = (clv >= 0).mean() * 100 if len(clv) else 0
            avg = clv.mean() if len(clv) else 0
            profit = pd.to_numeric(subset.get("profit_loss"), errors="coerce").fillna(0).sum()
            table_rows.append(
                [
                    f"<td>{_escape(value or 'Unclassified')}</td>",
                    f"<td class='clv-center'>{len(subset):,}</td>",
                    f"<td class='clv-right'>{_fmt_pct(beat)}</td>",
                    f"<td class='clv-right {('clv-green' if avg >= 0 else 'clv-red')}'>{_fmt_pct(avg, signed=True)}</td>",
                    f"<td class='clv-right {('clv-green' if profit >= 0 else 'clv-red')}'>{_fmt_money(profit, signed=True)}</td>",
                ]
            )
    return _table([label, "Bets", "Beat CL%", "Avg CLV", "P/L"], table_rows, f"No {label.lower()} rows available.")


def _avg_clv_over_time(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not rows.empty:
        work = rows.dropna(subset=["display_date"]).copy()
        work["date"] = pd.to_datetime(work["display_date"], utc=True, errors="coerce").dt.date
        grouped = work.groupby("date", dropna=False)["clv_pct"].mean().reset_index()
        fig.add_trace(go.Scatter(x=grouped["date"], y=grouped["clv_pct"], mode="lines+markers", line=dict(color="#31df63", width=3), fill="tozeroy", fillcolor="rgba(49,223,99,.12)", name="Avg CLV"))
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(height=315, margin=dict(l=30, r=20, t=10, b=35), yaxis_title="Average CLV %", xaxis_title="Date")
    return apply_plotly_theme(fig)


def _profitability_figure(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    buckets = ["<-5%", "-5% to 0%", "0% to +5%", "+5%+"]
    if not rows.empty:
        clv = pd.to_numeric(rows.get("clv_pct"), errors="coerce")
        work = rows.copy()
        work["clv_profit_bucket"] = pd.cut(clv, bins=[-999, -5, 0, 5, 999], labels=buckets, include_lowest=True)
        grouped = work.groupby("clv_profit_bucket", observed=False)["profit_loss"].sum().reindex(buckets).fillna(0)
        colors = ["#ff5b5b" if value < 0 else "#31df63" for value in grouped]
        fig.add_trace(go.Bar(x=list(grouped.index), y=grouped.values, marker_color=colors, name="Profit/Loss"))
    fig.update_layout(height=315, margin=dict(l=35, r=15, t=10, b=35), yaxis_title="Profit / Loss", xaxis_title="CLV Bucket")
    return apply_plotly_theme(fig)


def _line_history_figure(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not rows.empty:
        for fighter, subset in rows.groupby("fighter_name", dropna=False):
            subset = subset.sort_values("snapshot_timestamp")
            fig.add_trace(go.Scatter(x=subset["snapshot_timestamp"], y=subset["american_odds"], mode="lines+markers", name=str(fighter or "Unknown")))
    fig.update_layout(height=330, margin=dict(l=35, r=15, t=10, b=35), yaxis_title="American Odds", legend=dict(orientation="h", y=-0.18))
    return apply_plotly_theme(fig)


def _line_detail_rows(normalized: pd.DataFrame, movement: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    if normalized.empty:
        return pd.DataFrame(), None
    normalized = normalized.copy()
    normalized["line_label"] = (
        normalized["event_name"].astype(str)
        + " — "
        + normalized["fighter_name"].astype(str)
        + " vs "
        + normalized["opponent_name"].astype(str)
        + " — "
        + normalized["sportsbook"].astype(str)
    )
    options = normalized.sort_values("snapshot_timestamp").drop_duplicates(["fight_id", "fighter_id", "sportsbook", "market_type"], keep="last")
    selected = st.selectbox("Line Movement Detail", options.to_dict("records"), format_func=lambda row: row.get("line_label", "Unknown line"), key="clv_line_detail_select")
    mask = (
        (normalized["fight_id"].astype(str) == str(selected.get("fight_id")))
        & (normalized["sportsbook"].astype(str) == str(selected.get("sportsbook")))
        & (normalized["market_type"].astype(str) == str(selected.get("market_type")))
    )
    # Include both sides of the fight for the chart.
    rows = normalized[mask].copy()
    selected_summary = None
    if movement is not None and not movement.empty:
        movement_mask = (
            (movement.get("fight_id", pd.Series(dtype=str)).astype(str) == str(selected.get("fight_id")))
            & (movement.get("fighter_id", pd.Series(dtype=str)).astype(str) == str(selected.get("fighter_id")))
            & (movement.get("sportsbook", pd.Series(dtype=str)).astype(str) == str(selected.get("sportsbook")))
            & (movement.get("market_type", pd.Series(dtype=str)).astype(str) == str(selected.get("market_type")))
        )
        if movement_mask.any():
            selected_summary = movement[movement_mask].iloc[0]
    return rows, selected_summary


def _detail_tiles(rows: pd.DataFrame, movement_row: pd.Series | None, clv_rows: pd.DataFrame) -> str:
    opening = latest = movement = "—"
    fighter = opponent = result = "—"
    if movement_row is not None:
        opening = _fmt_american(movement_row.get("opening_odds"))
        latest = _fmt_american(movement_row.get("latest_odds"))
        movement = _fmt_american(movement_row.get("odds_movement"))
        fighter = movement_row.get("fighter_name", "—")
        opponent = movement_row.get("opponent_name", "—")
    elif not rows.empty:
        selected_side = rows.sort_values("snapshot_timestamp").iloc[-1]
        fighter = selected_side.get("fighter_name", "—")
        opponent = selected_side.get("opponent_name", "—")
        side_rows = rows[rows["fighter_name"].astype(str) == str(fighter)].sort_values("snapshot_timestamp")
        if not side_rows.empty:
            opening = _fmt_american(side_rows.iloc[0].get("american_odds"))
            latest = _fmt_american(side_rows.iloc[-1].get("american_odds"))
            movement_value = _as_float(side_rows.iloc[-1].get("american_odds"), 0) - _as_float(side_rows.iloc[0].get("american_odds"), 0)
            movement = _fmt_american(movement_value)

    matching = pd.DataFrame()
    if not clv_rows.empty and not rows.empty:
        fight_id = rows.iloc[0].get("fight_id")
        matching = clv_rows[clv_rows["fight_id"].astype(str) == str(fight_id)].head(1)
    clv = _fmt_pct(matching.iloc[0].get("clv_pct"), signed=True) if not matching.empty else "—"
    result = str(matching.iloc[0].get("result") or "Open") if not matching.empty else "—"

    tiles = [
        ("Selected Side", fighter),
        ("Opponent", opponent),
        ("Opening Odds", opening),
        ("Current Odds", latest),
        ("Move", movement),
        ("CLV / Result", f"{clv} · {result}"),
    ]
    return "<div class='clv-detail-grid'>" + "".join(
        f"<div class='clv-detail-tile'><div class='clv-detail-label'>{_escape(label)}</div><div class='clv-detail-value'>{_escape(value)}</div></div>"
        for label, value in tiles
    ) + "</div>"


# -----------------------------------------------------------------------------
# Main render
# -----------------------------------------------------------------------------


def render_line_movement():
    _inject_css()
    clv_results = _prepare_clv_results(_read(CLV_RESULTS_PATH))
    normalized = _prepare_normalized(_read(NORMALIZED_MARKET_SNAPSHOTS_PATH))
    movement = _prepare_line_movement(_read(LINE_MOVEMENT_PATH))
    closing_lines = _read(CLOSING_LINES_PATH)
    ledger = _read(BET_LEDGER_PATH)
    market_snapshots = _read(MARKET_SNAPSHOTS_PATH)

    last_updated = _latest_artifact_timestamp(clv_results, movement, normalized, closing_lines, market_snapshots)
    _render_header(last_updated)

    filtered = _render_filters(clv_results)
    summary = _summary(filtered)
    _render_kpis(summary)

    top_left, top_right = st.columns([1.25, 1], gap="medium")
    with top_left:
        st.html("<div class='clv-card'><div class='clv-section-title'>CLV Results (Official Bets)</div><div class='clv-section-subtitle'>One row per manually logged bankroll bet matched to closing line data.</div><div class='clv-body'>" + _results_table(filtered) + "</div></div>")
    with top_right:
        st.html("<div class='clv-card'><div class='clv-section-title'>Average CLV Over Time</div><div class='clv-section-subtitle'>Average closing-line value by logged bet date.</div><div class='clv-body'>")
        st.plotly_chart(_avg_clv_over_time(filtered), use_container_width=True, config={"displayModeBar": False})
        st.html("</div></div>")

    st.html("<div class='clv-grid-three'>")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.html("<div class='clv-card'><div class='clv-section-title'>CLV by Odds Bucket</div><div class='clv-body'>" + _group_table(filtered, "odds_bucket", "Odds Bucket") + "</div></div>")
    with c2:
        st.html("<div class='clv-card'><div class='clv-section-title'>CLV by Confidence Tier</div><div class='clv-body'>" + _group_table(filtered, "confidence_tier", "Confidence Tier") + "</div></div>")
    with c3:
        st.html("<div class='clv-card'><div class='clv-section-title'>CLV by Sportsbook</div><div class='clv-body'>" + _group_table(filtered, "sportsbook", "Sportsbook") + "</div></div>")
    st.html("</div>")

    bottom_left, bottom_right = st.columns([1.25, 1], gap="medium")
    with bottom_left:
        st.html("<div class='clv-card' style='margin-top:.75rem;'><div class='clv-section-title'>Line Movement Detail</div><div class='clv-section-subtitle'>Opening, latest, and movement detail from normalized market snapshots.</div><div class='clv-body'>")
        if normalized.empty:
            st.html("<div class='clv-empty'>No normalized market snapshots are available yet. Run the CLV tracker after market snapshots are available.</div>")
        else:
            detail_rows, movement_row = _line_detail_rows(normalized, movement)
            st.html(_detail_tiles(detail_rows, movement_row, filtered))
            st.plotly_chart(_line_history_figure(detail_rows), use_container_width=True, config={"displayModeBar": False})
        st.html("</div></div>")
    with bottom_right:
        st.html("<div class='clv-card' style='margin-top:.75rem;'><div class='clv-section-title'>CLV vs Profitability</div><div class='clv-section-subtitle'>Profit and loss grouped by CLV bucket.</div><div class='clv-body'>")
        st.plotly_chart(_profitability_figure(filtered), use_container_width=True, config={"displayModeBar": False})
        st.html("</div></div>")

    artifact_note = (
        f"Official bet rows: {len(clv_results):,} · Ledger rows: {len(ledger):,} · "
        f"Normalized market rows: {len(normalized):,} · Closing-line rows: {len(closing_lines):,}"
    )
    st.html(
        "<div class='clv-info'>"
        "<span><strong>How CLV is calculated:</strong> CLV compares the odds taken on logged bankroll bets against the best available closing-line artifact. Market data can lag sportsbook updates.</span>"
        f"<span>{_escape(artifact_note)}</span>"
        "</div>"
    )
