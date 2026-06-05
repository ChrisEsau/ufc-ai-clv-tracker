from __future__ import annotations

import html
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.common.paths import MARKET_SNAPSHOTS_PATH
from utils.data_loader import load_parquet
from utils.github_actions import trigger_workflow
from utils.ui.charts import apply_plotly_theme

MARKET_WORKFLOW = "run-market-update.yml"
CLV_WORKFLOW = "run-clv-tracker.yml"
CENTRAL_TZ = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class ClvSummary:
    bets: int
    beat_pct: float
    avg_clv: float
    median_clv: float
    positive: int
    negative: int
    total_clv: float


def _escape(value) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def _fmt_pct(value, decimals: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{float(value):.{decimals}f}%"


def _fmt_int(value) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(round(float(value))):,}"


def _fmt_american(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    odds = int(round(float(value)))
    return f"+{odds}" if odds > 0 else str(odds)


def _american_to_decimal(odds) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds == 0:
        return None
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def _clv_pct(taken_odds, closing_odds) -> float | None:
    """Return closing-line value percentage using decimal odds ratio."""

    taken_decimal = _american_to_decimal(taken_odds)
    closing_decimal = _american_to_decimal(closing_odds)
    if not taken_decimal or not closing_decimal:
        return None
    return (taken_decimal / closing_decimal - 1) * 100


def _normalize_snapshots(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    snapshots = raw.copy()
    snapshots["snapshot_timestamp"] = pd.to_datetime(
        snapshots.get("snapshot_timestamp"), utc=True, errors="coerce"
    )
    snapshots["commence_time"] = pd.to_datetime(
        snapshots.get("commence_time"), utc=True, errors="coerce"
    )
    for column in ["red_american_odds", "blue_american_odds", "model_confidence"]:
        if column in snapshots.columns:
            snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")
    return snapshots.dropna(subset=["fight_id", "snapshot_timestamp"]).copy()


def _selected_side(row: pd.Series) -> tuple[str, str, float | None]:
    pick = str(row.get("model_pick", "")).strip()
    red = str(row.get("red_fighter", "")).strip()
    blue = str(row.get("blue_fighter", "")).strip()
    if pick and pick == blue:
        return blue, red, row.get("blue_american_odds")
    return red, blue, row.get("red_american_odds")


def _build_clv_rows(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()

    rows = []
    sort_cols = ["fight_id", "bookmaker", "snapshot_timestamp"]
    for (_, bookmaker), group in snapshots.sort_values(sort_cols).groupby(["fight_id", "bookmaker"], dropna=False):
        first = group.iloc[0]
        last = group.iloc[-1]
        fighter, opponent, taken_odds = _selected_side(first)
        _, _, closing_odds = _selected_side(last)
        clv = _clv_pct(taken_odds, closing_odds)
        rows.append(
            {
                "date": last.get("commence_time") if pd.notna(last.get("commence_time")) else last.get("snapshot_timestamp"),
                "snapshot_timestamp": last.get("snapshot_timestamp"),
                "event_name": last.get("event_name", "Unknown Event"),
                "fight_id": last.get("fight_id"),
                "fight": f"{first.get('red_fighter', '')} vs {first.get('blue_fighter', '')}",
                "fighter": fighter,
                "opponent": opponent,
                "bookmaker": bookmaker or "Unknown Book",
                "market_type": "Moneyline",
                "odds_taken": taken_odds,
                "closing_odds": closing_odds,
                "clv_pct": clv,
                "beat_closing": bool(clv is not None and clv >= 0),
                "model_confidence": last.get("model_confidence"),
            }
        )
    return pd.DataFrame(rows)


def _date_bounds(snapshots: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    dates = pd.to_datetime(snapshots.get("commence_time"), utc=True, errors="coerce").dropna()
    if dates.empty:
        dates = pd.to_datetime(snapshots.get("snapshot_timestamp"), utc=True, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().date(), dates.max().date()


def _filter_clv_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    filtered = rows.copy()
    event = st.session_state.get("sidebar_clv_event", "All Events")
    if event != "All Events":
        filtered = filtered[filtered["event_name"] == event]
    book = st.session_state.get("sidebar_clv_book", "All Books")
    if book != "All Books":
        filtered = filtered[filtered["bookmaker"] == book]
    market = st.session_state.get("sidebar_clv_market", "Moneyline")
    if market != "All Markets":
        filtered = filtered[filtered["market_type"] == market]
    date_range = st.session_state.get("sidebar_clv_date_range")
    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        dates = pd.to_datetime(filtered["date"], utc=True, errors="coerce").dt.tz_localize(None)
        filtered = filtered[dates.between(start, end) | dates.isna()]
    if st.session_state.get("sidebar_clv_my_bets_only", False):
        filtered = filtered[filtered["odds_taken"].notna()]
    return filtered.reset_index(drop=True)


def _summary(rows: pd.DataFrame) -> ClvSummary:
    if rows.empty or "clv_pct" not in rows.columns:
        return ClvSummary(0, 0, 0, 0, 0, 0, 0)
    clv = pd.to_numeric(rows["clv_pct"], errors="coerce").dropna()
    bets = len(clv)
    if bets == 0:
        return ClvSummary(0, 0, 0, 0, 0, 0, 0)
    positive = int((clv >= 0).sum())
    negative = int((clv < 0).sum())
    return ClvSummary(
        bets=bets,
        beat_pct=positive / bets * 100,
        avg_clv=float(clv.mean()),
        median_clv=float(clv.median()),
        positive=positive,
        negative=negative,
        total_clv=float(clv.sum()),
    )


def _last_updated(snapshots: pd.DataFrame) -> str:
    latest = pd.to_datetime(snapshots.get("snapshot_timestamp"), utc=True, errors="coerce").max()
    if pd.isna(latest):
        return "Last Updated: No market snapshots"
    local = latest.tz_convert(CENTRAL_TZ)
    return f"Last Updated: {local.strftime('%b %-d, %Y %I:%M %p %Z')}"


def _inject_css() -> None:
    st.html(
        """
        <style>
        .clv-topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin:.1rem 0 1rem; }
        .clv-title { color:#f5f7fb; font-size:1.95rem; line-height:1; font-weight:900; letter-spacing:-.04em; }
        .clv-subtitle { color:#dbe7f5; font-size:1rem; margin-top:.35rem; }
        .clv-actions { display:flex; align-items:center; justify-content:flex-end; gap:.85rem; color:#f5f7fb; font-size:.82rem; font-weight:700; }
        .clv-kpi-grid { display:grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap:.65rem; margin:.7rem 0 1rem; }
        .clv-card { background:linear-gradient(180deg, rgba(17,31,49,.94), rgba(12,24,39,.98)); border:1px solid rgba(43,60,82,.96); border-radius:7px; box-shadow:0 20px 40px rgba(0,0,0,.22); }
        .clv-kpi { min-height:86px; padding:1rem .75rem .85rem; text-align:center; }
        .clv-kpi-label { color:#f5f7fb; text-transform:uppercase; font-size:.72rem; font-weight:800; }
        .clv-kpi-value { color:#f5f7fb; font-size:1.75rem; line-height:1.15; font-weight:900; margin-top:.35rem; }
        .clv-green { color:#31df63 !important; } .clv-red { color:#ff4949 !important; }
        .clv-kpi-caption { color:#f5f7fb; font-size:.78rem; margin-top:.25rem; }
        .clv-tabs { display:flex; gap:1.9rem; padding:.75rem 1rem 0; min-height:42px; align-items:flex-end; text-transform:uppercase; font-size:.82rem; letter-spacing:.02em; }
        .clv-tabs span { color:#aeb8c6; padding:0 0 .72rem; }
        .clv-tabs .active { color:#2f88ff; border-bottom:2px solid #2f72ff; }
        .clv-grid { display:grid; grid-template-columns:2.1fr 1.25fr; gap:.7rem; margin-top:.7rem; }
        .clv-card-title { color:#f5f7fb; text-transform:uppercase; font-size:.82rem; font-weight:900; margin:1rem 1rem .4rem; }
        .clv-chart-wrap { padding:.4rem 1rem 1rem; }
        .clv-line-layout { display:grid; grid-template-columns:1.65fr .68fr; gap:.75rem; align-items:stretch; }
        .clv-fight-panel { border:1px solid rgba(43,60,82,.9); border-radius:6px; padding:.85rem; color:#f5f7fb; align-self:center; }
        .clv-fight-panel h4 { margin:0 0 .25rem; font-size:1rem; }
        .clv-fight-panel p { margin:.12rem 0; color:#dbe7f5; font-size:.78rem; }
        .clv-mini-row { display:grid; grid-template-columns:1.1fr .75fr .75fr; border-top:1px solid rgba(43,60,82,.8); padding:.5rem 0; font-size:.78rem; }
        .clv-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.8rem; }
        .clv-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.68rem; letter-spacing:.03em; padding:.55rem .45rem; border-bottom:1px solid rgba(43,60,82,.9); }
        .clv-table td { padding:.62rem .45rem; border-bottom:1px solid rgba(43,60,82,.7); }
        .clv-center { text-align:center !important; } .clv-right { text-align:right !important; }
        .clv-link { color:#2f88ff; text-align:center; font-weight:800; padding:.75rem; }
        .clv-insight { display:flex; justify-content:space-between; align-items:center; margin-top:1rem; padding:.85rem 1rem; color:#d89cff; background:rgba(92, 42, 160, .16); border:1px solid rgba(168,85,247,.45); border-radius:5px; font-size:.84rem; }
        @media (max-width: 1200px) { .clv-kpi-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } .clv-grid, .clv-line-layout { grid-template-columns:1fr; } }
        </style>
        """
    )


def _kpi(label: str, value: str, caption: str = "", color_class: str = "") -> str:
    return (
        '<div class="clv-card clv-kpi">'
        f'<div class="clv-kpi-label">{_escape(label)}</div>'
        f'<div class="clv-kpi-value {color_class}">{_escape(value)}</div>'
        f'<div class="clv-kpi-caption">{_escape(caption)}</div>'
        '</div>'
    )


def _render_header(snapshots: pd.DataFrame) -> None:
    left, right = st.columns([1.8, 1])
    with left:
        st.html(
            '<div class="clv-title">LINE MOVEMENT / CLV</div>'
            '<div class="clv-subtitle">Track market movement and closing line value performance</div>'
        )
    with right:
        st.markdown(f"<div class='clv-actions'>{_escape(_last_updated(snapshots))}</div>", unsafe_allow_html=True)
        if st.button("↻  Refresh Data", use_container_width=True, key="clv_refresh_data"):
            ok_market, msg_market = trigger_workflow(MARKET_WORKFLOW)
            ok_clv, msg_clv = trigger_workflow(CLV_WORKFLOW)
            if ok_market or ok_clv:
                st.toast("Refresh workflows launched.", icon="↻")
            else:
                st.warning(f"Unable to launch refresh workflows: {msg_market}; {msg_clv}")


def _render_kpis(summary: ClvSummary) -> None:
    cards = [
        _kpi("Bets Tracked", _fmt_int(summary.bets)),
        _kpi("Beat Closing Line %", _fmt_pct(summary.beat_pct), f"{summary.positive} of {summary.bets}", "clv-green"),
        _kpi("Average CLV", _fmt_pct(summary.avg_clv, signed=True), color_class="clv-green" if summary.avg_clv >= 0 else "clv-red"),
        _kpi("Median CLV", _fmt_pct(summary.median_clv, signed=True), color_class="clv-green" if summary.median_clv >= 0 else "clv-red"),
        _kpi("Positive CLV %", _fmt_pct(summary.beat_pct), f"{summary.positive} of {summary.bets}", "clv-green"),
        _kpi("Negative CLV %", _fmt_pct(100 - summary.beat_pct if summary.bets else 0), f"{summary.negative} of {summary.bets}", "clv-red"),
        _kpi("Total CLV", _fmt_pct(summary.total_clv, signed=True), "Across all bets", "clv-green" if summary.total_clv >= 0 else "clv-red"),
    ]
    st.html('<div class="clv-kpi-grid">' + ''.join(cards) + '</div>')


def _render_tabs() -> None:
    st.html(
        '<div class="clv-card clv-tabs">'
        '<span class="active">Line Movement</span><span>CLV Results</span><span>Book Comparison</span><span>Steam Moves</span><span>CLV By Metrics</span>'
        '</div>'
    )


def _selected_fight_rows(snapshots: pd.DataFrame, clv_rows: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots
    fight_id = clv_rows.iloc[0]["fight_id"] if not clv_rows.empty else snapshots.iloc[0]["fight_id"]
    book = st.session_state.get("sidebar_clv_book", "All Books")
    rows = snapshots[snapshots["fight_id"] == fight_id].copy()
    if book != "All Books":
        rows = rows[rows["bookmaker"] == book]
    return rows.sort_values("snapshot_timestamp")


def _line_figure(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if rows.empty:
        return apply_plotly_theme(fig)
    x = rows["snapshot_timestamp"]
    red = pd.to_numeric(rows["red_american_odds"], errors="coerce")
    blue = pd.to_numeric(rows["blue_american_odds"], errors="coerce")
    fig.add_trace(go.Scatter(x=x, y=red, mode="lines+markers", name="Red Fighter", line=dict(color="#31df63", width=2)))
    fig.add_trace(go.Scatter(x=x, y=blue, mode="lines+markers", name="Blue Fighter", line=dict(color="#2f88ff", width=2)))
    if len(rows) > 0:
        opening = red.iloc[0]
        closing = red.iloc[-1]
        fig.add_trace(go.Scatter(x=x, y=[opening] * len(rows), mode="lines", name="Opening Line", line=dict(color="#facc15", dash="dot")))
        fig.add_trace(go.Scatter(x=x, y=[closing] * len(rows), mode="lines", name="Current Line", line=dict(color="#a855f7", dash="dash")))
    fig.update_layout(height=330, margin=dict(l=30, r=10, t=10, b=20), yaxis_title="American Odds", legend=dict(orientation="h", y=-0.18, font=dict(color="#f5f7fb")))
    return apply_plotly_theme(fig)


def _render_fight_panel(rows: pd.DataFrame, clv_rows: pd.DataFrame) -> str:
    if rows.empty:
        return '<div class="clv-fight-panel"><h4>No market movement selected</h4><p>Refresh market snapshots to populate line history.</p></div>'
    first = rows.iloc[0]
    last = rows.iloc[-1]
    clv_row = clv_rows[clv_rows["fight_id"] == first.get("fight_id")].head(1)
    clv_value = clv_row.iloc[0]["clv_pct"] if not clv_row.empty else None
    fight = f"{first.get('red_fighter', '')} vs {first.get('blue_fighter', '')}"
    event = first.get("event_name", "")
    date = pd.to_datetime(first.get("commence_time"), utc=True, errors="coerce")
    date_text = date.strftime("%b %-d, %Y") if pd.notna(date) else "Date TBD"
    return f"""
    <div class="clv-fight-panel">
      <h4>{_escape(fight)}</h4>
      <p>Moneyline · Selected market</p><p>{_escape(date_text)} · {_escape(event)}</p>
      <div class="clv-mini-row"><strong>Price</strong><strong>{_escape(first.get('red_fighter', 'Red'))}</strong><strong>{_escape(first.get('blue_fighter', 'Blue'))}</strong></div>
      <div class="clv-mini-row"><span>Opening</span><span>{_fmt_american(first.get('red_american_odds'))}</span><span>{_fmt_american(first.get('blue_american_odds'))}</span></div>
      <div class="clv-mini-row"><span>Current</span><span>{_fmt_american(last.get('red_american_odds'))}</span><span>{_fmt_american(last.get('blue_american_odds'))}</span></div>
      <div class="clv-mini-row"><span>CLV</span><span class="{'clv-green' if (clv_value or 0) >= 0 else 'clv-red'}">{_fmt_pct(clv_value, signed=True)}</span><span>—</span></div>
    </div>
    """


def _donut_figure(summary: ClvSummary) -> go.Figure:
    labels = ["Positive CLV (>= 0%)", "Small Negative (0% to -5%)", "Medium Negative (-5% to -10%)", "Large Negative (< -10%)"]
    values = [summary.positive, 0, 0, 0]
    # The detailed negative buckets are computed in _render_right_column when rows are available.
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.62, marker=dict(colors=["#31df63", "#facc15", "#f59e0b", "#ff4949"]), textinfo="none")])
    fig.update_layout(height=230, margin=dict(l=5, r=5, t=5, b=5), showlegend=False, annotations=[dict(text=f"<b>{summary.bets}</b><br>Total Bets", x=.5, y=.5, showarrow=False, font=dict(color="#f5f7fb", size=16))])
    return apply_plotly_theme(fig)


def _distribution_figure(rows: pd.DataFrame) -> go.Figure:
    clv = pd.to_numeric(rows.get("clv_pct"), errors="coerce").dropna()
    buckets = [
        ("Positive CLV (>= 0%)", int((clv >= 0).sum()), "#31df63"),
        ("Small Negative (0% to -5%)", int(((clv < 0) & (clv >= -5)).sum()), "#facc15"),
        ("Medium Negative (-5% to -10%)", int(((clv < -5) & (clv >= -10)).sum()), "#f59e0b"),
        ("Large Negative (< -10%)", int((clv < -10).sum()), "#ff4949"),
    ]
    total = max(1, sum(v for _, v, _ in buckets))
    fig = go.Figure(data=[go.Pie(labels=[b[0] for b in buckets], values=[b[1] for b in buckets], hole=.62, marker=dict(colors=[b[2] for b in buckets]), textinfo="none")])
    fig.update_layout(height=230, margin=dict(l=5, r=5, t=5, b=5), showlegend=False, annotations=[dict(text=f"<b>{len(clv)}</b><br>Total Bets", x=.5, y=.5, showarrow=False, font=dict(color="#f5f7fb", size=16))])
    return apply_plotly_theme(fig)


def _render_distribution_legend(rows: pd.DataFrame) -> str:
    clv = pd.to_numeric(rows.get("clv_pct"), errors="coerce").dropna()
    total = len(clv) or 1
    buckets = [
        ("Positive CLV (>= 0%)", int((clv >= 0).sum()), "#31df63"),
        ("Small Negative (0% to -5%)", int(((clv < 0) & (clv >= -5)).sum()), "#facc15"),
        ("Medium Negative (-5% to -10%)", int(((clv < -5) & (clv >= -10)).sum()), "#f59e0b"),
        ("Large Negative (< -10%)", int((clv < -10).sum()), "#ff4949"),
    ]
    rows_html = "".join(
        f'<div class="clv-mini-row"><span><b style="color:{color}">■</b> {_escape(label)}</span><span>{count}</span><span>{count / total * 100:.1f}%</span></div>'
        for label, count, color in buckets
    )
    return rows_html


def _odds_range_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        body = '<tr><td colspan="4">No CLV rows available.</td></tr>'
    else:
        odds = pd.to_numeric(rows["odds_taken"], errors="coerce")
        clv = pd.to_numeric(rows["clv_pct"], errors="coerce")
        ranges = [
            ("Favorites (-250 to -110)", odds.between(-250, -110)),
            ("Slight Underdogs (+100 to +200)", odds.between(100, 200)),
            ("Medium Underdogs (+201 to +400)", odds.between(201, 400)),
            ("Large Underdogs (+401 and up)", odds >= 401),
        ]
        lines = []
        for label, mask in ranges:
            subset = rows[mask]
            subset_clv = pd.to_numeric(subset.get("clv_pct"), errors="coerce").dropna()
            bets = len(subset_clv)
            beat = (subset_clv >= 0).mean() * 100 if bets else 0
            avg = subset_clv.mean() if bets else 0
            lines.append(f"<tr><td>{_escape(label)}</td><td class='clv-center'>{bets}</td><td class='clv-right clv-green'>{_fmt_pct(beat)}</td><td class='clv-right clv-green'>{_fmt_pct(avg, signed=True)}</td></tr>")
        body = "".join(lines)
    return f"<table class='clv-table'><thead><tr><th>Odds Range</th><th class='clv-center'>Bets</th><th class='clv-right'>Beat CL%</th><th class='clv-right'>Avg CLV</th></tr></thead><tbody>{body}</tbody></table>"


def _book_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        body = '<tr><td colspan="4">No sportsbook rows available.</td></tr>'
    else:
        lines = []
        for book, subset in rows.groupby("bookmaker"):
            clv = pd.to_numeric(subset["clv_pct"], errors="coerce").dropna()
            bets = len(clv)
            beat = (clv >= 0).mean() * 100 if bets else 0
            avg = clv.mean() if bets else 0
            lines.append(f"<tr><td>{_escape(book)}</td><td class='clv-center'>{bets}</td><td class='clv-right clv-green'>{_fmt_pct(beat)}</td><td class='clv-right clv-green'>{_fmt_pct(avg, signed=True)}</td></tr>")
        body = "".join(lines[:5])
    return f"<table class='clv-table'><thead><tr><th>Sportsbook</th><th class='clv-center'>Bets</th><th class='clv-right'>Beat CL%</th><th class='clv-right'>Avg CLV</th></tr></thead><tbody>{body}</tbody></table>"


def _recent_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        body = '<tr><td colspan="8">No recent CLV rows available.</td></tr>'
    else:
        recent = rows.sort_values("date", ascending=False).head(5)
        body = "".join(
            f"<tr><td>{pd.to_datetime(row.get('date')).strftime('%b %-d, %Y') if pd.notna(row.get('date')) else '—'}</td>"
            f"<td>{_escape(row.get('event_name'))}</td><td>{_escape(row.get('fight'))}</td><td>{_escape(row.get('fighter'))}</td>"
            f"<td class='clv-right'>{_fmt_american(row.get('odds_taken'))}</td><td class='clv-right'>{_fmt_american(row.get('closing_odds'))}</td>"
            f"<td class='clv-right {'clv-green' if (row.get('clv_pct') or 0) >= 0 else 'clv-red'}'>{_fmt_pct(row.get('clv_pct'), signed=True)}</td>"
            f"<td class='clv-right'>{'Win' if (row.get('clv_pct') or 0) >= 0 else 'Loss'}</td></tr>"
            for _, row in recent.iterrows()
        )
    return f"<table class='clv-table'><thead><tr><th>Date</th><th>Event</th><th>Fight</th><th>Pick</th><th class='clv-right'>Odds Taken</th><th class='clv-right'>Closing Odds</th><th class='clv-right'>CLV</th><th class='clv-right'>Result</th></tr></thead><tbody>{body}</tbody></table><div class='clv-link'>View all CLV results →</div>"


def render_line_movement():
    _inject_css()
    snapshots = _normalize_snapshots(load_parquet(MARKET_SNAPSHOTS_PATH))
    clv_rows_all = _build_clv_rows(snapshots)
    clv_rows = _filter_clv_rows(clv_rows_all)
    summary = _summary(clv_rows)

    _render_header(snapshots)
    _render_kpis(summary)
    _render_tabs()

    if snapshots.empty:
        st.info("No market snapshots found yet. Use Refresh Data after market workflows are configured.")
        return

    selected_rows = _selected_fight_rows(snapshots, clv_rows)
    left, right = st.columns([2.1, 1.25])
    with left:
        st.html('<div class="clv-card"><div class="clv-card-title">Line Movement Over Time ⓘ</div><div class="clv-chart-wrap">')
        chart_col, panel_col = st.columns([1.65, .68])
        with chart_col:
            st.plotly_chart(_line_figure(selected_rows), use_container_width=True, config={"displayModeBar": False})
        with panel_col:
            st.html(_render_fight_panel(selected_rows, clv_rows))
        st.html('</div></div>')

        st.html('<div class="clv-card" style="margin-top:.7rem;"><div class="clv-card-title">Recent CLV Results</div><div class="clv-chart-wrap">' + _recent_table(clv_rows) + '</div></div>')
    with right:
        st.html('<div class="clv-card"><div class="clv-card-title">CLV Distribution</div><div class="clv-chart-wrap">')
        st.plotly_chart(_distribution_figure(clv_rows), use_container_width=True, config={"displayModeBar": False})
        st.html(_render_distribution_legend(clv_rows) + '</div></div>')
        st.html('<div class="clv-card" style="margin-top:.7rem;"><div class="clv-card-title">CLV by Odds Range</div><div class="clv-chart-wrap">' + _odds_range_table(clv_rows) + '</div></div>')
        st.html('<div class="clv-card" style="margin-top:.7rem;"><div class="clv-card-title">Book Comparison <span style="font-weight:500; color:#dbe7f5; text-transform:none;">(Avg CLV %)</span></div><div class="clv-chart-wrap">' + _book_table(clv_rows) + '</div></div>')

    st.html(
        f"<div class='clv-insight'><span>🧬 <strong>CLV Insight:</strong> You are beating the closing line {_fmt_pct(summary.beat_pct)} of the time. Positive CLV is a strong indicator of long-term profitability.</span><span>Learn more about CLV →</span></div>"
    )
