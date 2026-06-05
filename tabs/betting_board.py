from __future__ import annotations

import html
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.common.paths import BETTING_BOARD_PATH, MARKET_MATCH_AUDIT_PATH
from utils.betting_board_artifacts import (
    get_betting_artifact_status,
    get_upcoming_artifact_status,
    load_upcoming_events,
    load_upcoming_fights,
)
from utils.betting_board_rules import normalize_betting_board_odds
from utils.data_loader import load_parquet
from utils.dm_workflow_status import remember_launched_workflow, render_workflow_status
from utils.github_actions import get_latest_workflow_run, trigger_workflow
from utils.ui.charts import apply_plotly_theme
from utils.ui.sections import page_header

SELECTED_EVENT_WORKFLOW = "run-betting-board-selected-event.yml"

RECOMMENDATION_COLORS = {
    "STRONG BET": "#35d96b",
    "LEAN BET": "#3b82f6",
    "WATCHLIST": "#facc15",
    "PASS": "#9aa8bd",
}


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def _escape(value) -> str:
    if pd.isna(value):
        return "—"
    return html.escape(str(value))


def _as_float(value, default=None):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return default
    return float(value)


def _money(value, decimals: int = 0) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    prefix = "-$" if value < 0 else "$"
    return f"{prefix}{abs(value):,.{decimals}f}"


def _signed_money(value, decimals: int = 0) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.{decimals}f}"


def _signed_pct(value, decimals: int = 1) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    if abs(value) <= 1:
        value *= 100
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _pct(value, decimals: int = 1) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    if abs(value) <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


def _american(value) -> str:
    value = _as_float(value)
    if value is None or value == 0:
        return "—"
    rounded = int(round(value))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def _confidence_value(value) -> float | None:
    value = _as_float(value)
    if value is None:
        return None
    return value * 100 if abs(value) <= 1 else value


def _probability(value) -> float | None:
    value = _as_float(value)
    if value is None:
        return None
    return value / 100 if value > 1 else value


def _ev_for_100(probability, american_odds) -> float | None:
    probability = _probability(probability)
    american_odds = _as_float(american_odds)
    if probability is None or american_odds is None or american_odds == 0:
        return None
    profit = american_odds if american_odds > 0 else 10000 / abs(american_odds)
    return probability * profit - (1 - probability) * 100


def _latest_timestamp(*frames: pd.DataFrame) -> str | None:
    candidates = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in ["decision_timestamp", "snapshot_timestamp", "run_timestamp"]:
            if column not in frame.columns:
                continue
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
            if not parsed.empty:
                candidates.append(parsed.max())
    if not candidates:
        return None
    latest = max(candidates).tz_convert(ZoneInfo("America/Chicago"))
    return f"Last Updated: {latest.strftime('%b %-d, %Y %I:%M %p %Z')}"


def _refresh_status_label(updated_label: str | None) -> str | None:
    """Show a compact running state in the timestamp slot after dispatch."""

    if not st.session_state.get("bb_refresh_running"):
        return updated_label

    ok, _, run = get_latest_workflow_run(SELECTED_EVENT_WORKFLOW)
    if ok and run is not None and run.get("status") == "completed":
        st.session_state["bb_refresh_running"] = False
        return updated_label

    return "Running....."


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def _event_id(row: dict | pd.Series | None):
    if row is None:
        return None
    return row.get("ufcstats_event_id") or row.get("event_id")


def _event_name(row: dict | pd.Series | None):
    if row is None:
        return None
    return row.get("ufcstats_event_name") or row.get("event_name")


def _event_date(row: dict | pd.Series | None):
    if row is None:
        return None
    return row.get("ufcstats_event_date") or row.get("event_date")


def _display_event_date(row: dict | pd.Series | None) -> str:
    date = _event_date(row)
    if not date:
        return "Date TBD"
    parsed = pd.to_datetime(date, errors="coerce")
    if pd.isna(parsed):
        return str(date)
    return parsed.strftime("%a, %b %-d, %Y")


def _event_location(row: dict | pd.Series | None):
    if row is None:
        return None
    return row.get("ufcstats_event_location") or row.get("event_location")


def _event_options(events: pd.DataFrame, board: pd.DataFrame) -> list[dict]:
    options: list[dict] = []
    if events is not None and not events.empty:
        sort_column = "ufcstats_event_date" if "ufcstats_event_date" in events.columns else None
        source = events.sort_values(sort_column, na_position="last") if sort_column else events
        options.extend(source.to_dict("records"))

    existing_names = {_event_name(option) for option in options}
    if board is not None and not board.empty and "event_name" in board.columns:
        for name in board["event_name"].dropna().astype(str).unique():
            if name not in existing_names:
                options.append({"event_name": name, "ufcstats_event_name": name})
    return options


def _event_label(option: dict) -> str:
    return _event_name(option) or "Unknown event"


def _current_event(options: list[dict]) -> dict | None:
    selected_name = st.session_state.get("bb_filter_event", "All Events")
    if not options or selected_name == "All Events":
        return None
    for option in options:
        if _event_label(option) == selected_name:
            return option
    return None


def _scope_board(board: pd.DataFrame, selected_event: dict | None) -> pd.DataFrame:
    if board is None or board.empty:
        return pd.DataFrame()
    if selected_event is None:
        return board.copy()

    selected_id = _event_id(selected_event)
    selected_name = _event_name(selected_event)
    masks = []
    if selected_id:
        for column in ["event_id", "ufcstats_event_id"]:
            if column in board.columns:
                masks.append(board[column].astype(str) == str(selected_id))
    if selected_name and "event_name" in board.columns:
        masks.append(board["event_name"].astype(str) == str(selected_name))
    if not masks:
        return board.copy()

    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    scoped = board[mask].copy()
    return scoped if not scoped.empty else board.copy()


def _fight_metadata(fights: pd.DataFrame) -> pd.DataFrame:
    if fights is None or fights.empty or "fight_id" not in fights.columns:
        return pd.DataFrame()
    columns = [
        column
        for column in ["fight_id", "fight_order", "weight_class", "event_date", "event_location"]
        if column in fights.columns
    ]
    return fights[columns].drop_duplicates("fight_id", keep="last")


def _recommendation(row: pd.Series) -> str:
    status = str(row.get("bet_status", "")).upper().strip()
    ev = _as_float(row.get("ev_dollars"), default=0.0) or 0.0
    if status == "OFFICIAL BET":
        return "STRONG BET" if ev >= 100 else "LEAN BET"
    if status == "WATCHLIST":
        return "WATCHLIST"
    if status in {"NO BET", "INVALID MODEL DATA", "LOW ODDS MATCH", "SPARSE FEATURES"}:
        return "PASS"
    if ev >= 100:
        return "STRONG BET"
    if ev >= 50:
        return "LEAN BET"
    if ev > 0:
        return "WATCHLIST"
    return "PASS"


def _display_board(board: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    if board is None or board.empty:
        return pd.DataFrame()

    prepared = normalize_betting_board_odds(board).copy()
    if not {"red_fighter", "blue_fighter"}.issubset(prepared.columns):
        return pd.DataFrame()
    valid = prepared.dropna(subset=["red_fighter", "blue_fighter"], how="any").copy()
    if valid.empty:
        return valid

    if "fight_id" in valid.columns:
        metadata = _fight_metadata(fights)
        if not metadata.empty:
            valid = valid.merge(metadata, on="fight_id", how="left", suffixes=("", "_card"))

    valid["ev_dollars"] = valid.apply(
        lambda row: _ev_for_100(row.get("best_prob"), row.get("best_american_odds")),
        axis=1,
    )
    if "best_ev" in valid.columns:
        valid["ev_dollars"] = valid["ev_dollars"].fillna(pd.to_numeric(valid["best_ev"], errors="coerce"))

    valid["recommendation"] = valid.apply(_recommendation, axis=1)
    valid["confidence_pct"] = valid["best_confidence"].apply(_confidence_value)
    valid["sort_ev"] = pd.to_numeric(valid["ev_dollars"], errors="coerce").fillna(-10_000)
    if "fight_order" in valid.columns:
        valid["sort_order"] = pd.to_numeric(valid["fight_order"], errors="coerce").fillna(999)
    else:
        valid["sort_order"] = range(1, len(valid) + 1)

    if "fight_id" in valid.columns:
        valid = valid.sort_values(["sort_order", "sort_ev"], ascending=[True, False])
        valid = valid.drop_duplicates("fight_id", keep="first")
    else:
        valid = valid.sort_values("sort_ev", ascending=False)
    return valid.reset_index(drop=True)


def _selected_event_title(selected_event: dict | None) -> str:
    if selected_event is not None:
        return _event_name(selected_event) or "Selected Event"
    selected_name = st.session_state.get("bb_filter_event", "All Events")
    return selected_name if selected_name != "All Events" else "All Events"


def _selected_event_date_label(selected_event: dict | None, display: pd.DataFrame) -> str:
    if selected_event is not None:
        return _display_event_date(selected_event)
    date_range = st.session_state.get("bb_filter_date_range")
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start = pd.to_datetime(date_range[0], errors="coerce")
        end = pd.to_datetime(date_range[1], errors="coerce")
        if not pd.isna(start) and not pd.isna(end):
            return f"{start.strftime('%b %-d')} – {end.strftime('%b %-d, %Y')}"
    if not display.empty and "event_date" in display.columns:
        parsed = pd.to_datetime(display["event_date"], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.min().strftime("%a, %b %-d, %Y")
    return "All upcoming dates"


def _selected_event_location_label(selected_event: dict | None, display: pd.DataFrame) -> str:
    if selected_event is not None:
        return _event_location(selected_event) or "Location TBD"
    if not display.empty and "event_location" in display.columns:
        locations = display["event_location"].dropna().astype(str).unique()
        if len(locations) == 1:
            return locations[0]
    return "All locations"


def _date_range_mask(display: pd.DataFrame) -> pd.Series:
    if "event_date" not in display.columns:
        return pd.Series(True, index=display.index)
    selected_range = st.session_state.get("bb_filter_date_range")
    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        return pd.Series(True, index=display.index)
    start = pd.to_datetime(selected_range[0], errors="coerce")
    end = pd.to_datetime(selected_range[1], errors="coerce")
    dates = pd.to_datetime(display["event_date"], errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return pd.Series(True, index=display.index)
    return dates.between(start, end) | dates.isna()


def _kelly_multiplier() -> float:
    return 0.25 if st.session_state.get("bb_kelly_mode") == "1/4 Kelly" else 0.50


def _is_bet(row: pd.Series) -> bool:
    return str(row.get("recommendation", "")).upper().strip() in {"STRONG BET", "LEAN BET"}


def _kelly_stake(row: pd.Series, bankroll: float = 10_000.0) -> float:
    if not _is_bet(row):
        return 0.0
    probability = _probability(row.get("best_prob"))
    american_odds = _as_float(row.get("best_american_odds"))
    if probability is None or american_odds is None or american_odds == 0:
        return 0.0
    b = american_odds / 100 if american_odds > 0 else 100 / abs(american_odds)
    if b <= 0:
        return 0.0
    full_kelly = max(0.0, ((b * probability) - (1 - probability)) / b)
    return round(bankroll * full_kelly * _kelly_multiplier())


def _apply_kelly_stakes(display: pd.DataFrame) -> pd.DataFrame:
    if display.empty:
        return display
    adjusted = display.copy()
    adjusted["recommended_stake"] = adjusted.apply(_kelly_stake, axis=1)
    return adjusted


def _apply_sidebar_filters(display: pd.DataFrame) -> pd.DataFrame:
    if display.empty:
        return display
    filtered = display.copy()
    filtered = filtered[_date_range_mask(filtered)]

    odds_min, odds_max = st.session_state.get("bb_filter_odds_range", (-250, 400))
    odds = pd.to_numeric(filtered.get("best_american_odds"), errors="coerce")
    hide_missing = st.session_state.get("bb_filter_hide_missing_odds", True)
    odds_mask = odds.between(odds_min, odds_max)
    if not hide_missing:
        odds_mask = odds_mask | odds.isna()
    filtered = filtered[odds_mask]

    ev_threshold = float(st.session_state.get("bb_filter_ev_threshold", 50))
    if ev_threshold > 0:
        ev = pd.to_numeric(filtered.get("ev_dollars"), errors="coerce")
        filtered = filtered[ev >= ev_threshold]
    elif st.session_state.get("bb_filter_positive_ev", False):
        ev = pd.to_numeric(filtered.get("ev_dollars"), errors="coerce")
        filtered = filtered[ev > 0]

    min_confidence = float(st.session_state.get("bb_filter_min_confidence", 70))
    confidence = pd.to_numeric(filtered.get("confidence_pct"), errors="coerce")
    filtered = filtered[confidence >= min_confidence]
    return filtered.reset_index(drop=True)


# -----------------------------------------------------------------------------
# UI rendering
# -----------------------------------------------------------------------------


def _metric_tile(label: str, value: str, caption: str, color: str = "#f5f7fb") -> None:
    st.html(
        "".join(
            [
                '<div class="bb-metric-card">',
                f'<div class="bb-metric-label">{html.escape(label)}</div>',
                f'<div class="bb-metric-value" style="color:{color};">{html.escape(value)}</div>',
                f'<div class="bb-metric-caption">{html.escape(caption)}</div>',
                "</div>",
            ]
        )
    )


def _render_header(updated_label: str | None, selected_event: dict | None) -> None:
    updated_label = _refresh_status_label(updated_label)
    top_cols = st.columns([1, 0.62])
    with top_cols[0]:
        page_header("Betting Board", "Live fight predictions and betting opportunities")
    with top_cols[1]:
        action_cols = st.columns([0.58, 0.42], vertical_alignment="center")
        with action_cols[0]:
            st.html(
                f'<div class="bb-top-actions"><span>{html.escape(updated_label or "Artifacts not refreshed yet")}</span></div>'
            )
        with action_cols[1]:
            event_id = _event_id(selected_event)
            disabled = not bool(event_id)
            if st.button("↻ Refresh Data", type="primary", use_container_width=True, disabled=disabled):
                ok, msg = trigger_workflow(
                    SELECTED_EVENT_WORKFLOW,
                    inputs={"event_id": str(event_id)},
                )
                if ok:
                    st.session_state["bb_refresh_running"] = True
                    remember_launched_workflow(
                        "betting_selected_event",
                        "Refresh Betting Board Data",
                        SELECTED_EVENT_WORKFLOW,
                        inputs={"event_id": str(event_id)},
                    )
                    st.rerun()
                else:
                    st.error(msg)


def _render_kpis(display: pd.DataFrame, total_fights: int | None = None) -> None:
    total_fights = len(display) if total_fights is None else total_fights
    displayed_fights = len(display)
    positive = int((display["ev_dollars"] > 0).sum()) if not display.empty else 0
    strong = int((display["recommendation"] == "STRONG BET").sum()) if not display.empty else 0
    total_ev = display["ev_dollars"].dropna().sum() if not display.empty else 0
    positive_ev = display.loc[display["ev_dollars"] > 0, "ev_dollars"].dropna()
    avg_ev = positive_ev.mean() if not positive_ev.empty else 0
    bet_display = display[display.apply(_is_bet, axis=1)] if not display.empty else display
    stake = bet_display["recommended_stake"].dropna().sum() if "recommended_stake" in bet_display.columns else 0

    cols = st.columns(6)
    with cols[0]:
        _metric_tile("Total Fights", str(total_fights), "Today / Upcoming", "#3b82f6")
    with cols[1]:
        pct_card = f"{positive / displayed_fights:.0%} displayed" if displayed_fights else "0% displayed"
        _metric_tile("Positive EV Fights", str(positive), pct_card, "#35d96b")
    with cols[2]:
        strong_card = f"{strong / displayed_fights:.0%} displayed" if displayed_fights else "0% displayed"
        _metric_tile("Strong Bets", str(strong), strong_card, "#35d96b")
    with cols[3]:
        _metric_tile("Total EV", _money(total_ev), "Across all fights", "#35d96b" if total_ev >= 0 else "#ef4444")
    with cols[4]:
        _metric_tile("Avg EV per Fight", _money(avg_ev), "Positive EV only", "#35d96b")
    with cols[5]:
        _metric_tile("Bankroll at Risk", _money(stake), "Recommended stakes", "#a855f7")


def _render_event_bar(selected_event: dict | None, display: pd.DataFrame) -> None:
    title = _selected_event_title(selected_event)
    date_label = _selected_event_date_label(selected_event, display)
    location = _selected_event_location_label(selected_event, display)
    st.html(
        "".join(
            [
                '<div class="bb-event-bar">',
                f'<div class="bb-event-heading">{_escape(title)}</div>',
                f'<div class="bb-event-meta-inline">📅 {_escape(date_label)}</div>',
                f'<div class="bb-event-meta-inline">📍 {_escape(location)}</div>',
                '<div class="bb-card-tabs"><span class="active">All Fights</span><span>Main Card</span><span>Prelims</span><span>Early Prelims</span></div>',
                '</div>',
            ]
        )
    )


def _corner_html(label: str, fighter: str, probability, odds, implied, edge, ev, color: str) -> str:
    return "".join(
        [
            '<div class="bb-corner-row">',
            f'<span class="bb-corner-badge" style="background:{color};">{label}</span>',
            f'<span class="bb-fighter-name">{_escape(fighter)}</span>',
            f'<span class="bb-prob">{_pct(probability)}</span>',
            f'<span class="bb-odds">{_american(odds)}</span>',
            f'<span class="bb-implied">{_pct(implied)}</span>',
            f'<span class="bb-edge {"pos" if (_as_float(edge, 0) or 0) >= 0 else "neg"}">{_signed_pct(edge)}</span>',
            f'<span class="bb-ev {"pos" if (_as_float(ev, 0) or 0) >= 0 else "neg"}">{_signed_money(ev)}</span>',
            "</div>",
        ]
    )


def _confidence_ring(confidence) -> str:
    confidence = _confidence_value(confidence) or 0
    color = "#35d96b" if confidence >= 70 else "#facc15" if confidence >= 60 else "#ef4444"
    return (
        f'<div class="bb-ring" style="background: conic-gradient({color} {confidence:.0f}%, rgba(148,163,184,.24) 0);">'
        f'<div>{confidence:.0f}%</div></div>'
    )


def _stake_text(row: pd.Series) -> str:
    if not _is_bet(row):
        return "—"
    stake = _as_float(row.get("recommended_stake"), 0) or 0
    if stake <= 0:
        return "—"
    return f"{_money(stake)}<br><span>{stake / 150:.1f}u</span>"


def _render_main_table(display: pd.DataFrame) -> None:
    if display.empty:
        st.warning("No betting board rows are available for this event. Refresh data or select a different event.")
        return

    rows = []
    for idx, row in display.iterrows():
        rec = row.get("recommendation", "PASS")
        rec_color = RECOMMENDATION_COLORS.get(rec, "#9aa8bd")
        rows.append(
            "".join(
                [
                    '<div class="bb-table-row">',
                    f'<div class="bb-rank">{idx + 1}</div>',
                    '<div class="bb-fight-cell">',
                    _corner_html("R", row.get("red_fighter"), row.get("red_model_prob"), row.get("red_american_odds"), row.get("red_implied_prob"), row.get("red_edge"), row.get("red_ev"), "#ef4444"),
                    _corner_html("B", row.get("blue_fighter"), row.get("blue_model_prob"), row.get("blue_american_odds"), row.get("blue_implied_prob"), row.get("blue_edge"), row.get("blue_ev"), "#3b82f6"),
                    "</div>",
                    f'<div class="bb-confidence-cell">{_confidence_ring(row.get("best_confidence"))}</div>',
                    f'<div class="bb-rec" style="color:{rec_color};">{_escape(rec)}</div>',
                    f'<div class="bb-stake">{_stake_text(row)}</div>',
                    '<div class="bb-menu">⋮</div>',
                    "</div>",
                ]
            )
        )

    bet_rows = display[display.apply(_is_bet, axis=1)] if not display.empty else display
    total_stake = bet_rows["recommended_stake"].dropna().sum() if "recommended_stake" in bet_rows.columns else 0
    table_html = "".join(
        [
            '<div class="bb-table">',
            f'<div class="bb-table-head"><span class="bb-rank-head"></span><div class="bb-fight-heading"><span class="corner-head"></span><span class="fight-label">Fight</span><span>Model<br>Prob.</span><span>Market<br>Odds</span><span>Implied<br>Prob.</span><span>Edge</span><span>EV ($)</span></div><span class="confidence-head">Confidence</span><span class="recommendation-head">Recommendation</span><span class="stake-head">Suggested Stake<br>({_escape(st.session_state.get("bb_kelly_mode", "1/2 Kelly"))})</span><span></span></div>',
            *rows,
            '<div class="bb-table-foot">',
            f'<span>{len(display)} fights</span>',
            '<span class="bb-legend"><b style="color:#35d96b;">◼ Strong Bet</b> EV >= $100&nbsp;&nbsp;&nbsp;<b style="color:#3b82f6;">◼ Lean Bet</b> $50 <= EV < $100&nbsp;&nbsp;&nbsp;<b style="color:#facc15;">◼ Watchlist</b> $0 < EV < $50&nbsp;&nbsp;&nbsp;<b style="color:#9aa8bd;">◼ Pass</b> EV <= $0</span>',
            f'<span class="bb-total-stake">Total Recommended Stake: {_money(total_stake)}</span>',
            "</div>",
            "</div>",
        ]
    )
    st.html(table_html)


def _render_top_ev(display: pd.DataFrame) -> None:
    st.html('<div class="bb-panel-title">Top Positive EV Opportunities</div>')
    if display.empty:
        st.info("No opportunities available.")
        return
    rows = display[display["ev_dollars"] > 0].sort_values("ev_dollars", ascending=False).head(5)
    if rows.empty:
        st.info("No positive EV opportunities for this card.")
        return
    table_rows = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        table_rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{_escape(row.get('best_side'))}</td>"
            f"<td>{_escape(row.get('blue_fighter') if row.get('best_side') == row.get('red_fighter') else row.get('red_fighter'))}</td>"
            f"<td>{_american(row.get('best_american_odds'))}</td>"
            f"<td>{_signed_money(row.get('ev_dollars'))}</td>"
            f"<td>{_pct(row.get('best_confidence'))}</td>"
            f"<td>{_stake_text(row) if _is_bet(row) else '—'}</td>"
            "</tr>"
        )
    st.html(
        '<table class="bb-mini-table"><thead><tr><th>Rank</th><th>Fighter</th><th>Opponent</th><th>Odds</th><th>EV ($)</th><th>Conf.</th><th>Stake</th></tr></thead><tbody>'
        + "".join(table_rows)
        + "</tbody></table>"
    )


def _donut(title: str, labels: list[str], values: list[int], colors: list[str], center: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                marker={"colors": colors},
                textinfo="none",
            )
        ]
    )
    fig.add_annotation(text=center, x=0.5, y=0.5, showarrow=False, font={"size": 18, "color": "#f5f7fb"})
    fig.update_layout(
        title={"text": title, "font": {"color": "#f5f7fb", "size": 14}},
        showlegend=True,
        legend={"font": {"color": "#f5f7fb"}},
    )
    return apply_plotly_theme(fig, height=245)


def _render_charts(display: pd.DataFrame) -> None:
    total = len(display)
    ev = display["ev_dollars"] if not display.empty else pd.Series(dtype=float)
    pos = int((ev > 0).sum())
    even = int((ev == 0).sum())
    neg = int((ev < 0).sum())
    ev_fig = _donut(
        "EV Distribution",
        [f"Positive EV ({pos})", f"Break Even ({even})", f"Negative EV ({neg})"],
        [pos, even, neg],
        ["#35d96b", "#9aa8bd", "#ef4444"],
        f"{total}<br>Total Fights",
    )

    conf = display["confidence_pct"] if not display.empty else pd.Series(dtype=float)
    high = int((conf >= 70).sum())
    med = int(((conf >= 60) & (conf < 70)).sum())
    low = int((conf < 60).sum())
    conf_fig = _donut(
        "Confidence Distribution",
        [f"High (70%+) ({high})", f"Medium (60-70%) ({med})", f"Low (<60%) ({low})"],
        [high, med, low],
        ["#35d96b", "#facc15", "#ef4444"],
        f"{total}<br>Total Fights",
    )
    st.plotly_chart(ev_fig, use_container_width=True)
    st.plotly_chart(conf_fig, use_container_width=True)


def _render_bottom(display: pd.DataFrame, chart_display: pd.DataFrame | None = None) -> None:
    chart_display = display if chart_display is None else chart_display
    cols = st.columns([1.25, 1, 1])
    with cols[0]:
        with st.container(border=True):
            _render_top_ev(display)
    with cols[1]:
        with st.container(border=True):
            ev = chart_display["ev_dollars"] if not chart_display.empty else pd.Series(dtype=float)
            pos = int((ev > 0).sum())
            even = int((ev == 0).sum())
            neg = int((ev < 0).sum())
            fig = _donut("EV Distribution", ["Positive EV", "Break Even", "Negative EV"], [pos, even, neg], ["#35d96b", "#9aa8bd", "#ef4444"], f"{len(chart_display)}<br>Total Fights")
            st.plotly_chart(fig, use_container_width=True)
    with cols[2]:
        with st.container(border=True):
            conf = chart_display["confidence_pct"] if not chart_display.empty else pd.Series(dtype=float)
            high = int((conf >= 70).sum())
            med = int(((conf >= 60) & (conf < 70)).sum())
            low = int((conf < 60).sum())
            fig = _donut("Confidence Distribution", ["High", "Medium", "Low"], [high, med, low], ["#35d96b", "#facc15", "#ef4444"], f"{len(chart_display)}<br>Total Fights")
            st.plotly_chart(fig, use_container_width=True)


def _render_artifacts_link() -> None:
    st.markdown(
        '<div class="bb-artifact-link-wrap"><a href="?betting_board_view=diagnostics" target="_self">Artifacts & audits →</a></div>',
        unsafe_allow_html=True,
    )


def _render_diagnostics() -> None:
    page_header("Betting Board Diagnostics", "Artifacts, audits, and workflow status")
    if st.button("← Back to Betting Board"):
        st.session_state["betting_board_view"] = "board"
        st.query_params.clear()
        st.rerun()

    status = pd.concat(
        [
            get_upcoming_artifact_status().assign(group="Card selection"),
            get_betting_artifact_status().assign(group="Betting outputs"),
        ],
        ignore_index=True,
    )
    st.dataframe(status, use_container_width=True, hide_index=True)
    audit = load_parquet(MARKET_MATCH_AUDIT_PATH)
    st.markdown("### Market Match Audit")
    if audit.empty:
        st.info(f"No market match audit found at `{MARKET_MATCH_AUDIT_PATH}`.")
    else:
        st.dataframe(audit, use_container_width=True, hide_index=True)
    render_workflow_status("betting_selected_event")


def _inject_betting_board_css() -> None:
    st.markdown(
        """
        <style>
        .bb-top-actions { display:flex; justify-content:flex-end; align-items:center; color:#dbe7f5; font-size:.78rem; line-height:1; white-space:nowrap; }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:linear-gradient(180deg, rgba(16,28,45,.92), rgba(13,23,39,.94)); border:1px solid rgba(38,54,74,.96); border-radius:10px; min-height:2.9rem; color:#f5f7fb; }
        div[data-testid="stSelectbox"] [data-baseweb="select"] span { color:#f5f7fb; font-size:.95rem; font-weight:900; letter-spacing:-.01em; }
        .bb-metric-card, .bb-panel, .bb-table, .bb-event-title, .bb-event-meta { background:linear-gradient(180deg, rgba(16,28,45,.92), rgba(13,23,39,.94)); border:1px solid rgba(38,54,74,.96); border-radius:10px; }
        .bb-metric-card { min-height:98px; text-align:center; padding:1rem .75rem; }
        .bb-metric-label { color:#f5f7fb; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
        .bb-metric-value { font-size:1.85rem; line-height:1.05; font-weight:900; margin-top:.35rem; }
        .bb-metric-caption { color:#dbe7f5; font-size:.78rem; margin-top:.45rem; }
        .bb-event-title { color:#f5f7fb; font-weight:900; font-size:1.05rem; padding:.72rem 1rem; min-height:2.9rem; }
        .bb-event-bar { display:grid; grid-template-columns:1.35fr .55fr .9fr 1.55fr; gap:1rem; align-items:center; margin-top:1rem; padding:.55rem .75rem; background:linear-gradient(180deg, rgba(16,28,45,.92), rgba(13,23,39,.94)); border:1px solid rgba(38,54,74,.96); border-radius:10px; }
        .bb-event-heading { color:#f5f7fb; font-size:1.05rem; font-weight:900; letter-spacing:-.02em; }
        .bb-event-meta, .bb-event-meta-inline { color:#dbe7f5; font-size:.82rem; font-weight:700; }
        .bb-card-tabs { display:flex; justify-content:flex-end; gap:.45rem; }
        .bb-card-tabs span { color:#f5f7fb; font-size:.78rem; font-weight:800; padding:.55rem .8rem; border:1px solid rgba(38,54,74,.96); border-radius:7px; }
        .bb-card-tabs .active { background:#143b78; border-color:#2f68d8; }
        .bb-table { overflow:hidden; margin-top:.45rem; }
        .bb-table-head { display:grid; grid-template-columns:.28fr 4.82fr .9fr 1.15fr 1.1fr .2fr; gap:0; align-items:center; padding:.75rem 1rem; border-bottom:1px solid rgba(38,54,74,.95); color:#f5f7fb; font-size:.72rem; font-weight:900; text-transform:uppercase; }
        .bb-fight-heading { display:grid; grid-template-columns:1.45rem 1.9fr .8fr .8fr .85fr .75fr .75fr; align-items:center; column-gap:.35rem; }
        .bb-fight-heading span { text-align:center; line-height:1.15; }
        .bb-fight-heading .fight-label { text-align:left; }
        .bb-table-head > span:nth-child(n+3) { text-align:center; line-height:1.15; display:flex; align-items:center; justify-content:center; min-height:2.1rem; }
        .bb-table-head .confidence-head { transform:translateX(-.15rem); }
        .bb-table-head .recommendation-head { transform:translateX(.1rem); }
        .bb-table-head .stake-head { transform:translateX(.15rem); }
        .bb-table-row { display:grid; grid-template-columns:.28fr 4.82fr .9fr 1.15fr 1.1fr .2fr; align-items:center; min-height:76px; padding:.45rem 1rem; border-bottom:1px solid rgba(38,54,74,.75); }
        .bb-rank, .bb-menu { color:#dbe7f5; font-size:.8rem; }
        .bb-fight-cell { display:flex; flex-direction:column; gap:.42rem; }
        .bb-corner-row { display:grid; grid-template-columns:1.45rem 1.9fr .8fr .8fr .85fr .75fr .75fr; align-items:center; gap:.35rem; color:#f5f7fb; font-size:.84rem; }
        .bb-corner-badge { color:#fff; border-radius:4px; width:1.05rem; height:1.05rem; display:inline-flex; justify-content:center; align-items:center; font-size:.68rem; font-weight:900; }
        .bb-fighter-name { font-weight:800; }
        .bb-prob { color:#35d96b; font-weight:800; }
        .bb-odds, .bb-implied { color:#f5f7fb; }
        .bb-edge.pos, .bb-ev.pos { color:#35d96b; } .bb-edge.neg, .bb-ev.neg { color:#ef4444; }
        .bb-confidence-cell { display:flex; justify-content:center; }
        .bb-ring { width:52px; height:52px; border-radius:50%; display:flex; align-items:center; justify-content:center; }
        .bb-ring div { background:#0d1727; width:39px; height:39px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#f5f7fb; font-size:.78rem; font-weight:900; }
        .bb-rec { text-align:center; font-size:.82rem; font-weight:900; }
        .bb-stake { color:#f5f7fb; text-align:center; font-weight:900; font-size:.83rem; } .bb-stake span { color:#dbe7f5; font-weight:500; }
        .bb-table-foot { display:grid; grid-template-columns:.5fr 2.5fr 1.1fr; gap:1rem; padding:.75rem 1rem; color:#dbe7f5; font-size:.78rem; align-items:center; }
        .bb-total-stake { color:#a855f7; font-size:.9rem; font-weight:900; text-align:right; }
        .bb-panel { padding:.9rem 1rem; min-height:225px; margin-top:.65rem; }
        .bb-panel-title { color:#f5f7fb; font-weight:900; text-transform:uppercase; font-size:.78rem; margin-bottom:.65rem; }
        .bb-mini-table { width:100%; border-collapse:collapse; color:#f5f7fb; font-size:.76rem; } .bb-mini-table th { color:#dbe7f5; text-align:left; text-transform:uppercase; font-size:.68rem; padding:.35rem; } .bb-mini-table td { padding:.42rem .35rem; border-top:1px solid rgba(38,54,74,.65); }
        .bb-artifact-link-wrap { display:flex; justify-content:flex-end; margin-top:.45rem; font-size:.72rem; }
        .bb-artifact-link-wrap a { color:#8fb4ff; text-decoration:none; }
        .bb-artifact-link-wrap a:hover { color:#dbe7f5; text-decoration:underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_betting_board():
    _inject_betting_board_css()

    if st.query_params.get("betting_board_view") == "diagnostics":
        st.session_state["betting_board_view"] = "diagnostics"

    if st.session_state.get("betting_board_view") == "diagnostics":
        _render_diagnostics()
        return

    raw_board = load_parquet(BETTING_BOARD_PATH)
    events, events_error = load_upcoming_events()
    fights, fights_error = load_upcoming_fights()
    options = _event_options(events, raw_board)
    selected_event = _current_event(options) if options else None
    scoped = _scope_board(raw_board, selected_event)
    display = _display_board(scoped, fights)
    card_display = display.copy()
    card_fight_count = len(card_display)
    updated = _latest_timestamp(raw_board, events, fights)

    _render_header(updated, selected_event)
    if events_error:
        st.warning(events_error)
    if fights_error:
        st.warning(fights_error)

    display = _apply_sidebar_filters(display)
    display = _apply_kelly_stakes(display)

    _render_kpis(display, total_fights=card_fight_count)
    _render_event_bar(selected_event, display)

    _render_main_table(display)
    _render_bottom(display, chart_display=card_display)
    _render_artifacts_link()
