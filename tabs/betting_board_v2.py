from __future__ import annotations

import html
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from pipeline.common.paths import BETTING_OUTCOMES_PATH
from pipeline.common.risk_settings import load_risk_settings
from utils.betting_board_artifacts import load_upcoming_events, load_upcoming_fights
from utils.data_loader import load_parquet
from utils.ui.sections import page_header


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
    return f"${value:,.{decimals}f}"


def _signed_money(value, decimals: int = 0) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.{decimals}f}"


def _pct(value, decimals: int = 1) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    if abs(value) <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


def _signed_pct(value, decimals: int = 1) -> str:
    value = _as_float(value)
    if value is None:
        return "—"
    if abs(value) <= 1:
        value *= 100
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _american(value) -> str:
    value = _as_float(value)
    if value is None or value == 0:
        return "—"
    value = int(round(value))
    return f"+{value}" if value > 0 else str(value)


def _decimal_odds(american_odds) -> float | None:
    odds = _as_float(american_odds)
    if odds is None or odds == 0:
        return None
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def _ev_for_100(probability, american_odds) -> float | None:
    p = _as_float(probability)
    odds = _as_float(american_odds)
    if p is None or odds is None or odds == 0:
        return None
    profit = odds if odds > 0 else 10000 / abs(odds)
    return p * profit - (1 - p) * 100


def _kelly_fraction(probability, american_odds) -> float:
    p = _as_float(probability)
    d = _decimal_odds(american_odds)
    if p is None or d is None or d <= 1:
        return 0.0
    b = d - 1.0
    return max(0.0, ((b * p) - (1.0 - p)) / b)


def _market_display(value) -> str:
    key = str(value or "").strip().lower()
    return {"moneyline": "Moneyline", "h2h": "Moneyline", "method": "Method of Victory", "goes_distance": "Goes Distance", "totals": "Totals", "round": "Round Props"}.get(key, str(value or "Unknown").replace("_", " ").title())


def _status(row: pd.Series, settings) -> str:
    if pd.isna(row.get("model_probability")) or pd.isna(row.get("american_odds")):
        return "NO ODDS"
    if _as_float(row.get("edge"), -999) < settings.min_edge:
        return "PASS / EDGE"
    if _as_float(row.get("confidence_pct"), -999) < settings.min_confidence:
        return "PASS / CONF"
    odds = _as_float(row.get("american_odds"), 0)
    if odds < settings.min_odds or odds > settings.max_odds:
        return "PASS / ODDS"
    return "BET CANDIDATE"


def _kelly_multiplier() -> float:
    if st.session_state.get("bb_kelly_mode") == "1/4 Kelly":
        return 0.25
    return load_risk_settings().kelly_fraction


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    settings = load_risk_settings()
    out = df.copy()
    out["market_display"] = out["market_key"].apply(_market_display)
    out["model_probability"] = pd.to_numeric(out["model_probability"], errors="coerce")
    out["american_odds"] = pd.to_numeric(out["american_odds"], errors="coerce")
    out["confidence_pct"] = pd.to_numeric(out["confidence_pct"], errors="coerce")
    if "implied_probability" not in out.columns:
        out["implied_probability"] = out["american_odds"].apply(lambda x: 1 / _decimal_odds(x) if _decimal_odds(x) else pd.NA)
    out["implied_probability"] = pd.to_numeric(out["implied_probability"], errors="coerce")
    out["edge"] = out["model_probability"] - out["implied_probability"]
    out["edge_pct"] = out["edge"] * 100
    out["ev_dollars_at_100"] = out.apply(lambda row: _ev_for_100(row.get("model_probability"), row.get("american_odds")), axis=1)
    out["full_kelly_fraction"] = out.apply(lambda row: _kelly_fraction(row.get("model_probability"), row.get("american_odds")), axis=1)
    out["recommended_stake"] = (settings.starting_bankroll * out["full_kelly_fraction"] * _kelly_multiplier()).clip(0, settings.starting_bankroll * settings.max_stake_pct)
    out["recommendation"] = out.apply(lambda row: _status(row, settings), axis=1)
    out["is_bet_candidate"] = out["recommendation"].eq("BET CANDIDATE")
    return out


def _event_options(events: pd.DataFrame, outcomes: pd.DataFrame) -> list[str]:
    names = []
    if events is not None and not events.empty:
        col = "ufcstats_event_name" if "ufcstats_event_name" in events.columns else "event_name" if "event_name" in events.columns else None
        if col:
            names.extend(events[col].dropna().astype(str).tolist())
    if outcomes is not None and not outcomes.empty and "event_name" in outcomes.columns:
        names.extend(outcomes["event_name"].dropna().astype(str).tolist())
    return ["All Events", *sorted(set(n for n in names if n.strip()))]


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    event = st.session_state.get("bb_filter_event", "All Events")
    if event != "All Events" and "event_name" in out.columns:
        out = out[out["event_name"].astype(str) == event]
    market = st.session_state.get("bb_filter_market_type", "All Markets")
    if market != "All Markets" and "market_display" in out.columns:
        out = out[out["market_display"].astype(str) == market]
    book = st.session_state.get("bb_filter_bookmaker", "All Books")
    if book != "All Books" and "bookmaker" in out.columns:
        out = out[out["bookmaker"].astype(str) == book]
    settings = load_risk_settings()
    lo, hi = st.session_state.get("bb_filter_odds_range", (settings.min_odds, settings.max_odds))
    out = out[pd.to_numeric(out["american_odds"], errors="coerce").between(lo, hi)]
    ev_threshold = float(st.session_state.get("bb_filter_ev_threshold", 50))
    if ev_threshold > 0:
        out = out[pd.to_numeric(out["ev_dollars_at_100"], errors="coerce") >= ev_threshold]
    elif st.session_state.get("bb_filter_positive_ev", False):
        out = out[pd.to_numeric(out["ev_dollars_at_100"], errors="coerce") > 0]
    min_conf = float(st.session_state.get("bb_filter_min_confidence", settings.min_confidence))
    out = out[pd.to_numeric(out["confidence_pct"], errors="coerce") >= min_conf]
    return out.reset_index(drop=True)


def _moneyline_rows(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    ml = outcomes[outcomes["market_key"].astype(str).str.lower().isin(["moneyline", "h2h"])].copy()
    if ml.empty:
        return pd.DataFrame()
    rows = []
    for _, group in ml.groupby(["fight_id", "bookmaker", "market_key"], dropna=False):
        first = group.iloc[0]
        red = group[group["outcome_fighter_id"].astype(str) == str(first.get("red_fighter_id"))]
        blue = group[group["outcome_fighter_id"].astype(str) == str(first.get("blue_fighter_id"))]
        if red.empty or blue.empty:
            continue
        red = red.iloc[0]
        blue = blue.iloc[0]
        best_pool = group[group["is_bet_candidate"]]
        best = (best_pool if not best_pool.empty else group).sort_values("ev_dollars_at_100", ascending=False).iloc[0]
        rows.append({
            "event_name": first.get("event_name"), "fight_id": first.get("fight_id"), "bookmaker": first.get("bookmaker"),
            "red_fighter": first.get("red_fighter"), "blue_fighter": first.get("blue_fighter"),
            "red_model_prob": red.get("model_probability"), "blue_model_prob": blue.get("model_probability"),
            "red_american_odds": red.get("american_odds"), "blue_american_odds": blue.get("american_odds"),
            "red_implied_prob": red.get("implied_probability"), "blue_implied_prob": blue.get("implied_probability"),
            "red_edge": red.get("edge"), "blue_edge": blue.get("edge"),
            "red_ev_dollars": red.get("ev_dollars_at_100"), "blue_ev_dollars": blue.get("ev_dollars_at_100"),
            "confidence_pct": best.get("confidence_pct"), "recommendation": best.get("recommendation"),
            "recommended_stake": best.get("recommended_stake"), "is_bet_candidate": bool(best.get("is_bet_candidate", False)),
            "sort_ev": best.get("ev_dollars_at_100"),
        })
    return pd.DataFrame(rows).sort_values("sort_ev", ascending=False, na_position="last").reset_index(drop=True) if rows else pd.DataFrame()


def _inject_css() -> None:
    st.markdown("""
    <style>
    .bb-card { background:linear-gradient(180deg, rgba(16,28,45,.92), rgba(13,23,39,.94)); border:1px solid rgba(38,54,74,.96); border-radius:10px; padding:.8rem; }
    .bb-grid { display:grid; grid-template-columns:.3fr 4.8fr .9fr 1.15fr 1.1fr; align-items:center; gap:.4rem; color:#f5f7fb; }
    .bb-head { color:#dbe7f5; font-size:.72rem; font-weight:900; text-transform:uppercase; border-bottom:1px solid rgba(38,54,74,.95); padding-bottom:.5rem; }
    .bb-row { border-bottom:1px solid rgba(38,54,74,.75); padding:.55rem 0; }
    .bb-fight { display:flex; flex-direction:column; gap:.35rem; }
    .bb-corner { display:grid; grid-template-columns:1.5rem 1.8fr .8fr .8fr .8fr .8fr .8fr; gap:.35rem; align-items:center; }
    .bb-badge { border-radius:4px; width:1.05rem; height:1.05rem; display:inline-flex; justify-content:center; align-items:center; color:#fff; font-size:.68rem; font-weight:900; }
    .bb-green { color:#35d96b; } .bb-red { color:#ef4444; } .bb-blue { color:#3b82f6; } .bb-yellow { color:#facc15; }
    .bb-ring { width:52px; height:52px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:auto; }
    .bb-ring div { background:#0d1727; width:39px; height:39px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#f5f7fb; font-size:.78rem; font-weight:900; }
    .bb-kpis { display:grid; grid-template-columns:repeat(6, 1fr); gap:.65rem; margin:.8rem 0; }
    .bb-kpi { text-align:center; min-height:92px; }
    .bb-label { color:#f5f7fb; font-size:.72rem; font-weight:800; text-transform:uppercase; }
    .bb-value { color:#35d96b; font-size:1.55rem; font-weight:900; margin-top:.35rem; }
    .bb-caption { color:#dbe7f5; font-size:.78rem; margin-top:.35rem; }
    </style>
    """, unsafe_allow_html=True)


def _ring(value) -> str:
    value = _as_float(value, 0) or 0
    color = "#35d96b" if value >= 70 else "#facc15" if value >= 60 else "#ef4444"
    return f'<div class="bb-ring" style="background: conic-gradient({color} {value:.0f}%, rgba(148,163,184,.24) 0);"><div>{value:.0f}%</div></div>'


def _corner(label, fighter, prob, odds, implied, edge, ev, color):
    edge_class = "bb-green" if (_as_float(edge, 0) or 0) >= 0 else "bb-red"
    ev_class = "bb-green" if (_as_float(ev, 0) or 0) >= 0 else "bb-red"
    return f'<div class="bb-corner"><span class="bb-badge" style="background:{color};">{label}</span><b>{_escape(fighter)}</b><span class="bb-green">{_pct(prob)}</span><span>{_american(odds)}</span><span>{_pct(implied)}</span><span class="{edge_class}">{_signed_pct(edge)}</span><span class="{ev_class}">{_signed_money(ev)}</span></div>'


def _kpi(label, value, caption):
    return f'<div class="bb-card bb-kpi"><div class="bb-label">{_escape(label)}</div><div class="bb-value">{_escape(value)}</div><div class="bb-caption">{_escape(caption)}</div></div>'


def _render_kpis(outcomes: pd.DataFrame):
    candidates = outcomes[outcomes["is_bet_candidate"]] if not outcomes.empty else outcomes
    total_fights = outcomes["fight_id"].nunique() if not outcomes.empty and "fight_id" in outcomes else 0
    positive = outcomes.loc[pd.to_numeric(outcomes.get("ev_dollars_at_100"), errors="coerce") > 0, "fight_id"].nunique() if not outcomes.empty and "fight_id" in outcomes else 0
    total_ev = candidates["ev_dollars_at_100"].sum() if not candidates.empty else 0
    stake = candidates["recommended_stake"].sum() if not candidates.empty else 0
    html = '<div class="bb-kpis">' + ''.join([
        _kpi("Total Fights", str(total_fights), "Visible market view"),
        _kpi("Positive EV", str(positive), "Fights with +EV outcome"),
        _kpi("Candidates", str(len(candidates)), "Pass risk filters"),
        _kpi("Candidate EV", _signed_money(total_ev), "At $100 stake"),
        _kpi("Avg Confidence", _pct(candidates["confidence_pct"].mean()) if not candidates.empty else "—", "Candidates"),
        _kpi("Risk", _money(stake), "Recommended stake"),
    ]) + '</div>'
    st.html(html)


def _render_table(rows: pd.DataFrame):
    if rows.empty:
        st.warning("No betting rows match this filter set.")
        return
    body = []
    for i, row in rows.iterrows():
        rec = row.get("recommendation", "PASS / EDGE")
        rec_class = "bb-green" if rec == "BET CANDIDATE" else "bb-yellow" if rec == "PASS / CONF" else "bb-red" if rec == "NO ODDS" else ""
        stake = _money(row.get("recommended_stake")) if bool(row.get("is_bet_candidate")) else "—"
        body.append('<div class="bb-grid bb-row">'
            f'<span>{i+1}</span><div class="bb-fight">'
            + _corner("R", row.get("red_fighter"), row.get("red_model_prob"), row.get("red_american_odds"), row.get("red_implied_prob"), row.get("red_edge"), row.get("red_ev_dollars"), "#ef4444")
            + _corner("B", row.get("blue_fighter"), row.get("blue_model_prob"), row.get("blue_american_odds"), row.get("blue_implied_prob"), row.get("blue_edge"), row.get("blue_ev_dollars"), "#3b82f6")
            + f'</div><div>{_ring(row.get("confidence_pct"))}</div><b class="{rec_class}">{_escape(rec)}</b><b>{stake}</b></div>')
    head = '<div class="bb-grid bb-head"><span></span><div>Fight / Market</div><span>Confidence</span><span>Status</span><span>Stake</span></div>'
    st.html('<div class="bb-card">' + head + ''.join(body) + '</div>')


def render_betting_board():
    _inject_css()
    page_header("Betting Board", "Live fight predictions and betting opportunities")
    outcomes = load_parquet(BETTING_OUTCOMES_PATH)
    events, _ = load_upcoming_events()
    load_upcoming_fights()
    if outcomes.empty:
        st.warning(f"No betting outcomes found at `{BETTING_OUTCOMES_PATH}`. Run `python -m pipeline.betting.run_betting_outcomes_v2`.")
        return
    updated = None
    for col in ["betting_timestamp", "snapshot_timestamp", "prediction_timestamp"]:
        if col in outcomes.columns:
            parsed = pd.to_datetime(outcomes[col], errors="coerce", utc=True).dropna()
            if not parsed.empty:
                updated = parsed.max().tz_convert(ZoneInfo("America/Chicago")).strftime("%b %-d, %Y %I:%M %p %Z")
                break
    if updated:
        st.caption(f"Last Updated: {updated}")
    enriched = _enrich(outcomes)
    filtered = _apply_filters(enriched)
    _render_kpis(filtered)
    _render_table(_moneyline_rows(filtered))
