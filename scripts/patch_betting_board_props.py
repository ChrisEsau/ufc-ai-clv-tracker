from pathlib import Path

p = Path("tabs/betting_board_v2.py")
s = p.read_text()

# Expand market display mapping so existing sidebar Market Type filter recognizes prop families.
s = s.replace(
'''        "goes_distance": "Goes Distance",
        "totals": "Totals",
        "round": "Round Props",
''',
'''        "goes_distance": "Goes Distance",
        "total_rounds": "Totals",
        "totals": "Totals",
        "round": "Round Props",
        "round_method": "Round Props",
        "exact_method": "Method of Victory",
        "win_by_ko_tko_dq": "Method of Victory",
        "win_by_submission": "Method of Victory",
        "win_by_decision": "Method of Victory",
''',
)

# Add prop outcome display helper.
marker = '''def _status(row: pd.Series, settings) -> str:\n'''
insert = r'''def _outcome_display(row: pd.Series) -> str:
    market_key = str(row.get("market_key") or "").strip().lower()
    label = str(row.get("outcome_label") or row.get("outcome_key") or row.get("side") or "").strip()
    side = str(row.get("side") or "").strip().lower()
    line = row.get("line")

    if market_key == "goes_distance":
        return "Goes Distance" if label == "goes_distance" or side == "yes" else "Inside Distance"
    if market_key in {"total_rounds", "totals"}:
        if side in {"over", "under"} and not pd.isna(line):
            return f"{side.title()} {float(line):g} Rounds"
    if market_key == "exact_method":
        return label.replace("_", " ").title()
    if market_key.startswith("win_by_"):
        fighter = row.get("outcome_fighter_name") or label
        method = market_key.replace("win_by_", "").replace("ko_tko_dq", "KO/TKO/DQ").replace("_", " ").title()
        return f"{fighter} by {method}"
    if market_key == "round_method":
        return str(row.get("provider_selection_name") or label).replace("_", " ").title()
    return label.replace("_", " ").title() if label else "—"


'''
if "def _outcome_display(" not in s:
    s = s.replace(marker, insert + marker)

# Make base filters honor existing sidebar odds/missing-odds controls for every market.
old = '''    book = st.session_state.get("bb_filter_bookmaker", "All Books")
    if book != "All Books" and "bookmaker" in out.columns:
        out = out[out["bookmaker"].astype(str) == book]
    return out.reset_index(drop=True)
'''
new = '''    book = st.session_state.get("bb_filter_bookmaker", "All Books")
    if book != "All Books" and "bookmaker" in out.columns:
        out = out[out["bookmaker"].astype(str) == book]
    odds_range = st.session_state.get("bb_filter_odds_range")
    if odds_range and "american_odds" in out.columns:
        low, high = odds_range
        odds = pd.to_numeric(out["american_odds"], errors="coerce")
        out = out[odds.between(low, high)]
    if st.session_state.get("bb_filter_hide_missing_odds", True) and "american_odds" in out.columns:
        out = out[pd.to_numeric(out["american_odds"], errors="coerce").notna()]
    return out.reset_index(drop=True)
'''
s = s.replace(old, new)

s = s.replace(
    '"""Apply EV/positive/confidence filters to grouped fight rows, not individual outcomes."""',
    '"""Apply EV/positive/confidence filters to display rows."""',
)

# Mark existing moneyline rows with row type and market display.
s = s.replace(
'''        rows.append({
            "event_name": first.get("event_name"),
''',
'''        rows.append({
            "row_type": "moneyline",
            "event_name": first.get("event_name"),
''',
)
s = s.replace(
'''            "bookmaker": first.get("bookmaker"),
            "red_fighter": first.get("red_fighter"),
''',
'''            "bookmaker": first.get("bookmaker"),
            "market_display": "Moneyline",
            "red_fighter": first.get("red_fighter"),
''',
)

# Add prop row builder and universal display row router before CSS.
marker = '''def _inject_css() -> None:\n'''
insert = r'''def _prop_rows(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    props = outcomes[~outcomes["market_key"].astype(str).str.lower().isin(["moneyline", "h2h"])].copy()
    if props.empty:
        return pd.DataFrame()

    rows = []
    for _, row in props.iterrows():
        rows.append({
            "row_type": "prop",
            "event_name": row.get("event_name"),
            "fight_id": row.get("fight_id"),
            "bookmaker": row.get("bookmaker"),
            "market_display": row.get("market_display"),
            "red_fighter": row.get("red_fighter"),
            "blue_fighter": row.get("blue_fighter"),
            "prop_outcome": _outcome_display(row),
            "model_probability": row.get("model_probability"),
            "american_odds": row.get("american_odds"),
            "implied_probability": row.get("implied_probability"),
            "edge": row.get("edge"),
            "ev_dollars_at_100": row.get("ev_dollars_at_100"),
            "confidence_pct": row.get("confidence_pct"),
            "recommendation": row.get("recommendation"),
            "recommended_stake": row.get("recommended_stake"),
            "is_bet_candidate": bool(row.get("is_bet_candidate", False)),
            "sort_ev": row.get("ev_dollars_at_100"),
        })

    return pd.DataFrame(rows).sort_values("sort_ev", ascending=False, na_position="last").reset_index(drop=True)


def _display_rows(outcomes: pd.DataFrame) -> pd.DataFrame:
    selected_market = st.session_state.get("bb_filter_market_type", "All Markets")
    if selected_market == "Moneyline":
        return _moneyline_rows(outcomes)
    if selected_market == "All Markets":
        return pd.concat([_moneyline_rows(outcomes), _prop_rows(outcomes)], ignore_index=True)
    return _prop_rows(outcomes)


'''
if "def _prop_rows(" not in s:
    s = s.replace(marker, insert + marker)

# Add CSS for prop rows.
s = s.replace(
'''    .bb-corner { display:grid; grid-template-columns:1.5rem 1.8fr .8fr .8fr .8fr .8fr .8fr; gap:.35rem; align-items:center; }
''',
'''    .bb-corner { display:grid; grid-template-columns:1.5rem 1.8fr .8fr .8fr .8fr .8fr .8fr; gap:.35rem; align-items:center; }
    .bb-prop { display:grid; grid-template-columns:1.5rem 1.7fr 1.2fr .8fr .8fr .8fr .8fr .8fr; gap:.35rem; align-items:center; }
    .bb-subtle { color:#dbe7f5; font-size:.76rem; }
''',
)

# Add prop row HTML renderer after _corner.
marker = '''def _kpi(label, value, caption):\n'''
insert = r'''def _prop_line(row: pd.Series) -> str:
    edge_class = "bb-green" if (_as_float(row.get("edge"), 0) or 0) >= 0 else "bb-red"
    ev_class = "bb-green" if (_as_float(row.get("ev_dollars_at_100"), 0) or 0) >= 0 else "bb-red"
    return (
        '<div class="bb-prop">'
        '<span class="bb-badge" style="background:#8b5cf6;">P</span>'
        f'<div><b>{_escape(row.get("market_display"))}</b><div class="bb-subtle">{_escape(row.get("red_fighter"))} vs {_escape(row.get("blue_fighter"))}</div></div>'
        f'<b>{_escape(row.get("prop_outcome"))}</b>'
        f'<span class="bb-green">{_pct(row.get("model_probability"))}</span>'
        f'<span>{_american(row.get("american_odds"))}</span>'
        f'<span>{_pct(row.get("implied_probability"))}</span>'
        f'<span class="{edge_class}">{_signed_pct(row.get("edge"))}</span>'
        f'<span class="{ev_class}">{_signed_money(row.get("ev_dollars_at_100"))}</span>'
        '</div>'
    )


'''
if "def _prop_line(" not in s:
    s = s.replace(marker, insert + marker)

# Route existing table rows by row_type.
old = '''        body.append('<div class="bb-grid bb-row">'
            f'<span>{i+1}</span><div class="bb-fight">'
            + _corner("R", row.get("red_fighter"), row.get("red_model_prob"), row.get("red_american_odds"), row.get("red_implied_prob"), row.get("red_edge"), row.get("red_ev_dollars"), "#ef4444")
            + _corner("B", row.get("blue_fighter"), row.get("blue_model_prob"), row.get("blue_american_odds"), row.get("blue_implied_prob"), row.get("blue_edge"), row.get("blue_ev_dollars"), "#3b82f6")
            + f'</div><div>{_ring(row.get("confidence_pct"))}</div><b class="{rec_class}">{_escape(rec)}</b><b>{stake}</b></div>')
'''
new = '''        if row.get("row_type") == "prop":
            fight_html = _prop_line(row)
        else:
            fight_html = (
                _corner("R", row.get("red_fighter"), row.get("red_model_prob"), row.get("red_american_odds"), row.get("red_implied_prob"), row.get("red_edge"), row.get("red_ev_dollars"), "#ef4444")
                + _corner("B", row.get("blue_fighter"), row.get("blue_model_prob"), row.get("blue_american_odds"), row.get("blue_implied_prob"), row.get("blue_edge"), row.get("blue_ev_dollars"), "#3b82f6")
            )
        body.append('<div class="bb-grid bb-row">'
            f'<span>{i+1}</span><div class="bb-fight">'
            + fight_html
            + f'</div><div>{_ring(row.get("confidence_pct"))}</div><b class="{rec_class}">{_escape(rec)}</b><b>{stake}</b></div>')
'''
s = s.replace(old, new)

# Use existing table with row source selected by sidebar Market Type.
s = s.replace(
'''    moneyline_rows = _moneyline_rows(dimension_filtered)
    display_rows = _display_filter_best_rows(moneyline_rows)
''',
'''    display_rows = _display_filter_best_rows(_display_rows(dimension_filtered))
''',
)

p.write_text(s)
print(f"Patched {p}")
