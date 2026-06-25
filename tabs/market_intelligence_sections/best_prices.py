from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from pipeline.common.paths import MARKET_OUTCOMES_PATH
from utils.data_loader import load_parquet


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _fmt_odds(value) -> str:
    try:
        return f"{int(float(value)):+d}"
    except Exception:
        return "—"


def _book(value) -> str:
    value = str(value or "").strip()
    if not value:
        return "—"
    short = {
        "DraftKings": "DK",
        "FanDuel": "FD",
        "BetMGM": "MGM",
        "Caesars": "CZR",
    }.get(value, value)
    return short


def render_best_prices() -> None:
    outcomes = load_parquet(MARKET_OUTCOMES_PATH)

    if outcomes is None or outcomes.empty:
        st.html(
            '<div class="mi-card mi-panel">'
            '<div class="mi-panel-head"><div><div class="mi-section-title">Best Line Finder</div>'
            '<div class="mi-section-subtitle">Best available sportsbook price by canonical outcome.</div></div></div>'
            '<div class="mi-placeholder">No market outcomes available.</div>'
            '</div>'
        )
        return

    df = outcomes.copy()
    if "comparison_key" not in df.columns:
        if "outcome_join_key" in df.columns:
            df["comparison_key"] = df["outcome_join_key"]
        elif "provider_selection_name" in df.columns:
            df["comparison_key"] = df["provider_selection_name"].astype(str)
        else:
            df["comparison_key"] = df.index.astype(str)

    df["american_odds"] = pd.to_numeric(df.get("american_odds"), errors="coerce")
    key_cols = ["fight_id", "market_key", "comparison_key"]
    usable = df.dropna(subset=key_cols + ["american_odds"]).copy()

    rows = []
    for _, group in usable.groupby(key_cols, dropna=False):
        best = group.sort_values("american_odds", ascending=False).iloc[0]
        worst = group.sort_values("american_odds", ascending=True).iloc[0]
        diff = float(best.get("american_odds")) - float(worst.get("american_odds"))
        rows.append(
            {
                "fight": f"{best.get('red_fighter')} vs {best.get('blue_fighter')}",
                "market": str(best.get("market_key") or "").replace("_", " ").title(),
                "outcome": best.get("outcome_fighter_name") or best.get("provider_selection_name") or best.get("side"),
                "best": _fmt_odds(best.get("american_odds")),
                "best_book": _book(best.get("bookmaker")),
                "worst": _fmt_odds(worst.get("american_odds")),
                "worst_book": _book(worst.get("bookmaker")),
                "diff": diff,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("diff", ascending=False).head(10)

    table_rows = ""
    for _, row in out.iterrows():
        table_rows += (
            "<tr>"
            f"<td><b>{_escape(row['fight'])}</b><br><span>{_escape(row['market'])}</span></td>"
            f"<td>{_escape(row['outcome'])}</td>"
            f"<td class='mi-green'><b>{_escape(row['best'])}</b> <span class='mi-book'>{_escape(row['best_book'])}</span></td>"
            f"<td class='mi-red'><b>{_escape(row['worst'])}</b> <span class='mi-book'>{_escape(row['worst_book'])}</span></td>"
            f"<td class='mi-green'><b>{row['diff']:.0f}¢</b></td>"
            "</tr>"
        )

    st.html(
        '<div class="mi-card mi-panel">'
        '<div class="mi-panel-head"><div><div class="mi-section-title">Best Line Finder</div>'
        '<div class="mi-section-subtitle">Best available sportsbook price by canonical outcome.</div></div>'
        '<div class="mi-panel-link">View full table →</div></div>'
        '<div class="mi-body">'
        '<table class="mi-table">'
        '<thead><tr><th>Fight / Market</th><th>Outcome</th><th>Best</th><th>Worst</th><th>Diff</th></tr></thead>'
        f'<tbody>{table_rows}</tbody>'
        '</table>'
        '</div></div>'
    )
