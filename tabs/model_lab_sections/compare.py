from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]

BETTING_OUTCOMES_PATH = Path("data/predictions/betting_outcomes.parquet")
BETTING_OUTCOMES_AUDIT_PATH = Path("data/audits/ufc_betting_outcomes_audit.parquet")

STRONG_THRESHOLD = 75.0
LEAN_THRESHOLD = 55.0


@st.cache_data(show_spinner=False)
def _read_parquet(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _fmt_pct(value: Any, decimals: int = 0) -> str:
    try:
        number = float(value)
    except Exception:
        return "—"
    if abs(number) <= 1:
        number *= 100.0
    return f"{number:.{decimals}f}%"


def _fmt_odds(value: Any) -> str:
    try:
        number = int(round(float(value)))
    except Exception:
        return "—"
    return f"+{number}" if number > 0 else str(number)


def _status_label(agreement_pct: float, spread_pct: float | None = None) -> str:
    if spread_pct is not None and spread_pct >= 20:
        return "Outlier"
    if agreement_pct >= STRONG_THRESHOLD:
        return "Strong Consensus"
    if agreement_pct >= LEAN_THRESHOLD:
        return "Lean Consensus"
    return "Split"


def _status_icon(status: str) -> str:
    if status == "Strong Consensus":
        return "🟢"
    if status == "Lean Consensus":
        return "🟡"
    if status == "Outlier":
        return "🟣"
    return "🔴"


def _latest_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "betting_run_id" not in df.columns:
        return df
    if "betting_timestamp" in df.columns:
        ranked = df[["betting_run_id", "betting_timestamp"]].dropna().drop_duplicates()
        if not ranked.empty:
            ranked["betting_timestamp"] = pd.to_datetime(ranked["betting_timestamp"], errors="coerce", utc=True)
            latest = ranked.sort_values("betting_timestamp").tail(1)["betting_run_id"].iloc[0]
            return df[df["betting_run_id"].astype(str) == str(latest)].copy()
    latest = df["betting_run_id"].dropna().astype(str).iloc[-1]
    return df[df["betting_run_id"].astype(str) == latest].copy()


def _model_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    if df.empty or "model_id" not in df.columns:
        return 0, 0, 0
    if "model_registry_status" in df.columns:
        models = df[["model_id", "model_registry_status"]].drop_duplicates()
        total = int(models["model_id"].nunique())
        statuses = models["model_registry_status"].astype(str).str.lower()
        return total, int((statuses == "production").sum()), int((statuses == "draft").sum())
    return int(df["model_id"].nunique()), 0, 0


def _best_market_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ev_dollars_at_100" in out.columns:
        out["_ev_sort"] = pd.to_numeric(out["ev_dollars_at_100"], errors="coerce")
    sort_cols = [col for col in ["fight_id", "market_key", "outcome_join_key", "_ev_sort"] if col in out.columns]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if sort_cols[-1] == "_ev_sort":
            ascending[-1] = False
        out = out.sort_values(sort_cols, ascending=ascending)
    dedupe = [col for col in ["fight_id", "market_key", "outcome_join_key", "model_id"] if col in out.columns]
    if dedupe:
        out = out.drop_duplicates(dedupe, keep="first")
    return out.drop(columns=["_ev_sort"], errors="ignore")


def _build_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"fight_id", "market_key", "model_id", "outcome_label", "model_probability"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    source = _best_market_by_outcome(df)
    rows: list[dict[str, Any]] = []
    group_keys = ["event_name", "fight_id", "fight_display", "market_key"]
    for key_values, group in source.groupby(group_keys, dropna=False):
        event_name, fight_id, fight_display, market_key = key_values
        model_ids = sorted([str(x) for x in group["model_id"].dropna().unique()])
        models_included = len(model_ids)

        if "is_model_pick" in group.columns:
            model_pick_rows = group[group["is_model_pick"].fillna(False).astype(bool)]
        else:
            model_pick_rows = pd.DataFrame()
        if model_pick_rows.empty:
            model_pick_rows = group.sort_values("model_probability", ascending=False).drop_duplicates("model_id")

        pick_counts = model_pick_rows.groupby("outcome_label")["model_id"].nunique().sort_values(ascending=False)
        consensus_pick = str(pick_counts.index[0]) if not pick_counts.empty else "—"
        support = int(pick_counts.iloc[0]) if not pick_counts.empty else 0
        agreement_pct = (support / models_included * 100.0) if models_included else 0.0

        probabilities = pd.to_numeric(group["model_probability"], errors="coerce")
        avg_prob = float(probabilities.mean()) if not probabilities.dropna().empty else 0.0
        spread_pct = float((probabilities.max() - probabilities.min()) * 100.0) if len(probabilities.dropna()) else 0.0

        ev_values = pd.to_numeric(group.get("ev_dollars_at_100", pd.Series(dtype=float)), errors="coerce")
        avg_ev = float(ev_values.mean()) if not ev_values.dropna().empty else 0.0
        max_ev = float(ev_values.max()) if not ev_values.dropna().empty else 0.0

        implied = pd.to_numeric(group.get("implied_probability", pd.Series(dtype=float)), errors="coerce")
        implied_display = "—"
        if not implied.dropna().empty:
            implied_display = f"{_fmt_pct(implied.min())} / {_fmt_pct(implied.max())}"

        odds = pd.to_numeric(group.get("american_odds", pd.Series(dtype=float)), errors="coerce")
        odds_display = "—" if odds.dropna().empty else " / ".join(_fmt_odds(x) for x in sorted(odds.dropna().unique())[:3])

        prod_pick = "—"
        if "model_registry_status" in model_pick_rows.columns:
            prod_rows = model_pick_rows[model_pick_rows["model_registry_status"].astype(str).str.lower() == "production"]
            if not prod_rows.empty:
                prod_pick = str(prod_rows.sort_values("model_probability", ascending=False)["outcome_label"].iloc[0])

        status = _status_label(agreement_pct, spread_pct)
        row: dict[str, Any] = {
            "event_name": event_name,
            "fight_id": str(fight_id),
            "fight_display": fight_display,
            "market_key": market_key,
            "Fight": fight_display,
            "Market": market_key,
            "Odds (Best)": odds_display,
            "Implied Prob (%)": implied_display,
            "Consensus Pick": consensus_pick,
            "Agreement %": round(agreement_pct, 1),
            "Agreement": f"{_status_icon(status)} {_fmt_pct(agreement_pct)}",
            "Avg Prob (%)": round(avg_prob * 100.0, 1),
            "Prob Spread (%)": round(spread_pct, 1),
            "Avg EV ($100 stake)": round(avg_ev, 2),
            "Max EV ($100 stake)": round(max_ev, 2),
            "Production Pick": prod_pick,
            "Status": status,
            "Models Included": models_included,
        }
        for model_id in model_ids:
            model_group = group[group["model_id"].astype(str) == model_id]
            if model_group.empty:
                continue
            pick_row = model_group.sort_values("model_probability", ascending=False).iloc[0]
            model_prob = pd.to_numeric(pd.Series([pick_row.get("model_probability")]), errors="coerce").iloc[0]
            row[f"{model_id} Prob (%)"] = round(float(model_prob) * 100.0, 1) if pd.notna(model_prob) else pd.NA
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["event_name", "fight_display", "market_key"]).reset_index(drop=True)


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    filter_cols = st.columns([1.55, 1.0, 1.0, 1.22, 0.9, 0.95])

    event_options = ["All Events"] + sorted([str(x) for x in out.get("event_name", pd.Series(dtype=str)).dropna().unique()])
    with filter_cols[0]:
        selected_event = st.selectbox("Event", event_options, key="compare_event")
    if selected_event != "All Events" and "event_name" in out.columns:
        out = out[out["event_name"].astype(str) == selected_event]

    market_options = ["All Markets"] + sorted([str(x) for x in out.get("market_key", pd.Series(dtype=str)).dropna().unique()])
    with filter_cols[1]:
        selected_market = st.selectbox("Market Type", market_options, key="compare_market")
    if selected_market != "All Markets" and "market_key" in out.columns:
        out = out[out["market_key"].astype(str) == selected_market]

    with filter_cols[2]:
        model_set = st.selectbox("Model Set", ["All Models", "Production Only", "Draft Only"], key="compare_model_set")
    if "model_registry_status" in out.columns:
        status_series = out["model_registry_status"].astype(str).str.lower()
        if model_set == "Production Only":
            out = out[status_series == "production"]
        elif model_set == "Draft Only":
            out = out[status_series == "draft"]

    with filter_cols[3]:
        model_status = st.selectbox("Model Status", ["All", "production", "draft", "single"], key="compare_model_status")
    if model_status != "All" and "model_registry_status" in out.columns:
        out = out[out["model_registry_status"].astype(str).str.lower() == model_status]

    with filter_cols[4]:
        min_ev = st.number_input("Minimum EV ($)", value=0.0, step=1.0, key="compare_min_ev")
    if "ev_dollars_at_100" in out.columns:
        out = out[pd.to_numeric(out["ev_dollars_at_100"], errors="coerce").fillna(-10**9) >= float(min_ev)]

    with filter_cols[5]:
        min_agreement = st.number_input("Minimum Agreement (%)", value=0.0, min_value=0.0, max_value=100.0, step=5.0, key="compare_min_agreement")

    toggle_cols = st.columns([1.1, 1.1, 1.2, 3.2])
    with toggle_cols[0]:
        show_disagreements = st.toggle("Show Disagreements Only", value=False, key="compare_disagreements_only")
    with toggle_cols[1]:
        show_bets = st.toggle("Show Suggested Bets Only", value=False, key="compare_bets_only")
    with toggle_cols[2]:
        hide_incomplete = st.toggle("Hide Fights Without All Models", value=False, key="compare_hide_incomplete")

    overview = _build_overview_table(out)
    if overview.empty:
        return out

    filtered_overview = overview[pd.to_numeric(overview["Agreement %"], errors="coerce").fillna(0) >= float(min_agreement)].copy()
    if show_disagreements:
        filtered_overview = filtered_overview[filtered_overview["Status"].isin(["Split", "Outlier"])]
    if show_bets and "is_bet_candidate" in out.columns:
        bet_keys = out.loc[out["is_bet_candidate"].fillna(False).astype(bool), ["fight_id", "market_key"]].drop_duplicates()
        filtered_overview = filtered_overview.merge(bet_keys, on=["fight_id", "market_key"], how="inner")
    if hide_incomplete and "Models Included" in filtered_overview.columns:
        expected_models = max(1, _model_counts(df)[0])
        filtered_overview = filtered_overview[pd.to_numeric(filtered_overview["Models Included"], errors="coerce").fillna(0) >= expected_models]

    valid_keys = set(zip(filtered_overview["fight_id"].astype(str), filtered_overview["market_key"].astype(str)))
    return out[out.apply(lambda row: (str(row.get("fight_id")), str(row.get("market_key"))) in valid_keys, axis=1)]


def _build_suggested_bets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_bet_candidate" not in df.columns:
        return pd.DataFrame()
    bets = df[df["is_bet_candidate"].fillna(False).astype(bool)].copy()
    if bets.empty:
        return pd.DataFrame()
    support = bets.groupby(["fight_id", "market_key", "outcome_join_key"])["model_id"].nunique().rename("Models Supporting")
    bets = bets.join(support, on=["fight_id", "market_key", "outcome_join_key"])
    if "model_probability" in bets.columns:
        bets["Model Prob (%)"] = pd.to_numeric(bets["model_probability"], errors="coerce") * 100.0
    if "implied_probability" in bets.columns:
        bets["Implied Prob (%)"] = pd.to_numeric(bets["implied_probability"], errors="coerce") * 100.0
    display_cols = [
        "outcome_display",
        "bookmaker",
        "american_odds",
        "Implied Prob (%)",
        "model_id",
        "Models Supporting",
        "Model Prob (%)",
        "ev_dollars_at_100",
        "recommended_stake",
        "bet_status",
    ]
    out = bets[[col for col in display_cols if col in bets.columns]].copy()
    rename = {
        "outcome_display": "Bet (Outcome)",
        "bookmaker": "Book (Best)",
        "american_odds": "Odds",
        "model_id": "Model",
        "ev_dollars_at_100": "Avg EV ($100)",
        "recommended_stake": "Recommended Stake",
        "bet_status": "Status",
    }
    out = out.rename(columns=rename)
    if "Avg EV ($100)" in out.columns:
        out = out.sort_values("Avg EV ($100)", ascending=False)
    return out.reset_index(drop=True)


def _build_disagreements(overview: pd.DataFrame) -> pd.DataFrame:
    if overview.empty:
        return pd.DataFrame()
    disagreements = overview[overview["Status"].isin(["Split", "Outlier"])].copy()
    if disagreements.empty:
        return pd.DataFrame()
    keep = [
        "Fight",
        "Market",
        "Consensus Pick",
        "Agreement",
        "Avg Prob (%)",
        "Prob Spread (%)",
        "Avg EV ($100 stake)",
        "Max EV ($100 stake)",
        "Status",
    ]
    return disagreements[[col for col in keep if col in disagreements.columns]].reset_index(drop=True)


def _render_summary_cards(overview: pd.DataFrame, df: pd.DataFrame) -> None:
    total_models, prod_models, draft_models = _model_counts(df)
    total_markets = len(overview) if not overview.empty else 0
    latest_timestamp = "—"
    if not df.empty and "betting_timestamp" in df.columns:
        timestamps = pd.to_datetime(df["betting_timestamp"], errors="coerce", utc=True).dropna()
        if not timestamps.empty:
            latest_timestamp = timestamps.max().strftime("%Y-%m-%d %H:%M:%S UTC")

    c1, c2, c3, c4, c5 = st.columns([1.45, 0.75, 0.75, 0.75, 0.9])
    c1.caption(f"Last Outcomes Run: {latest_timestamp}")
    c2.metric("Models", total_models)
    c3.metric("Production", prod_models)
    c4.metric("Draft", draft_models)
    c5.metric("Markets", total_markets)


def _render_consensus_summary(overview: pd.DataFrame) -> None:
    if overview.empty:
        st.info("No consensus rows available for the selected filters.")
        return
    counts = overview["Status"].value_counts()
    total = int(counts.sum())
    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    with c1:
        st.metric("Strong Consensus", int(counts.get("Strong Consensus", 0)))
        st.metric("Lean Consensus", int(counts.get("Lean Consensus", 0)))
    with c2:
        st.metric("Split", int(counts.get("Split", 0)))
        st.metric("Outlier", int(counts.get("Outlier", 0)))
    with c3:
        st.metric("Average Agreement", _fmt_pct(pd.to_numeric(overview["Agreement %"], errors="coerce").mean()))
        st.metric("Average Prob Spread", _fmt_pct(pd.to_numeric(overview["Prob Spread (%)"], errors="coerce").mean()))
        st.caption(f"Total markets: {total}")


def _render_footer() -> None:
    st.caption(
        "Legend: Strong Consensus = 75%+ model agreement · Lean Consensus = 55–74% · "
        "Split = below 55% · Outlier = probability spread of 20+ points. "
        "EV is calculated from betting_outcomes.parquet at $100 stake."
    )


def render_compare(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    """Render upcoming event prediction comparison from Betting Outcomes V2."""

    st.markdown("## Upcoming Event Prediction Compare")
    st.caption("Compare model predictions, probabilities, agreement, EV, and suggested bets for upcoming events.")

    raw = _read_parquet(str(BETTING_OUTCOMES_PATH))
    audit = _read_parquet(str(BETTING_OUTCOMES_AUDIT_PATH))

    if raw.empty:
        st.warning(
            "No betting outcomes artifact found. Run Model Lab → Actions → Run Outcomes to populate "
            "data/predictions/betting_outcomes.parquet."
        )
        return

    latest = _latest_run(raw)
    _render_summary_cards(_build_overview_table(latest), latest)

    if not audit.empty:
        latest_audit = audit.tail(1)
        if "passes_validation" in latest_audit.columns and not bool(latest_audit["passes_validation"].iloc[0]):
            st.warning("Latest betting outcomes audit did not pass validation. Review join-key diagnostics before relying on this board.")

    st.divider()
    filtered = _apply_filters(latest)
    overview = _build_overview_table(filtered)
    suggested = _build_suggested_bets(filtered)
    disagreements = _build_disagreements(overview)

    tabs = st.tabs([
        "Overview",
        "Suggested Bets",
        "Disagreements",
        "Model Probabilities",
        "Model EV Comparison",
        "Consensus Summary",
    ])

    with tabs[0]:
        st.markdown("#### Fight / Market Comparison")
        if overview.empty:
            st.info("No rows match the selected filters.")
        else:
            hidden = ["event_name", "fight_id", "fight_display", "market_key", "Agreement %", "Models Included"]
            st.dataframe(overview.drop(columns=[col for col in hidden if col in overview.columns]), use_container_width=True, hide_index=True)

        c1, c2 = st.columns([1.1, 1.0], gap="medium")
        with c1:
            st.markdown("#### Suggested Bets Comparison")
            if suggested.empty:
                st.info("No suggested bets match the selected filters.")
            else:
                st.dataframe(suggested.head(10), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Disagreement Board")
            if disagreements.empty:
                st.success("No major model disagreements for the selected filters.")
            else:
                st.dataframe(disagreements.head(10), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("#### Suggested Bets")
        if suggested.empty:
            st.info("No suggested bets match the selected filters.")
        else:
            st.dataframe(suggested, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("#### Disagreements")
        if disagreements.empty:
            st.success("No split or outlier markets match the selected filters.")
        else:
            st.dataframe(disagreements, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("#### Model Probabilities")
        cols = [
            "event_name",
            "fight_display",
            "market_display",
            "outcome_display",
            "model_id",
            "model_registry_status",
            "model_probability",
            "model_confidence",
            "confidence_pct",
            "is_model_pick",
        ]
        display = filtered[[col for col in cols if col in filtered.columns]].copy()
        if "model_probability" in display.columns:
            display["model_probability"] = pd.to_numeric(display["model_probability"], errors="coerce") * 100.0
        st.dataframe(display, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown("#### Model EV Comparison")
        cols = [
            "event_name",
            "fight_display",
            "market_display",
            "outcome_display",
            "model_id",
            "bookmaker",
            "american_odds",
            "implied_probability",
            "edge_pct",
            "ev_dollars_at_100",
            "recommended_stake",
            "bet_status",
        ]
        display = filtered[[col for col in cols if col in filtered.columns]].copy()
        if "implied_probability" in display.columns:
            display["implied_probability"] = pd.to_numeric(display["implied_probability"], errors="coerce") * 100.0
        st.dataframe(display, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.markdown("#### Model Consensus Summary")
        _render_consensus_summary(overview)

    st.divider()
    _render_footer()
