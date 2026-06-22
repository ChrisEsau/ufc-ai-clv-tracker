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
OUTLIER_SPREAD_THRESHOLD = 25.0


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


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "—"


def _clean_text(value: Any, default: str = "—") -> str:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"nan", "none", "<na>"} else default


def _latest_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "betting_run_id" not in df.columns:
        return df.copy()

    run_ids = df["betting_run_id"].dropna().astype(str)
    run_ids = run_ids[~run_ids.str.strip().isin(["", "nan", "None", "<NA>"])]
    if run_ids.empty:
        return df.copy()

    if "betting_timestamp" in df.columns:
        ranked = df[["betting_run_id", "betting_timestamp"]].copy()
        ranked["betting_run_id"] = ranked["betting_run_id"].astype(str)
        ranked["betting_timestamp"] = pd.to_datetime(ranked["betting_timestamp"], errors="coerce", utc=True)
        ranked = ranked[ranked["betting_run_id"].isin(run_ids.unique())].dropna(subset=["betting_timestamp"])
        if not ranked.empty:
            latest = ranked.sort_values("betting_timestamp").tail(1)["betting_run_id"].iloc[0]
            return df[df["betting_run_id"].astype(str) == str(latest)].copy()

    latest = run_ids.iloc[-1]
    return df[df["betting_run_id"].astype(str) == latest].copy()


def _registry_status_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {str(row.get("model_id")): str(row.get("status") or "").lower() for row in rows if row.get("model_id")}


def _enrich_status(df: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    if "model_registry_status" not in out.columns:
        out["model_registry_status"] = pd.NA
    status_map = _registry_status_map(rows)
    mapped = out["model_id"].astype(str).map(status_map) if "model_id" in out.columns else pd.Series(dtype=str)
    out["model_registry_status"] = out["model_registry_status"].fillna(mapped)
    out["model_registry_status"] = out["model_registry_status"].astype(str).str.lower().replace({"nan": "", "none": "", "<na>": ""})
    out.loc[out["model_registry_status"].eq(""), "model_registry_status"] = mapped
    return out


def _model_order(df: pd.DataFrame) -> list[str]:
    if df.empty or "model_id" not in df.columns:
        return []
    model_status = df[["model_id", "model_registry_status"]].drop_duplicates() if "model_registry_status" in df.columns else df[["model_id"]].drop_duplicates()

    def sort_key(row: pd.Series) -> tuple[int, str]:
        status = str(row.get("model_registry_status") or "").lower()
        rank = 0 if status == "production" else 1 if status == "draft" else 2
        return rank, str(row.get("model_id"))

    ordered = model_status.sort_values(by=list(model_status.columns)).apply(sort_key, axis=1)
    model_status = model_status.assign(_sort=ordered)
    return [str(x) for x in model_status.sort_values("_sort")["model_id"].dropna().unique()]


def _model_aliases(model_ids: list[str], df: pd.DataFrame) -> dict[str, str]:
    aliases: dict[str, str] = {}
    draft_idx = 0
    for model_id in model_ids:
        status = ""
        if "model_registry_status" in df.columns:
            model_rows = df[df["model_id"].astype(str) == str(model_id)]
            if not model_rows.empty:
                status = str(model_rows["model_registry_status"].dropna().astype(str).iloc[0]).lower()
        if status == "production" and "Prod" not in aliases.values():
            aliases[model_id] = "Prod"
        else:
            draft_idx += 1
            aliases[model_id] = chr(64 + draft_idx) if draft_idx <= 26 else f"M{draft_idx}"
    return aliases


def _consensus_status(agreement_pct: float, spread_pct: float, support: int, model_count: int) -> str:
    # Outlier means one model is far away while the rest broadly agree, not merely a low-agreement split.
    if model_count >= 3 and support >= model_count - 1 and spread_pct >= OUTLIER_SPREAD_THRESHOLD:
        return "Outlier"
    if agreement_pct >= STRONG_THRESHOLD:
        return "Strong Consensus"
    if agreement_pct >= LEAN_THRESHOLD:
        return "Lean Consensus"
    return "Split"


def _status_icon(status: str) -> str:
    return {
        "Strong Consensus": "🟢",
        "Lean Consensus": "🟡",
        "Split": "🔴",
        "Outlier": "🟣",
    }.get(status, "⚪")


def _one_row_per_model_outcome(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model_probability" in out.columns:
        out["_prob_sort"] = pd.to_numeric(out["model_probability"], errors="coerce")
    if "ev_dollars_at_100" in out.columns:
        out["_ev_sort"] = pd.to_numeric(out["ev_dollars_at_100"], errors="coerce")
    sort_cols = [col for col in ["fight_id", "market_key", "model_id", "_prob_sort", "_ev_sort"] if col in out.columns]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if "_prob_sort" in sort_cols:
            ascending[sort_cols.index("_prob_sort")] = False
        if "_ev_sort" in sort_cols:
            ascending[sort_cols.index("_ev_sort")] = False
        out = out.sort_values(sort_cols, ascending=ascending)
    dedupe = [col for col in ["fight_id", "market_key", "model_id", "outcome_join_key"] if col in out.columns]
    if dedupe:
        out = out.drop_duplicates(dedupe, keep="first")
    return out.drop(columns=["_prob_sort", "_ev_sort"], errors="ignore")


def _model_pick_rows(group: pd.DataFrame) -> pd.DataFrame:
    if "is_model_pick" in group.columns:
        picks = group[group["is_model_pick"].fillna(False).astype(bool)].copy()
        if not picks.empty:
            return picks.sort_values("model_probability", ascending=False).drop_duplicates("model_id")
    return group.sort_values("model_probability", ascending=False).drop_duplicates("model_id")


def _best_market_display(group: pd.DataFrame, consensus_pick: str) -> tuple[str, str, float | None]:
    outcome_rows = group[group["outcome_label"].astype(str) == str(consensus_pick)] if "outcome_label" in group.columns else group
    if outcome_rows.empty:
        outcome_rows = group
    odds = pd.to_numeric(outcome_rows.get("american_odds", pd.Series(dtype=float)), errors="coerce").dropna()
    implied = pd.to_numeric(outcome_rows.get("implied_probability", pd.Series(dtype=float)), errors="coerce").dropna()
    best_odds = None if odds.empty else odds.max()
    best_implied = None if implied.empty else implied.min()
    return _fmt_odds(best_odds), _fmt_pct(best_implied), best_implied


def _build_compare_board(df: pd.DataFrame, model_ids: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    required = {"fight_id", "market_key", "model_id", "outcome_label", "model_probability"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    source = _one_row_per_model_outcome(df)
    rows: list[dict[str, Any]] = []
    for key_values, group in source.groupby(["event_name", "fight_id", "fight_display", "market_key"], dropna=False):
        event_name, fight_id, fight_display, market_key = key_values
        picks = _model_pick_rows(group)
        present_models = [model_id for model_id in model_ids if model_id in set(picks["model_id"].astype(str))]
        model_count = len(present_models)
        if model_count == 0:
            continue

        pick_counts = picks.groupby("outcome_label")["model_id"].nunique().sort_values(ascending=False)
        consensus_pick = str(pick_counts.index[0]) if not pick_counts.empty else "—"
        support = int(pick_counts.iloc[0]) if not pick_counts.empty else 0
        agreement_pct = support / model_count * 100.0 if model_count else 0.0

        pick_probs = pd.to_numeric(picks["model_probability"], errors="coerce")
        avg_prob_pct = float(pick_probs.mean() * 100.0) if not pick_probs.dropna().empty else 0.0
        spread_pct = float((pick_probs.max() - pick_probs.min()) * 100.0) if not pick_probs.dropna().empty else 0.0
        ev_values = pd.to_numeric(picks.get("ev_dollars_at_100", pd.Series(dtype=float)), errors="coerce")
        avg_ev = float(ev_values.mean()) if not ev_values.dropna().empty else 0.0
        max_ev = float(ev_values.max()) if not ev_values.dropna().empty else 0.0
        odds_display, implied_display, best_implied = _best_market_display(group, consensus_pick)

        prod_pick = "—"
        prod_prob = pd.NA
        prod_rows = picks[picks.get("model_registry_status", pd.Series(dtype=str)).astype(str).str.lower() == "production"]
        if not prod_rows.empty:
            prod_row = prod_rows.iloc[0]
            prod_pick = _clean_text(prod_row.get("outcome_label"))
            prod_prob = pd.to_numeric(pd.Series([prod_row.get("model_probability")]), errors="coerce").iloc[0]

        status = _consensus_status(agreement_pct, spread_pct, support, model_count)
        row: dict[str, Any] = {
            "event_name": event_name,
            "fight_id": str(fight_id),
            "market_key": market_key,
            "Fight": fight_display,
            "Market": market_key,
            "Odds (Best)": odds_display,
            "Implied Prob (%)": implied_display,
            "Consensus Pick": consensus_pick,
            "Agreement": f"{_status_icon(status)} {_fmt_pct(agreement_pct)}",
            "Agreement %": round(agreement_pct, 1),
            "Avg Prob (%)": round(avg_prob_pct, 1),
            "Prob Spread (%)": round(spread_pct, 1),
            "Avg EV ($100 stake)": round(avg_ev, 2),
            "Max EV ($100 stake)": round(max_ev, 2),
            "Production Pick": prod_pick,
            "Status": status,
            "Models Included": model_count,
            "_best_implied": best_implied,
            "_prod_prob": prod_prob,
        }

        for model_id in model_ids:
            alias = aliases.get(model_id, model_id)
            model_pick = picks[picks["model_id"].astype(str) == str(model_id)]
            if model_pick.empty:
                row[f"{alias} Prob (%)"] = "—"
                row[f"{alias} Pick"] = "—"
                continue
            model_row = model_pick.iloc[0]
            prob = pd.to_numeric(pd.Series([model_row.get("model_probability")]), errors="coerce").iloc[0]
            outcome = _clean_text(model_row.get("outcome_label"))
            row[f"{alias} Prob (%)"] = _fmt_pct(prob)
            row[f"{alias} Pick"] = outcome
        rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    return board.sort_values(["event_name", "Fight", "Market"]).reset_index(drop=True)


def _build_suggested_bets(df: pd.DataFrame, model_ids: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    if df.empty or "is_bet_candidate" not in df.columns:
        return pd.DataFrame()
    bets = df[df["is_bet_candidate"].fillna(False).astype(bool)].copy()
    if bets.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["event_name", "fight_id", "fight_display", "market_key", "outcome_join_key", "outcome_label"]
    for key_values, group in bets.groupby(group_cols, dropna=False):
        event_name, fight_id, fight_display, market_key, outcome_join_key, outcome_label = key_values
        models = sorted([str(x) for x in group["model_id"].dropna().unique()], key=lambda x: model_ids.index(x) if x in model_ids else 999)
        model_aliases = [aliases.get(model_id, model_id) for model_id in models]
        prob = pd.to_numeric(group.get("model_probability", pd.Series(dtype=float)), errors="coerce")
        implied = pd.to_numeric(group.get("implied_probability", pd.Series(dtype=float)), errors="coerce")
        ev = pd.to_numeric(group.get("ev_dollars_at_100", pd.Series(dtype=float)), errors="coerce")
        stake = pd.to_numeric(group.get("recommended_stake", pd.Series(dtype=float)), errors="coerce")
        odds = pd.to_numeric(group.get("american_odds", pd.Series(dtype=float)), errors="coerce").dropna()
        book = group.get("bookmaker", pd.Series(dtype=str)).dropna().astype(str)
        rows.append(
            {
                "Fight": fight_display,
                "Bet (Outcome)": outcome_label,
                "Book (Best)": book.iloc[0] if not book.empty else "—",
                "Odds": _fmt_odds(odds.max() if not odds.empty else None),
                "Implied Prob (%)": round(float(implied.mean() * 100.0), 1) if not implied.dropna().empty else pd.NA,
                "Models Supporting": f"{len(models)}/{len(model_ids)} " + ", ".join(model_aliases),
                "Avg Model Prob (%)": round(float(prob.mean() * 100.0), 1) if not prob.dropna().empty else pd.NA,
                "Avg EV ($100)": round(float(ev.mean()), 2) if not ev.dropna().empty else pd.NA,
                "Kelly Range ($)": "—" if stake.dropna().empty else f"{stake.min():.0f} - {stake.max():.0f}",
                "Status": "Consensus Bet" if len(models) >= max(1, len(model_ids) - 1) else "Lean Bet",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Avg EV ($100)", ascending=False).reset_index(drop=True)


def _build_disagreements(df: pd.DataFrame, model_ids: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    source = _one_row_per_model_outcome(df)
    for key_values, group in source.groupby(["event_name", "fight_id", "fight_display", "market_key"], dropna=False):
        event_name, fight_id, fight_display, market_key = key_values
        picks = _model_pick_rows(group)
        side_map: dict[str, list[str]] = {}
        for _, row in picks.iterrows():
            side = _clean_text(row.get("outcome_label"))
            model_id = str(row.get("model_id"))
            side_map.setdefault(side, []).append(aliases.get(model_id, model_id))
        if len(side_map) < 2:
            continue
        sorted_sides = sorted(side_map.items(), key=lambda item: len(item[1]), reverse=True)
        probs = pd.to_numeric(picks.get("model_probability", pd.Series(dtype=float)), errors="coerce")
        ev = pd.to_numeric(picks.get("ev_dollars_at_100", pd.Series(dtype=float)), errors="coerce")
        implied = pd.to_numeric(picks.get("implied_probability", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "Fight / Market": f"{fight_display} ({market_key})",
                "Side A (Models)": f"{sorted_sides[0][0]} ({', '.join(sorted_sides[0][1])})",
                "Side B (Models)": f"{sorted_sides[1][0]} ({', '.join(sorted_sides[1][1])})",
                "Prob Spread (%)": round(float((probs.max() - probs.min()) * 100.0), 1) if not probs.dropna().empty else pd.NA,
                "Implied Prob Diff (%)": round(float((implied.max() - implied.min()) * 100.0), 1) if not implied.dropna().empty else pd.NA,
                "EV Spread ($)": round(float(ev.max() - ev.min()), 2) if not ev.dropna().empty else pd.NA,
                "Notes": "Model split",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Prob Spread (%)", ascending=False).reset_index(drop=True)


def _filter_data(df: pd.DataFrame, board: pd.DataFrame, model_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filter_cols = st.columns([1.55, 1.0, 1.0, 1.22, 0.9, 0.95])
    out = df.copy()
    board_out = board.copy()

    event_options = ["All Events"] + sorted([str(x) for x in out.get("event_name", pd.Series(dtype=str)).dropna().unique()])
    with filter_cols[0]:
        event = st.selectbox("Event", event_options, key="compare_event")
    if event != "All Events" and "event_name" in out.columns:
        out = out[out["event_name"].astype(str) == event]
        board_out = board_out[board_out["event_name"].astype(str) == event]

    market_options = ["All Markets"] + sorted([str(x) for x in out.get("market_key", pd.Series(dtype=str)).dropna().unique()])
    with filter_cols[1]:
        market = st.selectbox("Market Type", market_options, key="compare_market")
    if market != "All Markets" and "market_key" in out.columns:
        out = out[out["market_key"].astype(str) == market]
        board_out = board_out[board_out["market_key"].astype(str) == market]

    with filter_cols[2]:
        model_set = st.selectbox("Model Set", ["All Models", "Production Only", "Draft Only"], key="compare_model_set")
    if "model_registry_status" in out.columns:
        statuses = out["model_registry_status"].astype(str).str.lower()
        if model_set == "Production Only":
            out = out[statuses == "production"]
        elif model_set == "Draft Only":
            out = out[statuses == "draft"]

    with filter_cols[3]:
        model_status = st.selectbox("Model Status", ["All", "production", "draft", "single"], key="compare_model_status")
    if model_status != "All" and "model_registry_status" in out.columns:
        out = out[out["model_registry_status"].astype(str).str.lower() == model_status]

    with filter_cols[4]:
        min_ev = st.number_input("Minimum EV ($)", value=0.0, step=1.0, key="compare_min_ev")
    if "Max EV ($100 stake)" in board_out.columns:
        board_out = board_out[pd.to_numeric(board_out["Max EV ($100 stake)"], errors="coerce").fillna(-10**9) >= float(min_ev)]

    with filter_cols[5]:
        min_agreement = st.number_input("Minimum Agreement (%)", value=0.0, min_value=0.0, max_value=100.0, step=5.0, key="compare_min_agreement")
    if "Agreement %" in board_out.columns:
        board_out = board_out[pd.to_numeric(board_out["Agreement %"], errors="coerce").fillna(0) >= float(min_agreement)]

    toggle_cols = st.columns([1.1, 1.1, 1.2, 3.2])
    with toggle_cols[0]:
        disagreements_only = st.toggle("Show Disagreements Only", value=False, key="compare_disagreements_only")
    with toggle_cols[1]:
        bets_only = st.toggle("Show Suggested Bets Only", value=False, key="compare_bets_only")
    with toggle_cols[2]:
        hide_incomplete = st.toggle("Hide Fights Without All Models", value=False, key="compare_hide_incomplete")

    if disagreements_only and "Status" in board_out.columns:
        board_out = board_out[board_out["Status"].isin(["Split", "Outlier"])]
    if hide_incomplete and "Models Included" in board_out.columns:
        board_out = board_out[pd.to_numeric(board_out["Models Included"], errors="coerce").fillna(0) >= len(model_ids)]
    if bets_only and "is_bet_candidate" in out.columns:
        bet_keys = out.loc[out["is_bet_candidate"].fillna(False).astype(bool), ["fight_id", "market_key"]].drop_duplicates()
        board_out = board_out.merge(bet_keys, on=["fight_id", "market_key"], how="inner")

    valid_keys = set(zip(board_out["fight_id"].astype(str), board_out["market_key"].astype(str))) if not board_out.empty else set()
    if valid_keys:
        out = out[out.apply(lambda row: (str(row.get("fight_id")), str(row.get("market_key"))) in valid_keys, axis=1)]
    else:
        out = out.iloc[0:0]
    return out, board_out


def _model_key_text(model_ids: list[str], aliases: dict[str, str], df: pd.DataFrame) -> str:
    parts = []
    for model_id in model_ids:
        status = ""
        if "model_registry_status" in df.columns:
            rows = df[df["model_id"].astype(str) == model_id]
            if not rows.empty:
                status = rows["model_registry_status"].dropna().astype(str).iloc[0]
        parts.append(f"{aliases.get(model_id, model_id)} = {model_id}" + (f" ({status})" if status else ""))
    return "Models: " + " · ".join(parts)


def _render_summary_cards(df: pd.DataFrame, board: pd.DataFrame) -> None:
    latest_timestamp = "—"
    if "betting_timestamp" in df.columns:
        timestamps = pd.to_datetime(df["betting_timestamp"], errors="coerce", utc=True).dropna()
        if not timestamps.empty:
            latest_timestamp = timestamps.max().strftime("%Y-%m-%d %H:%M:%S UTC")
    models = df[["model_id", "model_registry_status"]].drop_duplicates() if "model_registry_status" in df.columns else pd.DataFrame()
    total = int(models["model_id"].nunique()) if not models.empty else int(df.get("model_id", pd.Series(dtype=str)).nunique())
    prod = int((models["model_registry_status"].astype(str).str.lower() == "production").sum()) if not models.empty else 0
    draft = int((models["model_registry_status"].astype(str).str.lower() == "draft").sum()) if not models.empty else 0
    c1, c2, c3, c4, c5 = st.columns([1.5, .7, .7, .7, .8])
    c1.caption(f"Last Outcomes Run: {latest_timestamp}")
    c2.metric("Models", total)
    c3.metric("Production", prod)
    c4.metric("Draft", draft)
    c5.metric("Markets", len(board))


def _render_consensus_summary(board: pd.DataFrame) -> None:
    if board.empty:
        st.info("No consensus rows available for the selected filters.")
        return
    counts = board["Status"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Strong Consensus", int(counts.get("Strong Consensus", 0)))
    c1.metric("Lean Consensus", int(counts.get("Lean Consensus", 0)))
    c2.metric("Split", int(counts.get("Split", 0)))
    c2.metric("Outlier", int(counts.get("Outlier", 0)))
    c3.metric("Average Agreement", _fmt_pct(pd.to_numeric(board["Agreement %"], errors="coerce").mean()))
    c3.metric("Average Prob Spread", _fmt_pct(pd.to_numeric(board["Prob Spread (%)"], errors="coerce").mean()))
    c3.caption(f"Total markets: {len(board)}")


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
        st.warning("No betting outcomes artifact found. Run Model Lab → Actions → Run Outcomes first.")
        return

    latest = _enrich_status(_latest_run(raw), rows)
    if latest.empty:
        st.warning("Betting outcomes artifact exists, but no usable rows were found for the latest run.")
        return

    model_ids = _model_order(latest)
    aliases = _model_aliases(model_ids, latest)
    base_board = _build_compare_board(latest, model_ids, aliases)
    if base_board.empty:
        st.warning("Betting outcomes loaded, but the compare board could not be built from the available columns.")
        return

    _render_summary_cards(latest, base_board)
    if not audit.empty and "passes_validation" in audit.columns and not bool(audit.tail(1)["passes_validation"].iloc[0]):
        st.warning("Latest betting outcomes audit did not pass validation. Review join-key diagnostics before relying on this board.")

    st.divider()
    filtered, board = _filter_data(latest, base_board, model_ids)
    suggested = _build_suggested_bets(filtered, model_ids, aliases)
    disagreements = _build_disagreements(filtered, model_ids, aliases)

    tabs = st.tabs(["Overview", "Suggested Bets", "Disagreements", "Model Probabilities", "Model EV Comparison", "Consensus Summary"])

    model_prob_cols = [f"{aliases[mid]} Prob (%)" for mid in model_ids]
    overview_cols = [
        "Fight",
        "Market",
        "Odds (Best)",
        "Implied Prob (%)",
        *model_prob_cols,
        "Consensus Pick",
        "Agreement",
        "Avg Prob (%)",
        "Prob Spread (%)",
        "Avg EV ($100 stake)",
        "Max EV ($100 stake)",
        "Production Pick",
        "Status",
    ]

    with tabs[0]:
        st.markdown("#### Fight / Market Comparison")
        if board.empty:
            st.info("No rows match the selected filters.")
        else:
            st.dataframe(board[[col for col in overview_cols if col in board.columns]], use_container_width=True, hide_index=True)

        c1, c2 = st.columns([1.0, 1.0], gap="medium")
        with c1:
            st.markdown("#### Suggested Bets Comparison")
            st.dataframe(suggested.head(10), use_container_width=True, hide_index=True) if not suggested.empty else st.info("No suggested bets match the selected filters.")
        with c2:
            st.markdown("#### Disagreement Board")
            st.dataframe(disagreements.head(10), use_container_width=True, hide_index=True) if not disagreements.empty else st.success("No model disagreements for the selected filters.")

    with tabs[1]:
        st.markdown("#### Suggested Bets")
        st.dataframe(suggested, use_container_width=True, hide_index=True) if not suggested.empty else st.info("No suggested bets match the selected filters.")

    with tabs[2]:
        st.markdown("#### Disagreements")
        st.dataframe(disagreements, use_container_width=True, hide_index=True) if not disagreements.empty else st.success("No split or outlier markets match the selected filters.")

    with tabs[3]:
        st.markdown("#### Model Probabilities")
        pick_cols = [f"{aliases[mid]} Pick" for mid in model_ids]
        cols = ["Fight", "Market", *model_prob_cols, *pick_cols, "Consensus Pick", "Agreement", "Status"]
        st.dataframe(board[[col for col in cols if col in board.columns]], use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown("#### Model EV Comparison")
        cols = ["event_name", "fight_display", "market_display", "outcome_display", "model_id", "model_registry_status", "bookmaker", "american_odds", "implied_probability", "edge_pct", "ev_dollars_at_100", "recommended_stake", "bet_status"]
        display = filtered[[col for col in cols if col in filtered.columns]].copy()
        if "implied_probability" in display.columns:
            display["implied_probability"] = pd.to_numeric(display["implied_probability"], errors="coerce") * 100.0
        st.dataframe(display, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.markdown("#### Model Consensus Summary")
        _render_consensus_summary(board)

    st.divider()
    st.caption(_model_key_text(model_ids, aliases, latest))
    st.caption("Legend: Strong = 75%+ agreement · Lean = 55–74% · Split = under 55% · Outlier = broad agreement with a large probability spread. EV is at $100 stake.")
