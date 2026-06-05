import numpy as np
import pandas as pd
import streamlit as st

from pipeline.common.paths import BETTING_BOARD_PATH, MARKET_MATCH_AUDIT_PATH
from utils.betting_board_artifacts import (
    artifact_readiness_summary,
    event_label,
    get_betting_artifact_status,
    get_upcoming_artifact_status,
    load_upcoming_events,
    load_upcoming_fights,
)
from utils.betting_board_rules import (
    BettingRules,
    apply_betting_rules,
    normalize_betting_board_odds,
    default_betting_rules,
    rules_changed_from_default,
    scenario_comparison,
)
from utils.bankroll_artifacts import append_official_bets
from utils.data_loader import load_parquet
from utils.dm_workflow_status import remember_launched_workflow, render_workflow_status
from utils.github_actions import trigger_workflow
from utils.panels import render_section_header
from utils.ui_components import american, money, pct, render_metric
from utils.ui.sections import page_header


REFRESH_UPCOMING_WORKFLOW = "run-refresh-upcoming-events.yml"
SELECTED_EVENT_WORKFLOW = "run-betting-board-selected-event.yml"


def _artifact_health_table():
    status = pd.concat(
        [
            get_upcoming_artifact_status().assign(group="Card selection"),
            get_betting_artifact_status().assign(group="Betting outputs"),
        ],
        ignore_index=True,
    )
    return status[
        [
            "group",
            "artifact",
            "required_for",
            "health",
            "optional",
            "rows",
            "size",
            "age_hours",
            "modified_utc",
            "path",
        ]
    ]


def render_artifact_readiness(status):
    summary = artifact_readiness_summary(status)
    cols = st.columns(4)
    cols[0].metric("Required ready", f"{summary['required_ready']} / {summary['required_total']}")
    cols[1].metric("Missing required", summary["missing_required"])
    cols[2].metric("Empty required", summary["empty_required"])
    cols[3].metric("Optional missing", summary["optional_missing"])

    if summary["ready_to_review"]:
        st.success("Required Betting Board artifacts are present and non-empty.")
    else:
        st.warning("One or more required Betting Board artifacts are missing or empty. Refresh/run the upcoming-events workflow before trusting board output.")


def _selected_event_id(event_row):
    if event_row is None:
        return None
    return event_row.get("ufcstats_event_id") or event_row.get("event_id")


def _selected_event_name(event_row):
    if event_row is None:
        return None
    return event_row.get("ufcstats_event_name") or event_row.get("event_name")


def _scope_board_to_selected_event(board, selected_event):
    """Return only rows for the event currently selected in the card selector."""

    if selected_event is None or board.empty:
        return board, None, False

    selected_event_id = _selected_event_id(selected_event)
    selected_event_name = _selected_event_name(selected_event)
    selected_label = selected_event_name or selected_event_id

    masks = []
    if selected_event_id:
        for event_id_column in ["event_id", "ufcstats_event_id"]:
            if event_id_column in board.columns:
                masks.append(board[event_id_column].astype(str) == str(selected_event_id))

    if selected_event_name and "event_name" in board.columns:
        masks.append(board["event_name"].astype(str) == str(selected_event_name))

    if not masks:
        return board, selected_label, False

    selected_mask = masks[0]
    for mask in masks[1:]:
        selected_mask = selected_mask | mask

    return board[selected_mask].copy(), selected_label, True


def render_upcoming_event_selection():
    render_section_header("Upcoming Event Selection")

    with st.expander("Select an upcoming UFCStats event for betting predictions", expanded=True):
        st.caption(
            "Refresh the UFCStats upcoming-events artifact, choose a card, then launch the upcoming-events "
            "betting workflow. The workflow builds model predictions, market odds, and betting board outputs "
            "for all upcoming fights; this selector controls which event appears in the Action Board."
        )

        artifact_status = _artifact_health_table()
        render_artifact_readiness(artifact_status)
        st.dataframe(artifact_status, use_container_width=True, hide_index=True)

        control_cols = st.columns([1, 1])

        with control_cols[0]:
            if st.button("Refresh Upcoming Events", use_container_width=True):
                ok, msg = trigger_workflow(REFRESH_UPCOMING_WORKFLOW)
                if ok:
                    remember_launched_workflow(
                        "betting_refresh_upcoming_events",
                        "Refresh Upcoming Events",
                        REFRESH_UPCOMING_WORKFLOW,
                    )
                    st.success(msg)
                else:
                    st.error(msg)

        with control_cols[1]:
            st.caption(
                "Refresh before selecting a new card if UFCStats has changed. The prediction workflow evaluates "
                "every upcoming fight; the selected event only scopes the dashboard display."
            )

        render_workflow_status("betting_refresh_upcoming_events")

        events, events_error = load_upcoming_events()
        fights, fights_error = load_upcoming_fights()

        if events_error:
            st.warning(events_error)
            return None

        if events.empty:
            st.warning("No upcoming events are available yet. Refresh upcoming events first.")
            return None

        events = events.sort_values("ufcstats_event_date", na_position="last").reset_index(drop=True)
        event_options = events.to_dict("records")
        selected_event = st.selectbox(
            "Upcoming event",
            options=event_options,
            format_func=event_label,
            key="betting_selected_upcoming_event",
        )

        selected_event_id = _selected_event_id(selected_event)

        event_cols = [
            column
            for column in [
                "ufcstats_event_id",
                "ufcstats_event_date",
                "ufcstats_event_name",
                "ufcstats_event_location",
                "ufcstats_event_url",
            ]
            if column in events.columns
        ]
        st.dataframe(pd.DataFrame([selected_event])[event_cols], use_container_width=True, hide_index=True)

        if fights_error:
            st.warning(fights_error)
        elif not fights.empty and "event_id" in fights.columns:
            selected_fights = fights[fights["event_id"].astype(str) == str(selected_event_id)]
            fight_cols = [
                column
                for column in [
                    "fight_order",
                    "red_fighter",
                    "blue_fighter",
                    "weight_class",
                    "fight_id",
                ]
                if column in selected_fights.columns
            ]
            st.markdown(f"**Selected card fights:** {len(selected_fights)}")
            st.dataframe(selected_fights[fight_cols], use_container_width=True, hide_index=True)

        if st.button("Run Betting Predictions for Upcoming Events", type="primary", use_container_width=True):
            ok, msg = trigger_workflow(
                SELECTED_EVENT_WORKFLOW,
                inputs={"event_id": str(selected_event_id)},
            )
            if ok:
                remember_launched_workflow(
                    "betting_selected_event",
                    "Run Betting Predictions for Upcoming Events",
                    SELECTED_EVENT_WORKFLOW,
                    inputs={"event_id": str(selected_event_id)},
                )
                st.success(msg)
            else:
                st.error(msg)

        render_workflow_status("betting_selected_event")

        return selected_event


def _rule_state_key(name):
    return f"betting_rule_{name}"


def _reset_rule_state(defaults):
    default_values = {
        "min_edge": defaults.min_edge,
        "min_confidence": defaults.min_confidence,
        "min_odds": defaults.min_odds,
        "max_odds": defaults.max_odds,
        "require_positive_ev": defaults.require_positive_ev,
        "watchlist_max_failed_thresholds": defaults.watchlist_max_failed_thresholds,
        "watchlist_high_ev_override": defaults.watchlist_high_ev_override,
        "bankroll": defaults.bankroll,
        "kelly_fraction": defaults.kelly_fraction,
        "max_stake_pct_percent": defaults.max_stake_pct * 100,
        "min_stake": defaults.min_stake,
        "stake_rounding": defaults.stake_rounding,
    }

    for key, value in default_values.items():
        st.session_state[_rule_state_key(key)] = value


def render_betting_rules_controls():
    defaults = default_betting_rules()

    with st.expander("Betting Rules / Scenario Controls", expanded=True):
        st.caption(
            "Workflow runs use the production defaults. Adjust these controls after a run to recalculate "
            "the displayed board as a dashboard-only scenario. Scenario values are not committed."
        )

        if st.button("Reset rules to production defaults", use_container_width=True):
            _reset_rule_state(defaults)
            st.rerun()

        st.markdown("#### Bet Qualification")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            min_edge = st.number_input(
                "Minimum edge",
                min_value=-1.0,
                max_value=1.0,
                value=float(st.session_state.get(_rule_state_key("min_edge"), defaults.min_edge)),
                step=0.01,
                format="%.2f",
                key=_rule_state_key("min_edge"),
            )

        with c2:
            min_confidence = st.number_input(
                "Minimum confidence (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get(_rule_state_key("min_confidence"), defaults.min_confidence)),
                step=1.0,
                format="%.1f",
                key=_rule_state_key("min_confidence"),
            )

        with c3:
            min_odds = st.number_input(
                "Minimum American odds",
                min_value=-1000,
                max_value=1000,
                value=int(st.session_state.get(_rule_state_key("min_odds"), defaults.min_odds)),
                step=5,
                key=_rule_state_key("min_odds"),
            )

        with c4:
            max_odds = st.number_input(
                "Maximum American odds",
                min_value=-1000,
                max_value=2000,
                value=int(st.session_state.get(_rule_state_key("max_odds"), defaults.max_odds)),
                step=5,
                key=_rule_state_key("max_odds"),
            )

        require_positive_ev = st.checkbox(
            "Require positive EV for official bets",
            value=bool(st.session_state.get(_rule_state_key("require_positive_ev"), defaults.require_positive_ev)),
            key=_rule_state_key("require_positive_ev"),
        )

        st.markdown("#### Watchlist Rules")
        w1, w2 = st.columns(2)

        with w1:
            watchlist_max_failed_thresholds = st.number_input(
                "Watchlist max failed betting thresholds",
                min_value=0,
                max_value=4,
                value=int(st.session_state.get(_rule_state_key("watchlist_max_failed_thresholds"), defaults.watchlist_max_failed_thresholds)),
                step=1,
                key=_rule_state_key("watchlist_max_failed_thresholds"),
            )

        with w2:
            watchlist_high_ev_override = st.number_input(
                "Watchlist high-EV override",
                min_value=-100.0,
                max_value=500.0,
                value=float(st.session_state.get(_rule_state_key("watchlist_high_ev_override"), defaults.watchlist_high_ev_override)),
                step=1.0,
                format="%.2f",
                key=_rule_state_key("watchlist_high_ev_override"),
            )

        st.markdown("#### Bankroll / Kelly Staking")
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            bankroll = st.number_input(
                "Bankroll ($)",
                min_value=0.0,
                max_value=10000000.0,
                value=float(st.session_state.get(_rule_state_key("bankroll"), defaults.bankroll)),
                step=100.0,
                format="%.2f",
                key=_rule_state_key("bankroll"),
            )

        with s2:
            kelly_fraction = st.number_input(
                "Kelly fraction",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.get(_rule_state_key("kelly_fraction"), defaults.kelly_fraction)),
                step=0.05,
                format="%.2f",
                key=_rule_state_key("kelly_fraction"),
            )

        with s3:
            max_stake_pct_percent = st.number_input(
                "Max stake (% bankroll)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get(_rule_state_key("max_stake_pct_percent"), defaults.max_stake_pct * 100)),
                step=0.25,
                format="%.2f",
                key=_rule_state_key("max_stake_pct_percent"),
            )

        with s4:
            min_stake = st.number_input(
                "Minimum stake ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.get(_rule_state_key("min_stake"), defaults.min_stake)),
                step=1.0,
                format="%.2f",
                key=_rule_state_key("min_stake"),
            )

        stake_rounding = st.number_input(
            "Round stake to nearest ($)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get(_rule_state_key("stake_rounding"), defaults.stake_rounding)),
            step=1.0,
            format="%.2f",
            key=_rule_state_key("stake_rounding"),
        )

        rules = BettingRules(
            min_edge=float(min_edge),
            min_confidence=float(min_confidence),
            min_odds=int(min_odds),
            max_odds=int(max_odds),
            require_positive_ev=bool(require_positive_ev),
            watchlist_max_failed_thresholds=int(watchlist_max_failed_thresholds),
            watchlist_high_ev_override=float(watchlist_high_ev_override),
            bankroll=float(bankroll),
            kelly_fraction=float(kelly_fraction),
            max_stake_pct=float(max_stake_pct_percent) / 100,
            min_stake=float(min_stake),
            stake_rounding=float(stake_rounding),
        )

        if rules_changed_from_default(rules):
            st.warning("Scenario rules differ from production defaults. Displayed results are dashboard-only what-if results.")
        else:
            st.success("Using production default betting rules.")

    return rules


def render_scenario_summary(board, scenario):
    comparison = scenario_comparison(board, scenario)

    render_section_header("Production vs Scenario Summary")

    cols = st.columns(6)
    cols[0].metric("Prod Official", comparison["production_official_bets"])
    cols[1].metric("Scenario Official", comparison["scenario_official_bets"])
    cols[2].metric("Prod Stake", money(comparison["production_total_stake"]))
    cols[3].metric("Scenario Stake", money(comparison["scenario_total_stake"]))
    cols[4].metric("Added / Removed", f"+{comparison['added_official_bets']} / -{comparison['removed_official_bets']}")
    cols[5].metric("Stake Delta", money(comparison["stake_delta"]))


def render_board_filters(board):
    with st.expander("Betting Board Filters", expanded=True):
        events = sorted(board["event_name"].dropna().unique().tolist()) if "event_name" in board.columns else []
        selected_event = st.selectbox("Event", ["All Events"] + events)

        status_order = [
            "OFFICIAL BET",
            "WATCHLIST",
            "LOW ODDS MATCH",
            "SPARSE FEATURES",
            "INVALID MODEL DATA",
            "NO BET",
        ]
        status_column = "scenario_bet_status" if "scenario_bet_status" in board.columns else "bet_status"
        available_statuses = [s for s in status_order if status_column in board.columns and s in board[status_column].dropna().unique()]
        selected_statuses = st.multiselect("Bet status", available_statuses, default=available_statuses)
        show_only_actionable = st.checkbox("Show only actionable statuses", value=False)
        min_ev = st.slider("Minimum EV", min_value=-100.0, max_value=100.0, value=-100.0, step=1.0)
        min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    filtered = board.copy()

    if selected_event != "All Events" and "event_name" in filtered.columns:
        filtered = filtered[filtered["event_name"] == selected_event]

    status_column = "scenario_bet_status" if "scenario_bet_status" in filtered.columns else "bet_status"

    if selected_statuses and status_column in filtered.columns:
        filtered = filtered[filtered[status_column].isin(selected_statuses)]

    if show_only_actionable and status_column in filtered.columns:
        filtered = filtered[filtered[status_column].isin(["OFFICIAL BET", "WATCHLIST"])]

    ev_col = "best_ev_pct" if "best_ev_pct" in filtered.columns else "best_ev"
    if ev_col in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered[ev_col], errors="coerce").fillna(-999) >= min_ev]

    if "best_confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["best_confidence"], errors="coerce").fillna(0) >= min_confidence]

    return filtered


def render_summary_cards(filtered):
    total_fights = len(filtered)
    status_column = "scenario_bet_status" if "scenario_bet_status" in filtered.columns else "bet_status"
    stake_column = "scenario_recommended_stake" if "scenario_recommended_stake" in filtered.columns else "recommended_stake"
    official_bets = int((filtered[status_column] == "OFFICIAL BET").sum()) if status_column in filtered.columns else 0
    watchlist = int((filtered[status_column] == "WATCHLIST").sum()) if status_column in filtered.columns else 0
    best_ev_col = "best_ev_pct" if "best_ev_pct" in filtered.columns else "best_ev"
    best_ev = filtered[best_ev_col].max() if best_ev_col in filtered.columns and not filtered.empty else np.nan
    recommended_stake = filtered[stake_column].sum() if stake_column in filtered.columns and not filtered.empty else 0
    latest_market_time = str(filtered["snapshot_timestamp"].max()) if "snapshot_timestamp" in filtered.columns and not filtered.empty else "N/A"

    cols = st.columns(6)
    with cols[0]:
        render_metric("Fights", total_fights)
    with cols[1]:
        render_metric("Official Bets", official_bets)
    with cols[2]:
        render_metric("Watchlist", watchlist)
    with cols[3]:
        render_metric("Best EV", f"{best_ev:.1f}%" if pd.notna(best_ev) else "N/A")
    with cols[4]:
        render_metric("Total Stake", money(recommended_stake))
    with cols[5]:
        render_metric("Latest Market", latest_market_time[:16])


def build_display_frame(filtered):
    display = filtered.copy()

    if "red_fighter" in display.columns and "blue_fighter" in display.columns:
        display["fight"] = display["red_fighter"].fillna("") + " vs " + display["blue_fighter"].fillna("")

    if "best_american_odds" in display.columns:
        display["odds_display"] = display["best_american_odds"].apply(american)

    for column in ["best_prob", "best_implied_prob", "best_edge"]:
        if column in display.columns:
            display[f"{column}_display"] = display[column].apply(pct)

    ev_col = "best_ev_pct" if "best_ev_pct" in display.columns else "best_ev"
    if ev_col in display.columns:
        display["best_ev_display"] = display[ev_col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    if "best_confidence" in display.columns:
        display["best_confidence_display"] = display["best_confidence"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    if "scenario_recommended_stake" in display.columns:
        display["scenario_stake_display"] = display["scenario_recommended_stake"].apply(money)

    if "recommended_stake" in display.columns:
        display["production_stake_display"] = display["recommended_stake"].apply(money)
        display["stake_display"] = display["recommended_stake"].apply(money)

    if "scenario_bet_status" in display.columns:
        display["display_bet_status"] = display["scenario_bet_status"]
        display["production_bet_status"] = display.get("bet_status", "")
        display["display_bet_reason"] = display.get("scenario_bet_reason", "")
    else:
        display["display_bet_status"] = display.get("bet_status", "")
        display["display_bet_reason"] = display.get("bet_reason", "")

    status_rank = {
        "OFFICIAL BET": 0,
        "WATCHLIST": 1,
        "LOW ODDS MATCH": 2,
        "SPARSE FEATURES": 3,
        "INVALID MODEL DATA": 4,
        "NO BET": 5,
    }
    display["_status_rank"] = display["display_bet_status"].map(status_rank).fillna(99) if "display_bet_status" in display.columns else 99

    sort_cols = ["_status_rank"]
    ascending = [True]
    ev_sort_col = "best_ev_pct" if "best_ev_pct" in display.columns else "best_ev"
    if ev_sort_col in display.columns:
        sort_cols.append(ev_sort_col)
        ascending.append(False)

    return display.sort_values(sort_cols, ascending=ascending, na_position="last")


def render_action_board(filtered):
    render_section_header("Primary Action Board")

    display = build_display_frame(filtered)
    main_cols = [
        "display_bet_status",
        "production_bet_status",
        "fight",
        "best_side",
        "odds_display",
        "best_prob_display",
        "best_implied_prob_display",
        "best_edge_display",
        "best_ev_display",
        "best_confidence_display",
        "scenario_stake_display",
        "production_stake_display",
        "display_bet_reason",
    ]
    main_cols = [column for column in main_cols if column in display.columns]
    st.dataframe(display[main_cols], use_container_width=True, hide_index=True)



def render_add_official_bets_to_ledger(board):
    render_section_header("Add Official Bets to Bankroll Ledger")

    status_column = "scenario_bet_status" if "scenario_bet_status" in board.columns else "bet_status"
    if status_column not in board.columns:
        st.info("No bet-status column is available to add ledger candidates.")
        return

    official = board[board[status_column] == "OFFICIAL BET"].copy()
    if official.empty:
        st.info("No official bets are available under the current rules.")
        return

    with st.expander("Review and add official bets to ledger", expanded=False):
        st.caption(
            "The Betting Board recommends bets; the bankroll ledger records wagers you actually place. "
            "Select only bets that were placed, then add them to the ledger as Open."
        )
        official["add_to_ledger"] = True
        preview_cols = [
            "add_to_ledger",
            "event_name",
            "red_fighter",
            "blue_fighter",
            "best_side",
            "best_american_odds",
            "best_prob",
            "best_edge",
            "best_ev",
            "scenario_recommended_stake",
            "recommended_stake",
            "scenario_bet_reason",
            "bet_reason",
        ]
        preview_cols = [column for column in preview_cols if column in official.columns]
        edited = st.data_editor(
            official[preview_cols],
            use_container_width=True,
            hide_index=True,
            disabled=[column for column in preview_cols if column != "add_to_ledger"],
            key="bankroll_ledger_add_candidates",
        )
        selected = official.loc[edited[edited["add_to_ledger"]].index].copy()
        st.caption(f"Selected ledger rows: {len(selected)}")

        if st.button("Add Selected Bets to Bankroll Ledger", use_container_width=True, disabled=selected.empty):
            added, skipped = append_official_bets(selected, source_workflow="Betting Board")
            if added:
                st.success(f"Added {added} open bet(s) to the bankroll ledger. Skipped {skipped} duplicate(s).")
            else:
                st.warning(f"No new bets were added. Skipped {skipped} duplicate(s).")
            st.cache_data.clear()


def render_status_and_diagnostics(filtered):
    render_section_header("Status Breakdown")
    status_column = "scenario_bet_status" if "scenario_bet_status" in filtered.columns else "bet_status"
    if status_column in filtered.columns:
        status_counts = filtered[status_column].value_counts().rename_axis("status").reset_index(name="count")
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No bet_status column found.")

    render_section_header("Artifact Diagnostics")
    betting_status = get_betting_artifact_status()
    render_artifact_readiness(betting_status)
    st.dataframe(betting_status, use_container_width=True, hide_index=True)


def render_selected_fight_detail(filtered):
    render_section_header("Selected Fight Detail")
    if filtered.empty:
        st.info("No fights match the current filters.")
        return

    display = build_display_frame(filtered)
    if "fight" not in display.columns:
        st.info("Fight names are not available in this artifact.")
        return

    selected_fight = st.selectbox("Inspect fight", display["fight"].dropna().unique().tolist())
    selected_rows = display[display["fight"] == selected_fight]
    detail_cols = [
        "event_name",
        "fight",
        "best_side",
        "display_bet_status",
        "production_bet_status",
        "best_american_odds",
        "best_prob",
        "best_implied_prob",
        "best_edge",
        "best_ev",
        "best_confidence",
        "scenario_recommended_stake",
        "recommended_stake",
        "display_bet_reason",
        "odds_match_score",
        "odds_min_single_score",
        "odds_match_type",
        "odds_match_order",
        "odds_inferred_match_order",
        "odds_side_mapping_status",
        "matched_fighter_1",
        "matched_fighter_2",
    ]
    detail_cols = [column for column in detail_cols if column in selected_rows.columns]
    st.dataframe(selected_rows[detail_cols], use_container_width=True, hide_index=True)


def render_betting_board():
    page_header(
        "Betting Board",
        "Live fight predictions, market odds, EV, quality gates, and recommended actions.",
    )

    selected_event = render_upcoming_event_selection()

    board = load_parquet(BETTING_BOARD_PATH)

    if board.empty:
        st.warning("No betting board data found. Select an upcoming event and run the betting workflow.")
        return

    market_audit = load_parquet(MARKET_MATCH_AUDIT_PATH)
    if not market_audit.empty and {"event_name", "red_fighter", "blue_fighter"}.issubset(market_audit.columns):
        audit_cols = [
            column
            for column in [
                "event_name",
                "red_fighter",
                "blue_fighter",
                "matched_fighter_1",
                "matched_fighter_2",
            ]
            if column in market_audit.columns
        ]
        deduped_audit = market_audit[audit_cols].drop_duplicates(
            subset=["event_name", "red_fighter", "blue_fighter"],
            keep="last",
        )
        board = board.merge(
            deduped_audit,
            how="left",
            on=["event_name", "red_fighter", "blue_fighter"],
            suffixes=("", "_audit"),
        )

    board = normalize_betting_board_odds(board)
    board, selected_event_label, event_filter_applied = _scope_board_to_selected_event(board, selected_event)
    if selected_event_label and event_filter_applied:
        if board.empty:
            st.warning(
                f"No Betting Board rows match the selected event: {selected_event_label}. "
                "Run the upcoming-events Betting Board workflow and refresh the dashboard artifacts."
            )
            return
        st.info(f"Primary Action Board is scoped to the selected event: {selected_event_label}.")
    elif selected_event_label:
        st.warning(
            "The Betting Board artifact does not include event identifiers or event names that can be matched "
            f"to the selected event ({selected_event_label}), so the full board is being displayed."
        )

    corrected_rows = int((board.get("odds_side_mapping_status", pd.Series(dtype=str)) == "corrected_reversed_order").sum())
    if corrected_rows:
        st.warning(
            f"Corrected {corrected_rows} reversed sportsbook odds row(s) in the dashboard display. "
            "Rerun the upcoming-events Betting Board workflow to regenerate the official artifacts with side-mapped odds."
        )

    rules = render_betting_rules_controls()
    scenario_board = apply_betting_rules(board, rules)
    render_scenario_summary(board, scenario_board)

    filtered = render_board_filters(scenario_board)
    render_summary_cards(filtered)
    render_action_board(filtered)
    render_add_official_bets_to_ledger(filtered)
    render_status_and_diagnostics(filtered)
    render_selected_fight_detail(filtered)

    with st.expander("Raw Betting Board Data", expanded=False):
        st.dataframe(board, use_container_width=True, hide_index=True)
