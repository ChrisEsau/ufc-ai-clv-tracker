from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from tabs import bankroll as base
from utils.github_actions import trigger_workflow

EDIT_BET_WORKFLOW = "run-edit-ledger-bet.yml"
EDIT_RESULT_OPTIONS = ["Open", "Win", "Loss", "Push", "Void"]


def _edit_bet_workflow_inputs(payload: dict) -> dict[str, str]:
    return {"edit_json": json.dumps(payload, default=str)}


def _dispatch_edit_bet(payload: dict) -> tuple[bool, str]:
    return trigger_workflow(EDIT_BET_WORKFLOW, inputs=_edit_bet_workflow_inputs(payload))


def _open_edit_bet_dialog(ledger: pd.DataFrame) -> None:
    if not hasattr(st, "dialog"):
        st.info("Your Streamlit version does not support popup dialogs. Edit controls are shown inline below.")
        return _render_edit_bet_form(ledger, in_dialog=False)

    @st.dialog("Edit Ledger Bet")
    def dialog():
        _render_edit_bet_form(ledger, in_dialog=True)

    dialog()


def _render_edit_bet_form(ledger: pd.DataFrame, in_dialog: bool = False) -> None:
    if ledger.empty:
        st.info("No ledger bets are available to edit.")
        return

    work = ledger.copy()
    work["placed_dt"] = pd.to_datetime(work.get("placed_timestamp"), errors="coerce")
    work = work.sort_values("placed_dt", ascending=False, na_position="last")
    work["label"] = work.apply(
        lambda row: (
            f"{row.get('event_name', '')} — "
            f"{row.get('fighter', '')} {base._american(row.get('odds_taken'))} "
            f"({base._money(row.get('stake'))}) — "
            f"{str(row.get('result', 'Open')).title()}"
        ),
        axis=1,
    )

    selected = st.selectbox(
        "Bet to edit",
        work.to_dict("records"),
        format_func=lambda row: row["label"],
        key="bankroll_edit_bet",
    )

    current_result = str(selected.get("result", "Open") or "Open").title()
    result = st.selectbox(
        "Result",
        EDIT_RESULT_OPTIONS,
        index=EDIT_RESULT_OPTIONS.index(current_result) if current_result in EDIT_RESULT_OPTIONS else 0,
        key="bankroll_edit_result",
    )

    odds_taken = st.number_input(
        "Odds Taken",
        min_value=-2000,
        max_value=3000,
        value=int(base._as_float(selected.get("odds_taken"), 0)),
        step=5,
        key="bankroll_edit_odds_taken",
    )
    stake = st.number_input(
        "Stake",
        min_value=0.0,
        value=base._as_float(selected.get("stake"), 0.0),
        step=25.0,
        key="bankroll_edit_stake",
    )
    closing_odds = st.number_input(
        "Closing Odds",
        min_value=-2000,
        max_value=3000,
        value=int(base._as_float(selected.get("closing_odds"), 0)),
        step=5,
        key="bankroll_edit_closing_odds",
    )
    clv = st.number_input(
        "CLV",
        min_value=-10.0,
        max_value=10.0,
        value=base._as_float(selected.get("clv"), 0.0),
        step=0.01,
        format="%.3f",
        key="bankroll_edit_clv",
    )
    notes = st.text_input("Notes", value=str(selected.get("notes", "") or ""), key="bankroll_edit_notes")

    st.caption("Profit/loss is not editable here. It will recalculate from result, stake, and odds taken.")

    if st.button("Save Bet Edit", use_container_width=True, key="bankroll_edit_submit"):
        if stake < 0 or odds_taken == 0:
            st.error("Stake cannot be negative and odds taken cannot be zero.")
            return

        payload = {
            "bet_id": selected["bet_id"],
            "result": result,
            "odds_taken": odds_taken,
            "stake": stake,
            "closing_odds": None if closing_odds == 0 else closing_odds,
            "clv": clv,
            "notes": notes,
        }
        ok, msg = _dispatch_edit_bet(payload)
        if ok:
            st.success("Edit workflow launched. Refresh after it completes to load the committed ledger.")
            st.cache_data.clear()
            if in_dialog:
                st.session_state["bankroll_dialog"] = None
            st.rerun()
        else:
            st.error(f"Could not launch edit workflow: {msg}")


def _render_action_buttons() -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Add Bet", use_container_width=True, key="bankroll_action_add"):
            st.session_state["bankroll_dialog"] = "add"
            st.rerun()
    with c2:
        if st.button("Settle Bet", use_container_width=True, key="bankroll_action_settle"):
            st.session_state["bankroll_dialog"] = "settle"
            st.rerun()
    with c3:
        if st.button("Edit Bet", use_container_width=True, key="bankroll_action_edit"):
            st.session_state["bankroll_dialog"] = "edit"
            st.rerun()
    with c4:
        if st.button("Risk Settings", use_container_width=True, key="bankroll_action_risk"):
            st.session_state["bankroll_dialog"] = "risk"
            st.rerun()


def render_bankroll() -> None:
    ledger = base.load_bet_ledger()
    if st.session_state.get("bankroll_dialog") == "edit":
        _open_edit_bet_dialog(ledger)
    _render_action_buttons()
    base.render_bankroll()
