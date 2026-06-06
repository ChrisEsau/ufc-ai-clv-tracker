"""Fighter state management for rolling UFC feature generation.

Notebook source section:
- FIGHTER STATE INITIALIZATION
- FIGHTER STATE UPDATE

Migration status:
- Migrated default_state() and update_fighter() from UFC_rolling_dataset_V4_refactored.ipynb.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pipeline.features.base.elo import RECENT_N, START_ELO, safe_div

FINISH_METHODS = ["KO/TKO", "TKO - Doctor's Stoppage", "Submission"]
KO_METHODS = ["KO/TKO", "TKO - Doctor's Stoppage"]


def default_state() -> dict[str, Any]:
    """Return the initial point-in-time state for a fighter."""
    return {
        "elo": START_ELO,
        "fights": 0,
        "wins": 0,
        "losses": 0,
        "kd_for": 0,
        "kd_against": 0,
        "sig_str_landed": 0,
        "sig_str_attempted": 0,
        "sig_str_absorbed": 0,
        "sig_str_attempted_against": 0,
        "td_landed": 0,
        "td_attempted": 0,
        "td_allowed": 0,
        "td_attempted_against": 0,
        "sub_att": 0,
        "ctrl": 0,
        "ctrl_against": 0,
        "fight_time_sec": 0,
        "finish_wins": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "decision_wins": 0,
        "finish_losses": 0,
        "decision_losses": 0,
        "opponent_elo_sum": 0,
        "best_win_elo": START_ELO,
        "worst_loss_elo": START_ELO,
        "win_streak": 0,
        "loss_streak": 0,
        "last_fight_date": None,
        "recent_results": deque(maxlen=RECENT_N),
        "recent_sig_landed": deque(maxlen=RECENT_N),
        "recent_sig_absorbed": deque(maxlen=RECENT_N),
        "recent_td_landed": deque(maxlen=RECENT_N),
        "recent_finish_results": deque(maxlen=RECENT_N),
        "recent_fight_times": deque(maxlen=RECENT_N),
    }


def update_fighter(
    fighter_state: dict[str, dict[str, Any]],
    fighter_id: str,
    fight_date: Any,
    won: bool,
    method: Any,
    own: dict[str, float],
    opp: dict[str, float],
    fight_time_sec: float,
    opponent_elo: float,
) -> None:
    """Update a fighter's rolling state after a completed fight."""
    s = fighter_state[fighter_id]
    method = str(method)

    s["opponent_elo_sum"] += opponent_elo
    s["fights"] += 1

    if won:
        s["wins"] += 1
        s["win_streak"] += 1
        s["loss_streak"] = 0
        s["best_win_elo"] = max(s["best_win_elo"], opponent_elo)

        if method in KO_METHODS:
            s["finish_wins"] += 1
            s["ko_wins"] += 1
        elif method == "Submission":
            s["finish_wins"] += 1
            s["sub_wins"] += 1
        elif "Decision" in method:
            s["decision_wins"] += 1
    else:
        s["losses"] += 1
        s["loss_streak"] += 1
        s["win_streak"] = 0
        s["worst_loss_elo"] = min(s["worst_loss_elo"], opponent_elo)

        if method in FINISH_METHODS:
            s["finish_losses"] += 1
        elif "Decision" in method:
            s["decision_losses"] += 1

    s["kd_for"] += own["kd"]
    s["kd_against"] += opp["kd"]
    s["sig_str_landed"] += own["sig_str_landed"]
    s["sig_str_attempted"] += own["sig_str_attempted"]
    s["sig_str_absorbed"] += opp["sig_str_landed"]
    s["sig_str_attempted_against"] += opp["sig_str_attempted"]
    s["td_landed"] += own["td_landed"]
    s["td_attempted"] += own["td_attempted"]
    s["td_allowed"] += opp["td_landed"]
    s["td_attempted_against"] += opp["td_attempted"]
    s["sub_att"] += own["sub_att"]
    s["ctrl"] += own["ctrl"]
    s["ctrl_against"] += opp["ctrl"]
    s["fight_time_sec"] += fight_time_sec
    s["last_fight_date"] = fight_date

    fight_minutes = fight_time_sec / 60
    finish_flag = 1 if method in FINISH_METHODS else 0

    s["recent_results"].append(1 if won else 0)
    s["recent_sig_landed"].append(safe_div(own["sig_str_landed"], fight_minutes))
    s["recent_sig_absorbed"].append(safe_div(opp["sig_str_landed"], fight_minutes))
    s["recent_td_landed"].append(own["td_landed"])
    s["recent_finish_results"].append(finish_flag)
    s["recent_fight_times"].append(fight_time_sec)
