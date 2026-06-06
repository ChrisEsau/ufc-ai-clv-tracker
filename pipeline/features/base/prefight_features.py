"""Prefight feature extraction for rolling UFC base features.

Notebook source section:
- PREFIGHT FEATURE FUNCTION

Migration status:
- Migrated get_prefight_features() from UFC_rolling_dataset_V4_refactored.ipynb.
"""

from __future__ import annotations

from typing import Any

from pipeline.features.base.elo import safe_div


def get_prefight_features(
    fighter_state: dict[str, dict[str, Any]],
    fighter_id: str,
    fight_date: Any,
) -> dict[str, float]:
    """Return point-in-time prefight features for a fighter.

    The function is parameterized with ``fighter_state`` rather than relying on
    the notebook-global ``fighter_state`` object. This preserves the notebook
    behavior while making the code reusable for production builders and tests.
    """
    s = fighter_state[fighter_id]

    minutes = s["fight_time_sec"] / 60
    fifteen_min_units = s["fight_time_sec"] / 900

    if s["last_fight_date"] is None:
        days_since_last_fight = 365
    else:
        days_since_last_fight = (fight_date - s["last_fight_date"]).days

    return {
        "elo": s["elo"],
        "fights": s["fights"],
        "wins": s["wins"],
        "losses": s["losses"],
        "win_pct": safe_div(s["wins"], s["fights"]),
        "kd_avg": safe_div(s["kd_for"], s["fights"]),
        "kd_absorbed_avg": safe_div(s["kd_against"], s["fights"]),
        "splm": safe_div(s["sig_str_landed"], minutes),
        "sapm": safe_div(s["sig_str_absorbed"], minutes),
        "str_acc": safe_div(s["sig_str_landed"], s["sig_str_attempted"]),
        "str_def": 1 - safe_div(s["sig_str_absorbed"], s["sig_str_attempted_against"]),
        "td_avg": safe_div(s["td_landed"], fifteen_min_units),
        "td_acc": safe_div(s["td_landed"], s["td_attempted"]),
        "td_def": 1 - safe_div(s["td_allowed"], s["td_attempted_against"]),
        "sub_avg": safe_div(s["sub_att"], fifteen_min_units),
        "ctrl_per_min": safe_div(s["ctrl"], minutes),
        "ctrl_against_per_min": safe_div(s["ctrl_against"], minutes),
        "finish_rate": safe_div(s["finish_wins"], s["wins"]),
        "ko_rate": safe_div(s["ko_wins"], s["wins"]),
        "sub_win_rate": safe_div(s["sub_wins"], s["wins"]),
        "decision_win_rate": safe_div(s["decision_wins"], s["wins"]),
        "finish_loss_rate": safe_div(s["finish_losses"], s["losses"]),
        "decision_loss_rate": safe_div(s["decision_losses"], s["losses"]),
        "avg_opponent_elo": safe_div(s["opponent_elo_sum"], s["fights"]),
        "best_win_elo": s["best_win_elo"],
        "worst_loss_elo": s["worst_loss_elo"],
        "avg_fight_time": safe_div(s["fight_time_sec"], s["fights"]),
        "win_streak": s["win_streak"],
        "loss_streak": s["loss_streak"],
        "days_since_last_fight": days_since_last_fight,
        "recent_win_pct": safe_div(sum(s["recent_results"]), len(s["recent_results"])),
        "recent_splm": safe_div(sum(s["recent_sig_landed"]), len(s["recent_sig_landed"])),
        "recent_sapm": safe_div(sum(s["recent_sig_absorbed"]), len(s["recent_sig_absorbed"])),
        "recent_td_avg": safe_div(sum(s["recent_td_landed"]), len(s["recent_td_landed"])),
        "recent_finish_rate": safe_div(sum(s["recent_finish_results"]), len(s["recent_finish_results"])),
        "recent_avg_fight_time": safe_div(sum(s["recent_fight_times"]), len(s["recent_fight_times"])),
    }
