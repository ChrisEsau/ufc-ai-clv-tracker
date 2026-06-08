from __future__ import annotations

import pandas as pd


LEGACY_BETTING_BOARD_COLUMNS = [
    "event_id",
    "event_name",
    "commence_time",
    "fight_id",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "bookmaker",
    "sportsbook",
    "market_key",
    "red_model_prob",
    "blue_model_prob",
    "red_american_odds",
    "blue_american_odds",
    "red_implied_prob",
    "blue_implied_prob",
    "red_edge",
    "blue_edge",
    "red_ev",
    "blue_ev",
    "red_ev_dollars",
    "blue_ev_dollars",
    "best_side",
    "best_fighter_id",
    "best_prob",
    "best_american_odds",
    "best_implied_prob",
    "best_edge",
    "best_ev",
    "best_confidence",
    "confidence_pct",
    "recommended_stake",
    "bet_status",
    "decision_timestamp",
    "snapshot_timestamp",
]


def _empty_legacy_board() -> pd.DataFrame:
    return pd.DataFrame(columns=LEGACY_BETTING_BOARD_COLUMNS)


def _pick_row(rows: pd.DataFrame, fighter_id) -> pd.Series | None:
    if rows.empty or fighter_id is None or pd.isna(fighter_id):
        return None
    matched = rows[rows["outcome_fighter_id"].astype(str) == str(fighter_id)]
    if matched.empty:
        return None
    return matched.iloc[0]


def _value(row: pd.Series | None, column: str, default=None):
    if row is None:
        return default
    return row.get(column, default)


def _best_outcome(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    if "is_bet_candidate" in rows.columns and rows["is_bet_candidate"].fillna(False).any():
        candidates = rows[rows["is_bet_candidate"].fillna(False)].copy()
        return candidates.sort_values("ev_dollars_at_100", ascending=False, na_position="last").iloc[0]
    sort_column = "ev_dollars_at_100" if "ev_dollars_at_100" in rows.columns else "ev_pct"
    return rows.sort_values(sort_column, ascending=False, na_position="last").iloc[0]


def betting_outcomes_to_legacy_board(betting_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Convert outcome-level Betting Outcomes V2 rows into the legacy fight-level board.

    This adapter lets the current Streamlit Betting Board render V2 data without
    changing the existing UI layout. It currently emits one row per fight/bookmaker
    for moneyline-style two-fighter markets.
    """

    if betting_outcomes is None or betting_outcomes.empty:
        return _empty_legacy_board()

    df = betting_outcomes.copy()
    if "market_key" in df.columns:
        df = df[df["market_key"].astype(str).str.lower().isin(["moneyline", "h2h"])]
    if df.empty:
        return _empty_legacy_board()

    rows = []
    group_cols = [col for col in ["fight_id", "bookmaker", "market_key"] if col in df.columns]
    if not group_cols:
        return _empty_legacy_board()

    for _, group in df.groupby(group_cols, dropna=False):
        first = group.iloc[0]
        red = _pick_row(group, first.get("red_fighter_id"))
        blue = _pick_row(group, first.get("blue_fighter_id"))
        best = _best_outcome(group)

        if red is None or blue is None or best is None:
            continue

        best_fighter_id = best.get("outcome_fighter_id")
        if str(best_fighter_id) == str(first.get("red_fighter_id")):
            best_side = first.get("red_fighter")
        elif str(best_fighter_id) == str(first.get("blue_fighter_id")):
            best_side = first.get("blue_fighter")
        else:
            best_side = best.get("outcome_label")

        rows.append(
            {
                "event_id": first.get("event_id"),
                "event_name": first.get("event_name"),
                "commence_time": first.get("commence_time"),
                "fight_id": first.get("fight_id"),
                "red_fighter": first.get("red_fighter"),
                "blue_fighter": first.get("blue_fighter"),
                "red_fighter_id": first.get("red_fighter_id"),
                "blue_fighter_id": first.get("blue_fighter_id"),
                "bookmaker": first.get("bookmaker"),
                "sportsbook": first.get("bookmaker"),
                "market_key": first.get("market_key"),
                "red_model_prob": _value(red, "model_probability"),
                "blue_model_prob": _value(blue, "model_probability"),
                "red_american_odds": _value(red, "american_odds"),
                "blue_american_odds": _value(blue, "american_odds"),
                "red_implied_prob": _value(red, "implied_probability"),
                "blue_implied_prob": _value(blue, "implied_probability"),
                "red_edge": _value(red, "edge"),
                "blue_edge": _value(blue, "edge"),
                "red_ev": _value(red, "ev"),
                "blue_ev": _value(blue, "ev"),
                "red_ev_dollars": _value(red, "ev_dollars_at_100"),
                "blue_ev_dollars": _value(blue, "ev_dollars_at_100"),
                "best_side": best_side,
                "best_fighter_id": best_fighter_id,
                "best_prob": best.get("model_probability"),
                "best_american_odds": best.get("american_odds"),
                "best_implied_prob": best.get("implied_probability"),
                "best_edge": best.get("edge"),
                "best_ev": best.get("ev_dollars_at_100"),
                "best_confidence": best.get("confidence_pct"),
                "confidence_pct": best.get("confidence_pct"),
                "recommended_stake": best.get("recommended_stake"),
                "bet_status": "OFFICIAL BET" if bool(best.get("is_bet_candidate", False)) else str(best.get("bet_status", "NO BET")),
                "decision_timestamp": best.get("betting_timestamp"),
                "snapshot_timestamp": best.get("snapshot_timestamp"),
            }
        )

    if not rows:
        return _empty_legacy_board()

    out = pd.DataFrame(rows)
    for column in LEGACY_BETTING_BOARD_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[LEGACY_BETTING_BOARD_COLUMNS]
