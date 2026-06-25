from __future__ import annotations

import pandas as pd

from pipeline.market.signals.factory import base_signal, outcome_display
from pipeline.market.signals.schema import ensure_market_signal_columns


def _latest_two_refreshes(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return latest and previous refresh slices from market intelligence history."""

    if history.empty or "refresh_id" not in history.columns:
        return pd.DataFrame(), pd.DataFrame()

    meta = (
        history[["refresh_id", "refresh_timestamp"]]
        .drop_duplicates()
        .assign(parsed_ts=lambda x: pd.to_datetime(x["refresh_timestamp"], errors="coerce", utc=True))
        .dropna(subset=["parsed_ts"])
        .sort_values("parsed_ts")
    )

    if len(meta) < 2:
        return pd.DataFrame(), pd.DataFrame()

    previous_id = meta.iloc[-2]["refresh_id"]
    latest_id = meta.iloc[-1]["refresh_id"]

    latest = history[history["refresh_id"] == latest_id].copy()
    previous = history[history["refresh_id"] == previous_id].copy()
    return latest, previous


def build_movement_signals(history: pd.DataFrame, run_id: str, timestamp: str) -> pd.DataFrame:
    """Build refresh-to-refresh line movement signals."""

    latest, previous = _latest_two_refreshes(history)
    if latest.empty or previous.empty:
        return ensure_market_signal_columns(pd.DataFrame())

    key_cols = ["bookmaker", "fight_id", "market_key", "comparison_key"]
    latest = latest.dropna(subset=key_cols + ["american_odds"]).copy()
    previous = previous.dropna(subset=key_cols + ["american_odds"]).copy()

    latest["american_odds"] = pd.to_numeric(latest["american_odds"], errors="coerce")
    previous["american_odds"] = pd.to_numeric(previous["american_odds"], errors="coerce")
    latest["implied_probability"] = pd.to_numeric(latest.get("implied_probability"), errors="coerce")
    previous["implied_probability"] = pd.to_numeric(previous.get("implied_probability"), errors="coerce")

    merged = latest.merge(
        previous[key_cols + ["american_odds", "implied_probability", "refresh_timestamp"]],
        on=key_cols,
        how="inner",
        suffixes=("", "_previous"),
    )

    if merged.empty:
        return ensure_market_signal_columns(pd.DataFrame())

    merged["line_move_cents"] = merged["american_odds"] - merged["american_odds_previous"]
    merged["line_move_probability"] = merged["implied_probability"] - merged["implied_probability_previous"]

    latest_ts = pd.to_datetime(merged["refresh_timestamp"], errors="coerce", utc=True)
    prev_ts = pd.to_datetime(merged["refresh_timestamp_previous"], errors="coerce", utc=True)
    merged["age_minutes"] = (latest_ts - prev_ts).dt.total_seconds() / 60.0

    rows = []
    moved = merged[merged["line_move_cents"].abs() >= 15].copy()

    for _, row in moved.iterrows():
        move = float(row["line_move_cents"])
        prob_move = row.get("line_move_probability")
        severity = "opportunity" if abs(move) >= 30 else "watch"
        confidence = min(0.95, 0.55 + min(abs(move), 50) / 100.0)

        direction = "improved" if move > 0 else "worsened"
        previous_odds = float(row["american_odds_previous"])
        current_odds = float(row["american_odds"])

        signal = base_signal(
            run_id=run_id,
            timestamp=timestamp,
            signal_type="line_movement",
            signal_family="movement",
            severity=severity,
            confidence_score=confidence,
            is_actionable=severity == "opportunity",
            action_label="Review move",
            row=row,
            explanation=(
                f"{row.get('bookmaker')} line {direction} for {outcome_display(row)} "
                f"from {int(previous_odds):+d} to {int(current_odds):+d} "
                f"({move:+.0f} cents) since the previous refresh."
            ),
            suggested_action="Review whether the move confirms market direction or removes available edge.",
        )
        signal.update(
            {
                "bookmakers_involved": row.get("bookmaker"),
                "book_american_odds": current_odds,
                "book_implied_probability": row.get("implied_probability"),
                "line_move_cents": move,
                "line_move_probability": prob_move,
                "age_minutes": row.get("age_minutes"),
                "snapshot_count": 2,
                "provider_count": 1,
            }
        )
        rows.append(signal)

    return ensure_market_signal_columns(pd.DataFrame(rows))

