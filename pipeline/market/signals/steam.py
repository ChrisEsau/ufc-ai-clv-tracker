from __future__ import annotations

import pandas as pd

from pipeline.market.signals.factory import base_signal, outcome_display
from pipeline.market.signals.movement import _latest_two_refreshes
from pipeline.market.signals.schema import ensure_market_signal_columns


def build_steam_signals(history: pd.DataFrame, run_id: str, timestamp: str) -> pd.DataFrame:
    """Detect coordinated same-direction movement across books."""

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

    # Ignore tiny book-to-book noise.
    moved = merged[merged["line_move_cents"].abs() >= 15].copy()
    if moved.empty:
        return ensure_market_signal_columns(pd.DataFrame())

    moved["move_direction"] = moved["line_move_cents"].apply(lambda x: "up" if x > 0 else "down")

    rows: list[dict] = []
    group_cols = ["fight_id", "market_key", "comparison_key", "move_direction"]

    for _, group in moved.groupby(group_cols, dropna=False):
        provider_count = int(group["bookmaker"].nunique())
        if provider_count < 2:
            continue

        # Representative row is the largest absolute mover.
        rep = group.loc[group["line_move_cents"].abs().idxmax()]
        avg_move = float(group["line_move_cents"].mean())
        max_move = float(group["line_move_cents"].abs().max())
        avg_prob_move = float(group["line_move_probability"].mean()) if "line_move_probability" in group else None
        involved = ", ".join(sorted(group["bookmaker"].astype(str).unique()))

        direction_label = "toward longer odds" if avg_move > 0 else "toward shorter odds"
        severity = "opportunity" if provider_count >= 3 or max_move >= 30 else "watch"
        confidence = min(0.97, 0.55 + provider_count * 0.12 + min(max_move, 50) / 150.0)

        signal = base_signal(
            run_id=run_id,
            timestamp=timestamp,
            signal_type="steam_move",
            signal_family="movement",
            severity=severity,
            confidence_score=confidence,
            is_actionable=severity == "opportunity",
            action_label="Review steam",
            row=rep,
            explanation=(
                f"{provider_count} books moved {direction_label} on {outcome_display(rep)}. "
                f"Average move was {avg_move:+.0f} cents; largest move was {max_move:.0f} cents. "
                f"Books involved: {involved}."
            ),
            suggested_action="Review whether coordinated movement confirms market pressure or creates lagging-book opportunity.",
        )
        signal.update(
            {
                "bookmakers_involved": involved,
                "line_move_cents": avg_move,
                "line_move_probability": avg_prob_move,
                "age_minutes": group["age_minutes"].max(),
                "snapshot_count": 2,
                "provider_count": provider_count,
            }
        )
        rows.append(signal)

    return ensure_market_signal_columns(pd.DataFrame(rows))
