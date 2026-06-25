from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import (
    MARKET_OUTCOMES_PATH,
    MARKET_INTELLIGENCE_HISTORY_PATH,
    MARKET_SIGNALS_AUDIT_PATH,
    MARKET_SIGNALS_PATH,
)
from pipeline.market.signals.schema import (
    ensure_market_signal_audit_columns,
    ensure_market_signal_columns,
)


def _now() -> tuple[str, str]:
    dt = datetime.now(timezone.utc)
    return dt.strftime("market_signals_%Y%m%d_%H%M%S"), dt.isoformat()


def _american_to_decimal(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def _american_to_implied(odds: float | int | None) -> float | None:
    dec = _american_to_decimal(odds)
    if dec is None or dec <= 0:
        return None
    return 1.0 / dec


def _signal_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _display_fight(row: pd.Series) -> str:
    red = row.get("red_fighter")
    blue = row.get("blue_fighter")
    if pd.notna(red) and pd.notna(blue):
        return f"{red} vs {blue}"
    return str(row.get("fight_display") or row.get("event_name") or "Unknown fight")


def _outcome_display(row: pd.Series) -> str:
    for col in ["fighter_name", "provider_selection_name", "outcome_display", "side"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return "Unknown outcome"


def _market_display(row: pd.Series) -> str:
    value = row.get("market_display")
    if pd.notna(value) and str(value).strip():
        return str(value)
    value = row.get("market_key")
    return str(value or "Unknown market").replace("_", " ").title()


def _base_signal(
    *,
    run_id: str,
    timestamp: str,
    signal_type: str,
    signal_family: str,
    severity: str,
    confidence_score: float,
    is_actionable: bool,
    action_label: str,
    row: pd.Series,
    explanation: str,
    suggested_action: str,
) -> dict:
    return {
        "signal_id": _signal_id(run_id, signal_type, row.get("fight_id"), row.get("market_key"), row.get("comparison_key"), row.get("bookmaker")),
        "signal_run_id": run_id,
        "signal_timestamp": timestamp,
        "signal_type": signal_type,
        "signal_family": signal_family,
        "severity": severity,
        "confidence_score": confidence_score,
        "is_actionable": is_actionable,
        "action_label": action_label,
        "fight_id": row.get("fight_id"),
        "event_name": row.get("event_name"),
        "fight_display": _display_fight(row),
        "market_key": row.get("market_key"),
        "market_display": _market_display(row),
        "outcome_key": row.get("outcome_key"),
        "comparison_key": row.get("comparison_key"),
        "outcome_display": _outcome_display(row),
        "side": row.get("side"),
        "fighter_name": row.get("fighter_name"),
        "bookmaker": row.get("bookmaker"),
        "explanation": explanation,
        "suggested_action": suggested_action,
        "source_path": str(MARKET_OUTCOMES_PATH),
    }



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

        signal = _base_signal(
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
                f"{row.get('bookmaker')} line {direction} for {_outcome_display(row)} "
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


def build_market_signals(market_outcomes: pd.DataFrame, run_id: str, timestamp: str) -> pd.DataFrame:
    if market_outcomes.empty:
        return ensure_market_signal_columns(pd.DataFrame())

    df = market_outcomes.copy()
    df["american_odds"] = pd.to_numeric(df.get("american_odds"), errors="coerce")
    df["implied_probability"] = pd.to_numeric(df.get("implied_probability"), errors="coerce")

    rows: list[dict] = []

    group_cols = ["fight_id", "market_key", "comparison_key"]
    usable = df.dropna(subset=["fight_id", "market_key", "comparison_key", "bookmaker", "american_odds"]).copy()

    for _, group in usable.groupby(group_cols, dropna=False):
        if group["bookmaker"].nunique() < 2:
            continue

        # For American odds, larger numeric odds are better for the bettor.
        best = group.sort_values("american_odds", ascending=False).iloc[0]
        worst = group.sort_values("american_odds", ascending=True).iloc[0]

        best_odds = float(best["american_odds"])
        worst_odds = float(worst["american_odds"])
        spread_cents = best_odds - worst_odds

        best_imp = _american_to_implied(best_odds)
        worst_imp = _american_to_implied(worst_odds)
        spread_prob = abs((worst_imp or 0) - (best_imp or 0))

        provider_count = int(group["bookmaker"].nunique())
        involved = ", ".join(sorted(group["bookmaker"].astype(str).unique()))

        if abs(spread_cents) >= 10:
            severity = "opportunity" if abs(spread_cents) >= 20 else "watch"
            confidence = min(0.95, 0.55 + min(abs(spread_cents), 40) / 100.0 + provider_count * 0.05)

            row = _base_signal(
                run_id=run_id,
                timestamp=timestamp,
                signal_type="best_price_available",
                signal_family="price",
                severity=severity,
                confidence_score=confidence,
                is_actionable=severity == "opportunity",
                action_label="Line shop",
                row=best,
                explanation=(
                    f"{best.get('bookmaker')} has the best available price for "
                    f"{_outcome_display(best)} at {int(best_odds):+d}. "
                    f"The worst available price is {int(worst_odds):+d} at {worst.get('bookmaker')}."
                ),
                suggested_action="Use the best available sportsbook before price changes.",
            )
            row.update(
                {
                    "bookmakers_involved": involved,
                    "best_bookmaker": best.get("bookmaker"),
                    "best_american_odds": best_odds,
                    "best_implied_probability": best_imp,
                    "worst_bookmaker": worst.get("bookmaker"),
                    "worst_american_odds": worst_odds,
                    "worst_implied_probability": worst_imp,
                    "book_american_odds": best_odds,
                    "book_implied_probability": best_imp,
                    "spread_cents": spread_cents,
                    "spread_probability": spread_prob,
                    "provider_count": provider_count,
                }
            )
            rows.append(row)

        if abs(spread_cents) >= 20:
            row = _base_signal(
                run_id=run_id,
                timestamp=timestamp,
                signal_type="book_disagreement",
                signal_family="price",
                severity="watch",
                confidence_score=min(0.9, 0.5 + min(abs(spread_cents), 50) / 100.0),
                is_actionable=False,
                action_label="Investigate",
                row=best,
                explanation=(
                    f"Sportsbooks disagree by {abs(spread_cents):.0f} cents on "
                    f"{_outcome_display(best)}. This may indicate incomplete market consensus."
                ),
                suggested_action="Check whether the outlier book is stale or reacting slower than consensus.",
            )
            row.update(
                {
                    "bookmakers_involved": involved,
                    "best_bookmaker": best.get("bookmaker"),
                    "best_american_odds": best_odds,
                    "best_implied_probability": best_imp,
                    "worst_bookmaker": worst.get("bookmaker"),
                    "worst_american_odds": worst_odds,
                    "worst_implied_probability": worst_imp,
                    "spread_cents": spread_cents,
                    "spread_probability": spread_prob,
                    "provider_count": provider_count,
                }
            )
            rows.append(row)

    return ensure_market_signal_columns(pd.DataFrame(rows))


def main() -> None:
    print("=" * 80)
    print("BUILD MARKET SIGNALS")
    print("=" * 80)

    run_id, timestamp = _now()

    if not MARKET_OUTCOMES_PATH.exists():
        raise FileNotFoundError(f"Missing market outcomes: {MARKET_OUTCOMES_PATH}")

    market_outcomes = pd.read_parquet(MARKET_OUTCOMES_PATH)
    signal_frames = [build_market_signals(market_outcomes, run_id=run_id, timestamp=timestamp)]

    if MARKET_INTELLIGENCE_HISTORY_PATH.exists():
        history = pd.read_parquet(MARKET_INTELLIGENCE_HISTORY_PATH)
        signal_frames.append(build_movement_signals(history, run_id=run_id, timestamp=timestamp))

    signals = ensure_market_signal_columns(pd.concat(signal_frames, ignore_index=True))

    MARKET_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKET_SIGNALS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    signals.to_parquet(MARKET_SIGNALS_PATH, index=False)

    counts = signals["signal_type"].value_counts(dropna=False).to_dict() if not signals.empty else {}
    audit = ensure_market_signal_audit_columns(
        pd.DataFrame(
            [
                {
                    "signal_run_id": run_id,
                    "signal_timestamp": timestamp,
                    "source_market_rows": len(market_outcomes),
                    "output_signal_rows": len(signals),
                    "signal_type_counts": json.dumps(counts, sort_keys=True),
                    "passes_validation": True,
                    "notes": "Initial price-signal prototype.",
                }
            ]
        )
    )
    audit.to_parquet(MARKET_SIGNALS_AUDIT_PATH, index=False)

    print("Run ID:", run_id)
    print("Market rows:", len(market_outcomes))
    print("Signals:", len(signals))
    print("Signal types:")
    print(signals["signal_type"].value_counts(dropna=False).to_string() if not signals.empty else "none")
    print("Output:", MARKET_SIGNALS_PATH)
    print("Audit:", MARKET_SIGNALS_AUDIT_PATH)


if __name__ == "__main__":
    main()
