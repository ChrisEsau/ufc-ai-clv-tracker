from __future__ import annotations

import pandas as pd

from pipeline.market.signals.factory import (
    american_to_implied,
    base_signal,
    outcome_display,
)
from pipeline.market.signals.schema import ensure_market_signal_columns


def build_price_signals(market_outcomes: pd.DataFrame, run_id: str, timestamp: str) -> pd.DataFrame:
    if market_outcomes.empty:
        return ensure_market_signal_columns(pd.DataFrame())

    df = market_outcomes.copy()
    df["american_odds"] = pd.to_numeric(df.get("american_odds"), errors="coerce")
    df["implied_probability"] = pd.to_numeric(df.get("implied_probability"), errors="coerce")

    rows: list[dict] = []
    group_cols = ["fight_id", "market_key", "comparison_key"]
    usable = df.dropna(subset=group_cols + ["bookmaker", "american_odds"]).copy()

    for _, group in usable.groupby(group_cols, dropna=False):
        if group["bookmaker"].nunique() < 2:
            continue

        best = group.sort_values("american_odds", ascending=False).iloc[0]
        worst = group.sort_values("american_odds", ascending=True).iloc[0]

        best_odds = float(best["american_odds"])
        worst_odds = float(worst["american_odds"])
        spread_cents = best_odds - worst_odds

        best_imp = american_to_implied(best_odds)
        worst_imp = american_to_implied(worst_odds)
        spread_prob = abs((worst_imp or 0) - (best_imp or 0))

        provider_count = int(group["bookmaker"].nunique())
        involved = ", ".join(sorted(group["bookmaker"].astype(str).unique()))

        if abs(spread_cents) >= 10:
            severity = "opportunity" if abs(spread_cents) >= 20 else "watch"
            confidence = min(0.95, 0.55 + min(abs(spread_cents), 40) / 100.0 + provider_count * 0.05)

            row = base_signal(
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
                    f"{outcome_display(best)} at {int(best_odds):+d}. "
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
            row = base_signal(
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
                    f"{outcome_display(best)}. This may indicate incomplete market consensus."
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
