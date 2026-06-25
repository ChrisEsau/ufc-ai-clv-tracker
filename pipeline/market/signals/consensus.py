from __future__ import annotations

import pandas as pd

from pipeline.market.signals.factory import base_signal, outcome_display
from pipeline.market.signals.schema import ensure_market_signal_columns


def _consensus_american_from_implied(prob: float | None) -> float | None:
    if prob is None or pd.isna(prob) or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def build_consensus_signals(market_outcomes: pd.DataFrame, run_id: str, timestamp: str) -> pd.DataFrame:
    if market_outcomes.empty:
        return ensure_market_signal_columns(pd.DataFrame())

    df = market_outcomes.copy()
    df["american_odds"] = pd.to_numeric(df.get("american_odds"), errors="coerce")
    df["implied_probability"] = pd.to_numeric(df.get("implied_probability"), errors="coerce")

    key_cols = ["fight_id", "market_key", "comparison_key"]
    usable = df.dropna(subset=key_cols + ["bookmaker", "american_odds", "implied_probability"]).copy()

    rows: list[dict] = []

    for _, group in usable.groupby(key_cols, dropna=False):
        provider_count = int(group["bookmaker"].nunique())
        if provider_count < 2:
            continue

        best = group.sort_values("american_odds", ascending=False).iloc[0]
        worst = group.sort_values("american_odds", ascending=True).iloc[0]

        best_odds = float(best["american_odds"])
        worst_odds = float(worst["american_odds"])
        spread_cents = best_odds - worst_odds

        consensus_prob = float(group["implied_probability"].mean())
        consensus_odds = _consensus_american_from_implied(consensus_prob)
        spread_prob = float(group["implied_probability"].max() - group["implied_probability"].min())

        # V1: only emit if disagreement is meaningful.
        if abs(spread_cents) < 25 and spread_prob < 0.04:
            continue

        involved = ", ".join(sorted(group["bookmaker"].astype(str).unique()))
        severity = "opportunity" if abs(spread_cents) >= 50 or spread_prob >= 0.08 else "watch"
        confidence = min(0.95, 0.50 + min(abs(spread_cents), 75) / 150.0 + provider_count * 0.05)

        row = base_signal(
            run_id=run_id,
            timestamp=timestamp,
            signal_type="market_consensus_gap",
            signal_family="consensus",
            severity=severity,
            confidence_score=confidence,
            is_actionable=False,
            action_label="Investigate consensus",
            row=best,
            explanation=(
                f"Books show a consensus gap on {outcome_display(best)}. "
                f"Best price is {int(best_odds):+d} at {best.get('bookmaker')}; "
                f"worst price is {int(worst_odds):+d} at {worst.get('bookmaker')}. "
                f"Consensus implied probability is {consensus_prob:.1%}."
            ),
            suggested_action="Review whether the outlier book is stale, mispriced, or reacting slower than the market.",
        )
        row.update(
            {
                "bookmakers_involved": involved,
                "best_bookmaker": best.get("bookmaker"),
                "best_american_odds": best_odds,
                "best_implied_probability": best.get("implied_probability"),
                "worst_bookmaker": worst.get("bookmaker"),
                "worst_american_odds": worst_odds,
                "worst_implied_probability": worst.get("implied_probability"),
                "consensus_american_odds": consensus_odds,
                "consensus_implied_probability": consensus_prob,
                "spread_cents": spread_cents,
                "spread_probability": spread_prob,
                "provider_count": provider_count,
            }
        )
        rows.append(row)

    return ensure_market_signal_columns(pd.DataFrame(rows))
