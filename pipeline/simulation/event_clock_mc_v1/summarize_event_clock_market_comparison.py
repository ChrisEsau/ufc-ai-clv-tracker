from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path("data/diagnostics/event_clock_mc_v1/market_comparisons")


def _market_family(row: pd.Series) -> str:
    key = str(row.get("market_key") or "").strip().lower()
    if key == "moneyline":
        return "MONEYLINE"
    if key == "goes_distance":
        return "DISTANCE"
    if key == "fighter_sig_strikes_total":
        return "SIG_STRIKE_TOTALS"
    if key == "total_rounds":
        return "ROUND_TOTALS"
    if key in {
        "win_by_ko_tko_dq",
        "win_by_ko_tko",
        "win_by_submission",
        "submission",
        "win_by_decision",
        "decision",
    }:
        return "FIGHTER_METHOD"
    return "OTHER"


def _mark_primary_round_line(df: pd.DataFrame) -> pd.Series:
    """Identify one main round-total line per fight from the captured board.

    DraftKings canonical normalization collapses main and alternate round totals
    to the same market_key in historical artifacts. The most balanced two-sided
    line is the best reproducible proxy for the main listed total: choose the
    line whose Over/Under raw implied probabilities are jointly closest to 50%.
    """
    primary = pd.Series(False, index=df.index)
    rounds = df[df["market_family"] == "ROUND_TOTALS"].copy()
    if rounds.empty:
        return primary

    rounds["line"] = pd.to_numeric(rounds["line"], errors="coerce")
    rounds["raw_implied_probability"] = pd.to_numeric(
        rounds["raw_implied_probability"], errors="coerce"
    )

    for fight_id, fight_rows in rounds.groupby("fight_id", sort=False):
        candidates: list[tuple[float, float]] = []
        for line, line_rows in fight_rows.dropna(subset=["line"]).groupby("line"):
            valid = line_rows.dropna(subset=["raw_implied_probability"])
            if len(valid) < 2:
                continue
            probs = valid["raw_implied_probability"].astype(float)
            # Lower is better. Balanced markets cluster around 0.50 on both sides.
            score = float((probs - 0.5).abs().sum())
            candidates.append((score, float(line)))
        if not candidates:
            continue
        _, best_line = min(candidates, key=lambda x: (x[0], abs(x[1] - 2.5)))
        mask = (df["fight_id"].astype(str) == str(fight_id)) & (
            pd.to_numeric(df["line"], errors="coerce") == best_line
        ) & (df["market_family"] == "ROUND_TOTALS")
        primary.loc[mask] = True
    return primary


def _ledger_stats(rows: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    bets = rows[mask & rows["won"].notna()].copy()
    n = len(bets)
    wins = int(bets["won"].astype(bool).sum()) if n else 0
    losses = n - wins
    risk = 100.0 * n
    pnl = float(pd.to_numeric(bets["flat_100_pnl"], errors="coerce").sum()) if n else 0.0
    return {
        "strategy": label,
        "bets": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n if n else np.nan,
        "risked": risk,
        "pnl": pnl,
        "roi": pnl / risk if risk else np.nan,
    }


def _print_stats(row: dict) -> None:
    win_rate = row["win_rate"]
    roi = row["roi"]
    win_rate_text = "n/a" if pd.isna(win_rate) else f"{win_rate:.1%}"
    roi_text = "n/a" if pd.isna(roi) else f"{roi:+.1%}"
    print(
        f"{row['strategy']:<34} "
        f"bets={row['bets']:>3}  W-L={row['wins']}-{row['losses']}  "
        f"win%={win_rate_text:>6}  P/L=${row['pnl']:+,.2f}  ROI={roi_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split an Event Clock market-comparison CSV into coherent betting ledgers."
    )
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.comparison)
    required = {
        "fight_id",
        "market_key",
        "positive_ev",
        "qualifies_strict",
        "won",
        "flat_100_pnl",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Comparison CSV missing required columns: " + ", ".join(missing))

    for col in ["positive_ev", "qualifies_strict", "won"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

    df["market_family"] = df.apply(_market_family, axis=1)
    df["is_primary_round_total"] = _mark_primary_round_line(df)
    df.loc[
        (df["market_family"] == "ROUND_TOTALS") & df["is_primary_round_total"],
        "market_family",
    ] = "ROUND_TOTALS_PRIMARY"
    df.loc[
        (df["market_family"] == "ROUND_TOTALS") & ~df["is_primary_round_total"],
        "market_family",
    ] = "ROUND_TOTALS_ALTERNATE"

    prefix = args.output_prefix or args.comparison.stem.replace("_market_comparison", "")
    out_dir = args.comparison.parent if args.comparison.parent else DEFAULT_OUT_DIR
    labeled_path = out_dir / f"{prefix}_market_comparison_labeled.csv"
    summary_path = out_dir / f"{prefix}_betting_ledger_summary.csv"

    ledgers = [
        "MONEYLINE",
        "DISTANCE",
        "ROUND_TOTALS_PRIMARY",
        "ROUND_TOTALS_ALTERNATE",
        "FIGHTER_METHOD",
        "SIG_STRIKE_TOTALS",
    ]

    stats: list[dict] = []
    print("=" * 112)
    print("EVENT CLOCK MC — BETTING LEDGERS BY MARKET FAMILY")
    print("=" * 112)
    print(f"comparison rows: {len(df):,}")
    print(f"fights: {df['fight_id'].nunique():,}")

    for family in ledgers:
        fam = df[df["market_family"] == family].copy()
        if fam.empty:
            continue
        print(f"\n{family}")
        pos = _ledger_stats(fam, fam["positive_ev"].fillna(False), f"{family} — positive EV")
        strict = _ledger_stats(
            fam, fam["qualifies_strict"].fillna(False), f"{family} — strict"
        )
        stats.extend([pos, strict])
        _print_stats(pos)
        _print_stats(strict)

    # A clean headline ledger: moneyline only. This is the running ML betting
    # record and should not be mixed with correlated prop families.
    ml = df[df["market_family"] == "MONEYLINE"].copy()
    if not ml.empty:
        print("\n" + "-" * 112)
        print("MONEYLINE RUNNING-LEDGER INPUT")
        print("-" * 112)
        display_cols = [
            c
            for c in [
                "red",
                "blue",
                "bet_key",
                "american_odds",
                "model_probability",
                "raw_implied_probability",
                "no_vig_probability",
                "edge_vs_raw",
                "expected_roi",
                "positive_ev",
                "qualifies_strict",
                "won",
                "flat_100_pnl",
            ]
            if c in ml.columns
        ]
        show = ml[ml["positive_ev"].fillna(False)].sort_values(
            "expected_roi", ascending=False
        )
        print(show[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    df.to_csv(labeled_path, index=False)
    summary = pd.DataFrame(stats)
    summary.to_csv(summary_path, index=False)

    print(f"\nlabeled comparison CSV: {labeled_path}")
    print(f"ledger summary CSV:      {summary_path}")


if __name__ == "__main__":
    main()
