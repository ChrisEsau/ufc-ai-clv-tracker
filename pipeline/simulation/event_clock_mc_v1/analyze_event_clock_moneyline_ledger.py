from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path).copy()
        df["source_file"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "market_family" in out.columns:
        out = out[out["market_family"].astype(str).str.upper() == "MONEYLINE"].copy()
    elif "bet_key" in out.columns:
        out = out[out["bet_key"].astype(str).str.endswith("_ML")].copy()
    else:
        raise RuntimeError("Could not identify moneyline rows.")

    for col in ["american_odds", "model_probability", "raw_implied_probability", "edge_vs_raw", "expected_roi", "flat_100_pnl"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["positive_ev_clean"] = out["expected_roi"] > 1e-9
    out["qualifies_strict_clean"] = (out["edge_vs_raw"] >= 0.05) | (out["expected_roi"] >= 0.10)
    out["price_class"] = np.where(out["american_odds"] > 0, "UNDERDOG", "FAVORITE")
    out["edge_band"] = pd.cut(
        out["edge_vs_raw"],
        bins=[-np.inf, 0.05, 0.10, 0.15, np.inf],
        labels=["<5pp", "5-10pp", "10-15pp", ">=15pp"],
        right=False,
    )
    return out


def _summary(df: pd.DataFrame, label: str) -> dict:
    graded = df[df["won"].notna()].copy()
    bets = len(graded)
    wins = int(graded["won"].astype(bool).sum()) if bets else 0
    pnl = float(graded["flat_100_pnl"].sum()) if bets else 0.0
    risk = 100.0 * bets
    return {
        "segment": label,
        "bets": bets,
        "wins": wins,
        "losses": bets - wins,
        "win_rate": wins / bets if bets else np.nan,
        "pnl": pnl,
        "roi": pnl / risk if risk else np.nan,
    }


def _print_row(row: dict) -> None:
    print(
        f"{row['segment']:<28} bets={row['bets']:>3}  "
        f"W-L={row['wins']}-{row['losses']}  "
        f"win%={row['win_rate']:.1%}  "
        f"P/L=${row['pnl']:+,.2f}  ROI={row['roi']:+.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Event Clock moneyline market ledgers and summarize edge behavior.")
    parser.add_argument("--comparison", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v1/market_comparisons/event_clock_moneyline_running_ledger.csv"))
    args = parser.parse_args()

    ml = _load(args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ml.to_csv(args.output, index=False)

    pos = ml[ml["positive_ev_clean"]].copy()
    strict = ml[ml["qualifies_strict_clean"]].copy()

    print("=" * 110)
    print("EVENT CLOCK MC — COMBINED MONEYLINE LEDGER")
    print("=" * 110)
    print(f"input cards: {len(args.comparison)}")
    print(f"moneyline rows: {len(ml)}")
    print()

    for label, frame in [("POSITIVE EV", pos), ("STRICT", strict)]:
        print(label)
        _print_row(_summary(frame, "ALL"))
        for price_class in ["UNDERDOG", "FAVORITE"]:
            _print_row(_summary(frame[frame["price_class"] == price_class], price_class))
        print()

    print("STRICT BY RAW EDGE BAND")
    for band in ["5-10pp", "10-15pp", ">=15pp"]:
        _print_row(_summary(strict[strict["edge_band"].astype(str) == band], band))

    print("\nSTRICT BETS")
    cols = [
        c for c in [
            "red", "blue", "bet_key", "american_odds", "model_probability",
            "raw_implied_probability", "edge_vs_raw", "expected_roi", "price_class",
            "edge_band", "won", "flat_100_pnl", "source_file"
        ] if c in strict.columns
    ]
    print(strict.sort_values("edge_vs_raw", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nrunning ledger CSV: {args.output}")


if __name__ == "__main__":
    main()
