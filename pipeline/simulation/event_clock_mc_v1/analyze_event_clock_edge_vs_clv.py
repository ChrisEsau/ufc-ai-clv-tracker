from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v1.compare_event_clock_to_market import (
    fighter_side,
    implied_probability,
    no_vig_probability,
    select_git_fallback_snapshot,
)

OUT_DIR = Path("data/diagnostics/event_clock_mc_v1/market_comparisons")
EPS = 1e-9


def _edge_band(edge: float) -> str:
    if not np.isfinite(edge):
        return "unknown"
    if edge < 0.05:
        return "<5pp"
    if edge < 0.10:
        return "5-10pp"
    if edge < 0.15:
        return "10-15pp"
    return ">=15pp"


def _price_class(odds: float) -> str:
    return "UNDERDOG" if odds > 0 else "FAVORITE"


def _summarize(label: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"{label:28s} bets=  0")
        return
    clv = pd.to_numeric(frame["clv_probability_points"], errors="coerce")
    beat = clv > EPS
    print(
        f"{label:28s} bets={len(frame):3d}  "
        f"beat-close={beat.mean():6.1%}  "
        f"mean CLV={clv.mean()*100:+6.2f}pp  "
        f"median={clv.median()*100:+6.2f}pp  "
        f"mean model edge={frame['model_edge_vs_entry_novig'].mean()*100:+6.2f}pp"
    )


def _prepare_master() -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="coerce")
    return master


def _closing_moneyline(fight: pd.Series, bookmaker: str) -> pd.DataFrame:
    snapshot = select_git_fallback_snapshot(fight, bookmaker)
    if snapshot.empty:
        return snapshot
    ml = snapshot[snapshot["market_key"].astype(str).str.lower().isin(["moneyline", "h2h"])].copy()
    return ml.reset_index(drop=True)


def _find_closing_side_row(snapshot: pd.DataFrame, fight: pd.Series, side: str) -> pd.Series | None:
    if snapshot.empty:
        return None
    matches = []
    for idx, row in snapshot.iterrows():
        if fighter_side(row, fight) == side:
            matches.append(idx)
    if not matches:
        return None
    # There should normally be one row per side in a single DraftKings moneyline market.
    # If duplicates survive, use the last deterministic row after dedupe.
    return snapshot.loc[matches[-1]]


def audit_file(path: Path, master: pd.DataFrame, bookmaker: str) -> pd.DataFrame:
    comp = pd.read_csv(path).copy()
    required = {
        "fight_id", "red", "blue", "bet_key", "american_odds",
        "model_probability", "raw_implied_probability", "no_vig_probability",
    }
    missing = sorted(required - set(comp.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {', '.join(missing)}")

    ml = comp[comp["bet_key"].astype(str).str.endswith("_ML")].copy()
    rows = []

    master_lookup = master.set_index("fight_id", drop=False)
    for _, entry in ml.iterrows():
        fight_id = str(entry["fight_id"])
        if fight_id not in master_lookup.index:
            continue
        m = master_lookup.loc[fight_id]
        fight = pd.Series({
            "fight_id": fight_id,
            "red": str(entry["red"]),
            "blue": str(entry["blue"]),
            "event_date": m["event_date"],
        })

        side = str(entry["bet_key"]).split("_")[0].lower()
        if side not in {"red", "blue"}:
            continue

        closing = _closing_moneyline(fight, bookmaker)
        close_row = _find_closing_side_row(closing, fight, side)
        if close_row is None:
            continue

        close_odds = pd.to_numeric(pd.Series([close_row.get("american_odds")]), errors="coerce").iloc[0]
        if pd.isna(close_odds):
            continue
        close_raw = implied_probability(float(close_odds))
        close_novig = no_vig_probability(closing, close_row)
        if close_novig is None or not np.isfinite(close_novig):
            continue

        entry_raw = float(entry["raw_implied_probability"])
        entry_novig = pd.to_numeric(pd.Series([entry.get("no_vig_probability")]), errors="coerce").iloc[0]
        if pd.isna(entry_novig):
            continue
        model_p = float(entry["model_probability"])
        entry_odds = float(entry["american_odds"])
        model_edge_novig = model_p - float(entry_novig)
        model_edge_raw = model_p - entry_raw
        clv = float(close_novig) - float(entry_novig)

        rows.append({
            "source_file": str(path),
            "fight_id": fight_id,
            "red": entry["red"],
            "blue": entry["blue"],
            "bet_key": entry["bet_key"],
            "side": side,
            "american_odds_entry": entry_odds,
            "american_odds_close": float(close_odds),
            "model_probability": model_p,
            "entry_raw_implied_probability": entry_raw,
            "entry_no_vig_probability": float(entry_novig),
            "closing_raw_implied_probability": float(close_raw),
            "closing_no_vig_probability": float(close_novig),
            "model_edge_vs_entry_raw": model_edge_raw,
            "model_edge_vs_entry_novig": model_edge_novig,
            "clv_probability_points": clv,
            "residual_model_edge_at_close": model_p - float(close_novig),
            "market_closed_toward_model": bool(clv > EPS) if model_edge_novig > EPS else bool(clv < -EPS),
            "price_class": _price_class(entry_odds),
            "edge_band": _edge_band(model_edge_raw),
            "positive_ev": bool(float(entry.get("expected_roi", np.nan)) > EPS) if pd.notna(entry.get("expected_roi")) else bool(model_edge_raw > EPS),
            "qualifies_strict": bool(entry.get("qualifies_strict", False)),
            "won": bool(entry.get("won", False)),
            "entry_refresh_timestamp": entry.get("refresh_timestamp"),
            "entry_refresh_id": entry.get("refresh_id"),
            "closing_refresh_timestamp": close_row.get("refresh_timestamp"),
            "closing_refresh_id": close_row.get("refresh_id"),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Event Clock moneyline model edge at historical snapshot with DraftKings closing-line movement."
    )
    parser.add_argument("--comparison", nargs="+", required=True, type=Path)
    parser.add_argument("--bookmaker", default="DraftKings")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "event_clock_moneyline_edge_vs_clv.csv")
    args = parser.parse_args()

    master = _prepare_master()
    frames = [audit_file(path, master, args.bookmaker) for path in args.comparison]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if out.empty:
        raise RuntimeError("No moneyline rows could be matched to a valid pre-fight closing snapshot.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("=" * 130)
    print("EVENT CLOCK MC — MODEL EDGE VS CLOSING-LINE VALUE")
    print("=" * 130)
    print(f"input cards: {len(args.comparison)}")
    print(f"matched moneyline sides: {len(out)}")
    print("CLV definition: closing no-vig probability - entry no-vig probability for the same fighter")
    print("positive CLV = market moved toward that fighter")
    print("prediction probabilities changed: NO")
    print()

    positive = out[out["positive_ev"]].copy()
    strict = out[out["qualifies_strict"]].copy()

    print("POSITIVE-EV MONEYLINES")
    _summarize("ALL", positive)
    _summarize("UNDERDOG", positive[positive["price_class"] == "UNDERDOG"])
    _summarize("FAVORITE", positive[positive["price_class"] == "FAVORITE"])
    print()

    print("STRICT MONEYLINES")
    _summarize("ALL", strict)
    _summarize("UNDERDOG", strict[strict["price_class"] == "UNDERDOG"])
    _summarize("FAVORITE", strict[strict["price_class"] == "FAVORITE"])
    print()

    print("STRICT BY RAW EDGE BAND")
    for band in ["5-10pp", "10-15pp", ">=15pp"]:
        _summarize(band, strict[strict["edge_band"] == band])
    print()

    print("STRICT: WINNERS VS LOSERS")
    _summarize("WON", strict[strict["won"]])
    _summarize("LOST", strict[~strict["won"]])
    print()

    show_cols = [
        "red", "blue", "bet_key", "american_odds_entry", "american_odds_close",
        "model_probability", "entry_no_vig_probability", "closing_no_vig_probability",
        "model_edge_vs_entry_novig", "clv_probability_points", "residual_model_edge_at_close",
        "price_class", "edge_band", "won",
    ]
    print("STRICT BETS — EDGE VS CLV")
    if strict.empty:
        print("none")
    else:
        print(strict.sort_values("model_edge_vs_entry_novig", ascending=False)[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"edge-vs-CLV CSV: {args.output}")


if __name__ == "__main__":
    main()
