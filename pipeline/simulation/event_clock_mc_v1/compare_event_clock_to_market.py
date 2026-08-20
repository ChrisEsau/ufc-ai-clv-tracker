from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import _slug

HISTORY_PATH = Path("data/market/market_intelligence_history.parquet")
OUT_DIR = Path("data/diagnostics/event_clock_mc_v1/market_comparisons")


def norm(value) -> str:
    text = str(value or "").lower().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def american_profit_per_1(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_probability(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def load_market_history(path: Path, bookmaker: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Market intelligence history not found: {path}")

    frame = pd.read_parquet(path).copy()
    required = {
        "refresh_id",
        "refresh_timestamp",
        "bookmaker",
        "fight_id",
        "market_key",
        "outcome_key",
        "american_odds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Market intelligence history missing required columns: "
            + ", ".join(missing)
        )

    frame["fight_id"] = frame["fight_id"].astype("string")
    frame["refresh_timestamp"] = pd.to_datetime(
        frame["refresh_timestamp"], utc=True, errors="coerce"
    )
    frame["american_odds"] = pd.to_numeric(frame["american_odds"], errors="coerce")
    frame["implied_probability"] = pd.to_numeric(
        frame.get("implied_probability"), errors="coerce"
    )
    frame["line"] = pd.to_numeric(frame.get("line"), errors="coerce")

    book_mask = frame["bookmaker"].astype(str).str.casefold() == bookmaker.casefold()
    frame = frame[
        book_mask
        & frame["fight_id"].notna()
        & frame["refresh_timestamp"].notna()
        & frame["american_odds"].notna()
    ].copy()

    if frame.empty:
        raise RuntimeError(
            f"No {bookmaker} rows found in market intelligence history: {path}"
        )

    return frame


def select_latest_fight_snapshot(history: pd.DataFrame, fight_id: str) -> pd.DataFrame:
    rows = history[history["fight_id"].astype(str) == str(fight_id)].copy()
    if rows.empty:
        return rows

    # The history artifact is append-only. Once a fight is no longer offered,
    # later refreshes contribute no rows for that fight, so the latest refresh
    # containing the fight is its latest captured market state.
    latest = rows["refresh_timestamp"].max()
    rows = rows[rows["refresh_timestamp"] == latest].copy()

    # Deduplicate repeated canonical rows within the same refresh while keeping
    # distinct outcomes/lines/books intact.
    dedup = [
        c
        for c in (
            "fight_id",
            "bookmaker",
            "market_key",
            "outcome_key",
            "comparison_key",
            "fighter_name",
            "side",
            "line",
            "american_odds",
            "provider_market_id",
            "provider_selection_id",
        )
        if c in rows.columns
    ]
    if dedup:
        rows = rows.drop_duplicates(subset=dedup)
    return rows


def fighter_side(market_row: pd.Series, fight: pd.Series) -> str | None:
    side = norm(market_row.get("side", ""))
    if side in {"red", "blue"}:
        return side

    fighter = norm(market_row.get("fighter_name", ""))
    if fighter:
        red = norm(fight["red"])
        blue = norm(fight["blue"])
        if fighter == red:
            return "red"
        if fighter == blue:
            return "blue"

        fparts = fighter.split()
        rparts = red.split()
        bparts = blue.split()
        if fparts and rparts and fparts[-1] == rparts[-1] and fparts[-1] != bparts[-1]:
            return "red"
        if fparts and bparts and fparts[-1] == bparts[-1] and fparts[-1] != rparts[-1]:
            return "blue"

    outcome = norm(market_row.get("outcome_display", ""))
    if outcome:
        red = norm(fight["red"])
        blue = norm(fight["blue"])
        if red and red in outcome:
            return "red"
        if blue and blue in outcome:
            return "blue"

    return None


def model_probability(
    row: pd.Series,
    fight: pd.Series,
    paths: pd.DataFrame | None,
) -> tuple[float | None, str | None]:
    market_key = str(row.get("market_key") or "").strip().lower()
    outcome_key = str(row.get("outcome_key") or "").strip().lower()
    outcome_display = norm(row.get("outcome_display", ""))
    side = fighter_side(row, fight)

    if market_key == "moneyline" and side:
        return float(fight[f"p_{side}_win"]), f"{side}_ML"

    if market_key in {"win_by_ko_tko_dq", "win_by_ko_tko"} and side:
        return float(fight[f"p_{side}_ko_tko"]), f"{side}_KO_TKO"

    if market_key in {"win_by_submission", "submission"} and side:
        return float(fight[f"p_{side}_sub"]), f"{side}_SUB"

    if market_key in {"win_by_decision", "decision"} and side:
        return float(fight[f"p_{side}_dec"]), f"{side}_DEC"

    if market_key == "goes_distance":
        yes = outcome_key == "yes" or outcome_display == "yes"
        no = outcome_key == "no" or outcome_display == "no"
        if yes:
            return float(fight["p_fight_dec"]), "GOES_DISTANCE_YES"
        if no:
            return float(1.0 - fight["p_fight_dec"]), "GOES_DISTANCE_NO"

    if market_key == "fighter_sig_strikes_total" and side and paths is not None:
        if pd.isna(row.get("line")):
            return None, None
        line = float(row["line"])
        vals = paths.loc[
            paths["fight_id"].astype(str) == str(fight["fight_id"]),
            f"{side}_sig_landed",
        ].astype(float)
        if vals.empty:
            return None, None

        if outcome_key == "over" or "over" in outcome_display.split():
            return float((vals > line).mean()), f"{side}_SIG_OVER_{line:g}"
        if outcome_key == "under" or "under" in outcome_display.split():
            return float((vals < line).mean()), f"{side}_SIG_UNDER_{line:g}"

    if market_key == "total_rounds" and paths is not None and pd.notna(row.get("line")):
        line = float(row["line"])
        threshold = line * 300.0
        vals = paths.loc[
            paths["fight_id"].astype(str) == str(fight["fight_id"]),
            "elapsed",
        ].astype(float)
        if vals.empty:
            return None, None
        if outcome_key == "over" or "over" in outcome_display.split():
            return float((vals > threshold).mean()), f"TOTAL_ROUNDS_OVER_{line:g}"
        if outcome_key == "under" or "under" in outcome_display.split():
            return float((vals < threshold).mean()), f"TOTAL_ROUNDS_UNDER_{line:g}"

    return None, None


def actual_result(key: str, fight: pd.Series, market_row: pd.Series) -> bool | None:
    if key.endswith("_ML"):
        return key.split("_")[0] == str(fight["actual_winner"])
    if key.endswith("_KO_TKO"):
        return (
            key.split("_")[0] == str(fight["actual_winner"])
            and str(fight["actual_method"]) == "KO_TKO"
        )
    if key.endswith("_SUB"):
        return (
            key.split("_")[0] == str(fight["actual_winner"])
            and str(fight["actual_method"]) == "SUB"
        )
    if key.endswith("_DEC"):
        return (
            key.split("_")[0] == str(fight["actual_winner"])
            and str(fight["actual_method"]) == "DEC"
        )
    if key == "GOES_DISTANCE_YES":
        return str(fight["actual_method"]) == "DEC"
    if key == "GOES_DISTANCE_NO":
        return str(fight["actual_method"]) != "DEC"

    if "_SIG_OVER_" in key or "_SIG_UNDER_" in key:
        side = key.split("_")[0]
        line = float(market_row["line"])
        actual = float(fight[f"hist_{side}_sig_landed"])
        return actual > line if "_SIG_OVER_" in key else actual < line

    if key.startswith("TOTAL_ROUNDS_"):
        line = float(market_row["line"])
        threshold = line * 300.0
        actual = float(fight["actual_elapsed"])
        return actual > threshold if "_OVER_" in key else actual < threshold

    return None


def no_vig_probability(snapshot: pd.DataFrame, row: pd.Series) -> float | None:
    market_id = row.get("provider_market_id")
    comparison_key = row.get("comparison_key")

    if pd.notna(market_id):
        peers = snapshot[snapshot["provider_market_id"].astype(str) == str(market_id)].copy()
    elif pd.notna(comparison_key):
        peers = snapshot[snapshot["comparison_key"].astype(str) == str(comparison_key)].copy()
    else:
        return None

    peers = peers[peers["american_odds"].notna()].copy()
    if len(peers) != 2:
        return None

    probs = peers["american_odds"].astype(float).map(implied_probability)
    total = float(probs.sum())
    if total <= 0:
        return None

    selection_id = row.get("provider_selection_id")
    if pd.notna(selection_id):
        match = peers[peers["provider_selection_id"].astype(str) == str(selection_id)]
    else:
        match = peers.loc[[row.name]] if row.name in peers.index else pd.DataFrame()

    if match.empty:
        return None
    return float(probs.loc[match.index[0]] / total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Event Clock MC output to append-only market intelligence "
            "history captured by the Operations Market Refresh workflow."
        )
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--bookmaker", default="DraftKings")
    parser.add_argument("--min-edge-pp", type=float, default=5.0)
    parser.add_argument("--min-ev", type=float, default=0.10)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    summary["fight_id"] = summary["fight_id"].astype(str)

    paths = pd.read_csv(args.paths) if args.paths and args.paths.exists() else None
    if paths is not None:
        paths["fight_id"] = paths["fight_id"].astype(str)

    history = load_market_history(args.history, args.bookmaker)
    print(f"market intelligence rows loaded ({args.bookmaker}): {len(history):,}")
    print(f"refreshes: {history['refresh_id'].nunique():,}")
    print(
        "history range: "
        f"{history['refresh_timestamp'].min()} through "
        f"{history['refresh_timestamp'].max()}"
    )

    records: list[dict] = []
    missing: list[str] = []

    for _, fight in summary.iterrows():
        snapshot = select_latest_fight_snapshot(history, str(fight["fight_id"]))
        if snapshot.empty:
            missing.append(f"{fight['red']} vs {fight['blue']}")
            continue

        for _, market_row in snapshot.iterrows():
            p_model, bet_key = model_probability(market_row, fight, paths)
            if p_model is None or bet_key is None:
                continue

            odds = float(market_row["american_odds"])
            p_raw = (
                float(market_row["implied_probability"])
                if pd.notna(market_row.get("implied_probability"))
                else implied_probability(odds)
            )
            p_no_vig = no_vig_probability(snapshot, market_row)
            edge_raw = p_model - p_raw
            edge_no_vig = p_model - p_no_vig if p_no_vig is not None else np.nan
            profit1 = american_profit_per_1(odds)
            expected_roi = p_model * profit1 - (1.0 - p_model)
            won = actual_result(bet_key, fight, market_row)
            flat_100_pnl = (
                np.nan
                if won is None
                else 100.0 * profit1 if won else -100.0
            )

            records.append(
                {
                    "fight_id": fight["fight_id"],
                    "event_name": fight.get("event_name", ""),
                    "event_date": fight.get("event_date", ""),
                    "red": fight["red"],
                    "blue": fight["blue"],
                    "bookmaker": args.bookmaker,
                    "refresh_id": market_row["refresh_id"],
                    "refresh_timestamp": market_row["refresh_timestamp"],
                    "market_key": market_row["market_key"],
                    "outcome_key": market_row.get("outcome_key"),
                    "market_display": market_row.get("market_display"),
                    "outcome_display": market_row.get("outcome_display"),
                    "fighter_name": market_row.get("fighter_name"),
                    "bet_key": bet_key,
                    "line": market_row.get("line"),
                    "american_odds": odds,
                    "model_probability": p_model,
                    "raw_implied_probability": p_raw,
                    "no_vig_probability": p_no_vig,
                    "edge_vs_raw": edge_raw,
                    "edge_vs_no_vig": edge_no_vig,
                    "expected_roi": expected_roi,
                    "positive_ev": expected_roi > 0,
                    "qualifies_strict": (
                        edge_raw >= args.min_edge_pp / 100.0
                        or expected_roi >= args.min_ev
                    ),
                    "won": won,
                    "flat_100_pnl": flat_100_pnl,
                    "provider_market_id": market_row.get("provider_market_id"),
                    "provider_selection_id": market_row.get("provider_selection_id"),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError(
            "No comparable market-intelligence rows found for this Event Clock output."
        )

    dedup = [
        c
        for c in (
            "fight_id",
            "bookmaker",
            "refresh_id",
            "provider_market_id",
            "provider_selection_id",
            "bet_key",
            "line",
            "american_odds",
        )
        if c in out.columns
    ]
    out = out.drop_duplicates(subset=dedup)

    prefix = args.output_prefix or _slug(
        str(summary["event_name"].iloc[0])
        if "event_name" in summary.columns
        else args.summary.stem
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{prefix}_market_comparison.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 155)
    print("EVENT CLOCK MC — HISTORICAL MARKET COMPARISON")
    print("=" * 155)
    print(f"matched fights: {out['fight_id'].nunique()} / {summary['fight_id'].nunique()}")
    if missing:
        print("missing fights:")
        for name in missing:
            print("  -", name)

    refreshes = (
        out[["fight_id", "red", "blue", "refresh_timestamp"]]
        .drop_duplicates()
        .sort_values("refresh_timestamp")
    )
    print("\nselected market refresh by fight:")
    print(refreshes.to_string(index=False))

    display = out.sort_values(
        ["qualifies_strict", "expected_roi"], ascending=[False, False]
    )
    cols = [
        "red",
        "blue",
        "bet_key",
        "american_odds",
        "model_probability",
        "raw_implied_probability",
        "no_vig_probability",
        "edge_vs_raw",
        "expected_roi",
        "qualifies_strict",
        "won",
        "flat_100_pnl",
    ]
    print(
        "\n"
        + display[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    for label, mask in (
        ("ALL POSITIVE EV", out["positive_ev"]),
        ("STRICT QUALIFIERS", out["qualifies_strict"]),
    ):
        bets = out[mask & out["won"].notna()].copy()
        risk = 100.0 * len(bets)
        pnl = float(bets["flat_100_pnl"].sum()) if len(bets) else 0.0
        wins = int(bets["won"].astype(bool).sum()) if len(bets) else 0
        losses = len(bets) - wins
        print(f"\n{label}")
        print(f"bets: {len(bets)} | wins: {wins} | losses: {losses}")
        print(
            f"risked: ${risk:,.2f} | P/L: ${pnl:+,.2f} | "
            f"ROI: {(pnl / risk if risk else 0):+.2%}"
        )

    print(f"\ncomparison CSV: {out_path}")


if __name__ == "__main__":
    main()
