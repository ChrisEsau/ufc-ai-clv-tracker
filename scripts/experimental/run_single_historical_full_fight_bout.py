"""Run one mature historical bout through KO + SUB + no-draw decision MC.

The runner also translates simulated path frequencies into a small betting board
of model probabilities and no-vig fair American odds. These are model prices,
not sportsbook lines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
DEFAULT_OUTPUT_DIR = Path("data/experimental/single_historical_full_fight")
ROUND_SECONDS = 300.0
SEGMENT_SECONDS = 10.0


def _select_bout(bout_id: str):
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    selected = cohort.loc[cohort["bout_id"].eq(str(bout_id))]
    if selected.empty:
        raise ValueError(f"bout_id {bout_id} is not in aligned mature 2020+ FSR-32 cohort")
    if len(selected) != 1:
        raise ValueError(f"expected one row for {bout_id}, found {len(selected)}")
    return selected.iloc[0], pairs[str(bout_id)]


def _age(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _fair_american(probability: float) -> str:
    """Convert a model probability to no-vig fair American odds."""
    p = float(probability)
    if p <= 0.0:
        return "+INF"
    if p >= 1.0:
        return "-INF"
    if p >= 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    return f"{odds:+.0f}"


def _elapsed_seconds(path) -> float:
    """Return simulated fight duration in seconds for totals markets."""
    if path.finish is None:
        return 3.0 * ROUND_SECONDS
    round_no = int(path.finish.round or 1)
    segment_no = int(path.finish.segment or 1)
    return (round_no - 1) * ROUND_SECONDS + segment_no * SEGMENT_SECONDS


def _market_row(market: str, selection: str, wins: int, paths: int) -> dict[str, object]:
    probability = wins / float(paths)
    return {
        "market": market,
        "selection": selection,
        "wins": int(wins),
        "paths": int(paths),
        "model_probability": probability,
        "fair_american_odds": _fair_american(probability),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bout-id", required=True)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-paths", action="store_true")
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    bout, pair = _select_bout(str(args.bout_id))
    red, blue = pair
    red_name = base._display_name(red)
    blue_name = base._display_name(blue)
    red_age = _age(bout, "r_age")
    blue_age = _age(bout, "b_age")

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    winners = {0: 0, 1: 0}
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    winner_methods = {
        0: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
        1: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
    }
    decision_winners = {0: 0, 1: 0}
    decision_scorecards: dict[tuple[int, int], int] = {}
    totals = {
        "over_1_5": 0,
        "under_1_5": 0,
        "over_2_5": 0,
        "under_2_5": 0,
    }
    rows = []
    example_decision = None

    for seed in seeds:
        sim = full.StaticFSRMCFullFightV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()
        winner = int(path.winner)
        method = path.method
        elapsed_seconds = _elapsed_seconds(path)

        winners[winner] += 1
        methods[method] = methods.get(method, 0) + 1
        winner_methods[winner][method] += 1

        # UFC round-total convention: 1.5 = 2:30 of R2; 2.5 = 2:30 of R3.
        if elapsed_seconds > 1.5 * ROUND_SECONDS:
            totals["over_1_5"] += 1
        else:
            totals["under_1_5"] += 1
        if elapsed_seconds > 2.5 * ROUND_SECONDS:
            totals["over_2_5"] += 1
        else:
            totals["under_2_5"] += 1

        red_total = np.nan
        blue_total = np.nan
        red_rounds = np.nan
        blue_rounds = np.nan
        finish_round = np.nan
        finish_segment = np.nan
        if path.finish is not None:
            finish_round = int(path.finish.round or 0)
            finish_segment = int(path.finish.segment or 0)
        if path.decision is not None:
            decision_winners[winner] += 1
            red_total = path.decision.red_total
            blue_total = path.decision.blue_total
            red_rounds = path.decision.red_rounds_won
            blue_rounds = path.decision.blue_rounds_won
            decision_scorecards[(red_total, blue_total)] = decision_scorecards.get((red_total, blue_total), 0) + 1
            if example_decision is None:
                example_decision = path.decision

        rows.append({
            "seed": int(seed),
            "winner_side": "red" if winner == 0 else "blue",
            "method": method,
            "elapsed_seconds": elapsed_seconds,
            "finish_round": finish_round,
            "finish_segment": finish_segment,
            "red_score": red_total,
            "blue_score": blue_total,
            "red_rounds_won": red_rounds,
            "blue_rounds_won": blue_rounds,
        })

    n = float(args.paths)
    dec_n = methods["DEC"]

    market_rows = [
        _market_row("Moneyline", red_name, winners[0], args.paths),
        _market_row("Moneyline", blue_name, winners[1], args.paths),
        _market_row("Fight Method", "KO/TKO", methods["KO/TKO"], args.paths),
        _market_row("Fight Method", "Submission", methods["SUB"], args.paths),
        _market_row("Fight Method", "Decision", methods["DEC"], args.paths),
        _market_row("Goes Distance", "Yes", methods["DEC"], args.paths),
        _market_row("Goes Distance", "No", args.paths - methods["DEC"], args.paths),
        _market_row("Total Rounds 1.5", "Over 1.5", totals["over_1_5"], args.paths),
        _market_row("Total Rounds 1.5", "Under 1.5", totals["under_1_5"], args.paths),
        _market_row("Total Rounds 2.5", "Over 2.5", totals["over_2_5"], args.paths),
        _market_row("Total Rounds 2.5", "Under 2.5", totals["under_2_5"], args.paths),
    ]
    for side, name in ((0, red_name), (1, blue_name)):
        market_rows.extend([
            _market_row("Method of Victory", f"{name} by KO/TKO", winner_methods[side]["KO/TKO"], args.paths),
            _market_row("Method of Victory", f"{name} by Submission", winner_methods[side]["SUB"], args.paths),
            _market_row("Method of Victory", f"{name} by Decision", winner_methods[side]["DEC"], args.paths),
        ])
    markets = pd.DataFrame(market_rows)

    print("\n" + "=" * 112)
    print("HISTORICAL BOUT FULL-FIGHT AUDIT — KO/TKO + SUB + NO-DRAW DECISION")
    print("=" * 112)
    print(f"bout_id: {args.bout_id}")
    print(f"RED : {red_name}")
    print(f"BLUE: {blue_name}")
    print(f"paths: {args.paths:,}")
    print(f"submission neutral P(SUB|attempt): {full.CALIBRATED_SUBMISSION_NEUTRAL_RATE:.0%}")

    print("\nMODEL BETTING BOARD — NO-VIG FAIR PRICES")
    display = markets.copy()
    display["model_probability"] = display["model_probability"].map(lambda x: f"{x:.2%}")
    print(display[["market", "selection", "model_probability", "fair_american_odds"]].to_string(index=False))

    print("\nMONEYLINE WINNER")
    print(f"RED  {red_name}: {winners[0] / n:.2%} ({winners[0]:,})")
    print(f"BLUE {blue_name}: {winners[1] / n:.2%} ({winners[1]:,})")
    print(f"sum: {(winners[0] + winners[1]) / n:.2%} | draws: 0")

    print("\nMETHOD")
    for method in ("KO/TKO", "SUB", "DEC"):
        print(f"{method:6s}: {methods[method] / n:.2%} ({methods[method]:,})")

    print("\nDECISION-ONLY WINNER")
    if dec_n:
        print(f"RED : {decision_winners[0] / dec_n:.2%} ({decision_winners[0]:,}/{dec_n:,})")
        print(f"BLUE: {decision_winners[1] / dec_n:.2%} ({decision_winners[1]:,}/{dec_n:,})")
    else:
        print("No paths reached a decision.")

    if example_decision is not None:
        print("\nEXAMPLE DECISION SCORECARD")
        print(f"final: {example_decision.red_total}-{example_decision.blue_total}")
        for score in example_decision.round_scores:
            winner_name = red_name if score.winner == 0 else blue_name
            print(
                f"R{score.round_number}: {score.red_points}-{score.blue_points} {winner_name} | "
                f"margin={score.margin:+.3f} | "
                f"sig={score.red.sig_landed}-{score.blue.sig_landed} | "
                f"KD={score.red.knockdowns}-{score.blue.knockdowns} | "
                f"TD={score.red.td_landed}-{score.blue.td_landed} | "
                f"SUBatt={score.red.sub_att}-{score.blue.sub_att} | "
                f"CTRL={score.red.control_seconds}-{score.blue.control_seconds}s | "
                f"tie_break={score.tie_break_used}"
            )

    out_dir = args.output_dir / str(args.bout_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "bout_id": str(args.bout_id),
        "red_name": red_name,
        "blue_name": blue_name,
        "paths": args.paths,
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n,
        "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
        "p_red_ko_tko": winner_methods[0]["KO/TKO"] / n,
        "p_blue_ko_tko": winner_methods[1]["KO/TKO"] / n,
        "p_red_sub": winner_methods[0]["SUB"] / n,
        "p_blue_sub": winner_methods[1]["SUB"] / n,
        "p_red_dec": winner_methods[0]["DEC"] / n,
        "p_blue_dec": winner_methods[1]["DEC"] / n,
        "p_over_1_5": totals["over_1_5"] / n,
        "p_under_1_5": totals["under_1_5"] / n,
        "p_over_2_5": totals["over_2_5"] / n,
        "p_under_2_5": totals["under_2_5"] / n,
        "p_red_win_given_dec": decision_winners[0] / dec_n if dec_n else np.nan,
        "p_blue_win_given_dec": decision_winners[1] / dec_n if dec_n else np.nan,
        "draws": 0,
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)
    markets.to_csv(out_dir / "model_markets.csv", index=False)
    if args.save_paths:
        pd.DataFrame(rows).to_csv(out_dir / "paths.csv", index=False)
    print(f"\nsummary:       {out_dir / 'summary.csv'}")
    print(f"model markets: {out_dir / 'model_markets.csv'}")
    if args.save_paths:
        print(f"paths:         {out_dir / 'paths.csv'}")
    print("Research-only. Fair odds are model-derived no-vig prices, not sportsbook lines.")


if __name__ == "__main__":
    main()
