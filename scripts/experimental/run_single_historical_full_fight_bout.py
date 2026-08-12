"""Run one mature historical bout through KO + SUB + no-draw decision MC."""
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
    decision_winners = {0: 0, 1: 0}
    decision_scorecards: dict[tuple[int, int], int] = {}
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
        winners[winner] += 1
        methods[method] = methods.get(method, 0) + 1

        red_total = np.nan
        blue_total = np.nan
        red_rounds = np.nan
        blue_rounds = np.nan
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
            "red_score": red_total,
            "blue_score": blue_total,
            "red_rounds_won": red_rounds,
            "blue_rounds_won": blue_rounds,
        })

    n = float(args.paths)
    dec_n = methods["DEC"]
    print("\n" + "=" * 112)
    print("HISTORICAL BOUT FULL-FIGHT AUDIT — KO/TKO + SUB + NO-DRAW DECISION")
    print("=" * 112)
    print(f"bout_id: {args.bout_id}")
    print(f"RED : {red_name}")
    print(f"BLUE: {blue_name}")
    print(f"paths: {args.paths:,}")
    print(f"submission neutral P(SUB|attempt): {full.CALIBRATED_SUBMISSION_NEUTRAL_RATE:.0%}")

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
        "p_red_win_given_dec": decision_winners[0] / dec_n if dec_n else np.nan,
        "p_blue_win_given_dec": decision_winners[1] / dec_n if dec_n else np.nan,
        "draws": 0,
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)
    if args.save_paths:
        pd.DataFrame(rows).to_csv(out_dir / "paths.csv", index=False)
    print(f"\nsummary: {out_dir / 'summary.csv'}")
    print("Research-only. Judge-specific variability is not yet calibrated.")


if __name__ == "__main__":
    main()
