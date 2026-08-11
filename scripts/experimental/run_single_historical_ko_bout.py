"""Run one historical mature-cohort UFC bout through the current KO/TKO simulator.

This is a research/audit convenience runner. It reuses the exact current shadow
configuration used by the full mature 2020+ KO validation:

- FSR-32 leakage-safe historical pre-fight profiles
- 10-second segments
- 3-round horizon
- fatigue exponent 2.0
- R2 entry recovery: 20% missing damage, 40% missing stamina
- R3 entry recovery: 60% missing damage, 0% missing stamina
- KD base -8.80
- collapse scale 2.0
- collapse curvature 16.0
- current damage reservoir / striking-power architecture unchanged

The default is 1,000 Monte Carlo paths for one historical bout_id. No judging
layer is used. The output is strictly KO/TKO-oriented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_full_cohort_ko_validation_r3_d60_s0 as full
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as population
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental.inspect_historical_bout_fsr32 import (
    HIGHLIGHT_TRAITS,
    _numeric_fsr_columns,
)

DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
DEFAULT_OUTPUT_DIR = Path("data/experimental/single_historical_ko_bout")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one historical mature-cohort bout through the current KO/TKO simulator"
    )
    p.add_argument("--bout-id", required=True, help="Canonical UFC fight_id / bout_id")
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--save-paths",
        action="store_true",
        help="Persist one CSV row per Monte Carlo path in addition to the summary",
    )
    return p.parse_args()


def _canonical_master_row(bout_id: str) -> pd.Series:
    raw = pd.read_parquet(population.modern.MASTER_PATH).copy()
    raw["fight_id"] = raw["fight_id"].astype(str)
    row = raw.loc[raw["fight_id"].eq(str(bout_id))].copy()
    if row.empty:
        raise ValueError(f"bout_id not found in UFC master: {bout_id}")

    date_col = population.modern._resolve_date_column(row)
    row[date_col] = pd.to_datetime(row[date_col], errors="coerce")
    row = row.sort_values([date_col, "fight_id"]).iloc[-1]
    return row


def _historical_winner_side(master: pd.Series) -> str:
    winner_id = str(master.get("winner_id", ""))
    if winner_id == str(master.get("r_id", "")):
        return "red"
    if winner_id == str(master.get("b_id", "")):
        return "blue"
    return "unknown"


def _select_bout(bout_id: str):
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    selected = cohort.loc[cohort["bout_id"].eq(str(bout_id))]
    if selected.empty:
        raise ValueError(
            f"bout_id {bout_id} is not in the aligned mature 2020+ FSR cohort"
        )
    if len(selected) != 1:
        raise ValueError(f"expected one cohort row for {bout_id}, found {len(selected)}")

    bout = selected.iloc[0].copy()
    pair = pairs[str(bout_id)]
    return bout, pair


def _profile_rows(red: pd.Series, blue: pd.Series, traits: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trait in traits:
        if trait not in red.index or trait not in blue.index:
            continue
        rv = pd.to_numeric(pd.Series([red[trait]]), errors="coerce").iloc[0]
        bv = pd.to_numeric(pd.Series([blue[trait]]), errors="coerce").iloc[0]
        if pd.isna(rv) and pd.isna(bv):
            continue
        diff = rv - bv if pd.notna(rv) and pd.notna(bv) else np.nan
        rows.append({"trait": trait, "red": rv, "blue": bv, "red_minus_blue": diff})
    return rows


def _round_entry_effective_state(sim) -> dict[int, dict[int, dict[str, float]]]:
    """Read segment-1 effective state already emitted by the rolling-FSR engine."""
    result: dict[int, dict[int, dict[str, float]]] = {}
    for event in sim.effective_fsr_events:
        round_no = int(event["round"])
        segment_no = int(event["segment"])
        fighter = int(event["fighter"])
        if segment_no != 1 or round_no not in (1, 2, 3):
            continue
        result.setdefault(round_no, {})[fighter] = {
            "stamina_fraction": float(event["stamina_fraction"]),
            "fatigue_penalty": float(event["fatigue_penalty"]),
            "effective_striking_power": float(event["effective_striking_power"]),
        }
    return result


def _run_one_path(
    red: pd.Series,
    blue: pd.Series,
    *,
    seed: int,
    red_age: float | None,
    blue_age: float | None,
) -> tuple[dict[str, object], dict[int, dict[str, int]], dict[int, dict[int, dict[str, float]]]]:
    # Prefix runs intentionally mirror the full-cohort validator so R1/R2/R3
    # incremental activity is directly comparable to that benchmark.
    sim1, path1, kd1, fr1 = full._run_prefix(
        red,
        blue,
        rounds=1,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )
    sim2, path2, kd2, fr2 = full._run_prefix(
        red,
        blue,
        rounds=2,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )
    sim3, path3, kd3, fr3 = full._run_prefix(
        red,
        blue,
        rounds=3,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )

    sig1 = int(sim1.stats[0].sig_landed) + int(sim1.stats[1].sig_landed)
    sig2_cum = int(sim2.stats[0].sig_landed) + int(sim2.stats[1].sig_landed)
    sig3_cum = int(sim3.stats[0].sig_landed) + int(sim3.stats[1].sig_landed)

    rounds = {
        1: {
            "reached": 1,
            "sig": sig1,
            "kd": int(kd1),
            "ko": int(path1.finish is not None and fr1 == 1),
        },
        2: {
            "reached": int(path1.finish is None),
            "sig": max(0, sig2_cum - sig1) if path1.finish is None else 0,
            "kd": max(0, int(kd2) - int(kd1)) if path1.finish is None else 0,
            "ko": int(path2.finish is not None and fr2 == 2) if path1.finish is None else 0,
        },
        3: {
            "reached": int(path2.finish is None),
            "sig": max(0, sig3_cum - sig2_cum) if path2.finish is None else 0,
            "kd": max(0, int(kd3) - int(kd2)) if path2.finish is None else 0,
            "ko": int(path3.finish is not None and fr3 == 3) if path2.finish is None else 0,
        },
    }

    winner_side = "none"
    finish_round = np.nan
    finish_segment = np.nan
    finish_clock = ""
    if path3.finish is not None:
        winner_side = "red" if int(path3.finish.winner) == 0 else "blue"
        finish_round = int(fr3)
        finish_segment = int(path3.finish.segment) if path3.finish.segment is not None else np.nan
        finish_clock = str(path3.finish.clock_start or "")

    path_row = {
        "seed": int(seed),
        "ko_tko": int(path3.finish is not None),
        "winner_side": winner_side,
        "finish_round": finish_round,
        "finish_segment": finish_segment,
        "finish_clock_start": finish_clock,
        "sig_landed_total": sig3_cum,
        "knockdowns_total": int(kd3),
        "terminal_collapse_finish": int(sim3.terminal_collapse_finishes > 0),
        "direct_strike_finish": int(sim3.direct_strike_finishes > 0),
        "r1_sig": rounds[1]["sig"],
        "r1_kd": rounds[1]["kd"],
        "r2_reached": rounds[2]["reached"],
        "r2_sig": rounds[2]["sig"],
        "r2_kd": rounds[2]["kd"],
        "r3_reached": rounds[3]["reached"],
        "r3_sig": rounds[3]["sig"],
        "r3_kd": rounds[3]["kd"],
    }
    return path_row, rounds, _round_entry_effective_state(sim3)


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    full._configure_locked_candidate()

    bout_id = str(args.bout_id)
    bout, pair = _select_bout(bout_id)
    red, blue = pair
    master = _canonical_master_row(bout_id)

    # Verify the historical FSR pair matches canonical corners. This protects
    # fighter-direction metrics from parquet row-order/corner drift.
    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    if str(red["fighter_id"]) != r_id or str(blue["fighter_id"]) != b_id:
        raise ValueError(
            "FSR pair is not aligned to historical red/blue corners: "
            f"pair=({red['fighter_id']}, {blue['fighter_id']}), master=({r_id}, {b_id})"
        )

    red_age = float(bout["r_age"]) if pd.notna(bout.get("r_age")) else None
    blue_age = float(bout["b_age"]) if pd.notna(bout.get("b_age")) else None

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)

    totals = {
        1: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
        2: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
        3: {"reached": 0, "sig": 0, "kd": 0, "ko": 0},
    }
    effective_samples = {
        rnd: {
            fighter: {"stamina_fraction": [], "fatigue_penalty": [], "effective_striking_power": []}
            for fighter in (0, 1)
        }
        for rnd in (1, 2, 3)
    }
    path_rows: list[dict[str, object]] = []

    for i, seed in enumerate(seeds, start=1):
        path_row, rounds, effective = _run_one_path(
            red,
            blue,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path_rows.append(path_row)
        for rnd in (1, 2, 3):
            for key in ("reached", "sig", "kd", "ko"):
                totals[rnd][key] += int(rounds[rnd][key])
            for fighter in (0, 1):
                state = effective.get(rnd, {}).get(fighter)
                if state is None:
                    continue
                for key in ("stamina_fraction", "fatigue_penalty", "effective_striking_power"):
                    effective_samples[rnd][fighter][key].append(float(state[key]))
        if i % 250 == 0 or i == args.paths:
            print(f"paths {i:,}/{args.paths:,}", flush=True)

    paths = pd.DataFrame(path_rows)
    red_ko = int(paths["winner_side"].eq("red").sum())
    blue_ko = int(paths["winner_side"].eq("blue").sum())
    any_ko = red_ko + blue_ko
    no_ko = args.paths - any_ko

    p_red = red_ko / args.paths
    p_blue = blue_ko / args.paths
    p_any = any_ko / args.paths
    predicted_side = "tie"
    if p_red > p_blue:
        predicted_side = "red"
    elif p_blue > p_red:
        predicted_side = "blue"

    historical_winner_side = _historical_winner_side(master)
    historical_method = str(master.get("method", ""))
    historical_finish_round = master.get("finish_round", bout.get("actual_finish_round", ""))

    summary = {
        "bout_id": bout_id,
        "event_date": str(master.get(population.modern._resolve_date_column(pd.DataFrame([master])), bout.get("event_date", ""))),
        "red_name": base._display_name(red),
        "blue_name": base._display_name(blue),
        "r_id": r_id,
        "b_id": b_id,
        "red_age": red_age,
        "blue_age": blue_age,
        "historical_winner_id": str(master.get("winner_id", "")),
        "historical_winner_side": historical_winner_side,
        "historical_method": historical_method,
        "historical_finish_round": historical_finish_round,
        "paths": args.paths,
        "p_red_ko": p_red,
        "p_blue_ko": p_blue,
        "p_any_ko": p_any,
        "p_no_ko": no_ko / args.paths,
        "predicted_ko_winner_side": predicted_side,
        "predicted_ko_winner_correct": (
            int(predicted_side == historical_winner_side)
            if historical_winner_side in {"red", "blue"} and predicted_side != "tie"
            else np.nan
        ),
        "r1_ko_rate": totals[1]["ko"] / totals[1]["reached"] if totals[1]["reached"] else np.nan,
        "r2_ko_rate": totals[2]["ko"] / totals[2]["reached"] if totals[2]["reached"] else np.nan,
        "r3_ko_rate": totals[3]["ko"] / totals[3]["reached"] if totals[3]["reached"] else np.nan,
        "r1_sig_mean": totals[1]["sig"] / totals[1]["reached"] if totals[1]["reached"] else np.nan,
        "r2_sig_mean": totals[2]["sig"] / totals[2]["reached"] if totals[2]["reached"] else np.nan,
        "r3_sig_mean": totals[3]["sig"] / totals[3]["reached"] if totals[3]["reached"] else np.nan,
        "r1_kd_mean": totals[1]["kd"] / totals[1]["reached"] if totals[1]["reached"] else np.nan,
        "r2_kd_mean": totals[2]["kd"] / totals[2]["reached"] if totals[2]["reached"] else np.nan,
        "r3_kd_mean": totals[3]["kd"] / totals[3]["reached"] if totals[3]["reached"] else np.nan,
        "terminal_collapse_finishes": int(paths["terminal_collapse_finish"].sum()),
        "direct_strike_finishes": int(paths["direct_strike_finish"].sum()),
    }

    print("\n" + "=" * 112)
    print("SINGLE HISTORICAL BOUT KO/TKO AUDIT — CURRENT LOCKED SHADOW CONFIGURATION")
    print("=" * 112)
    print(f"bout_id: {bout_id}")
    red_age_text = f"{red_age:.2f}" if red_age is not None else "n/a"
    blue_age_text = f"{blue_age:.2f}" if blue_age is not None else "n/a"
    print(f"historical: {summary['red_name']} (RED, age {red_age_text}) vs {summary['blue_name']} (BLUE, age {blue_age_text})")
    print(
        f"actual result: {historical_method}, round {historical_finish_round}; "
        f"winner side={historical_winner_side}"
    )
    print(f"paths: {args.paths:,}")
    print("R2 recovery: damage 20%, stamina 40%")
    print("R3 recovery: damage 60%, stamina 0%")

    print("\nEXACT PREFIGHT FSR-32 INPUTS")
    print(f"{'trait':36s} {'RED':>12s} {'BLUE':>12s} {'red-blue':>12s}")
    print("-" * 76)
    highlight_rows = _profile_rows(red, blue, HIGHLIGHT_TRAITS)
    for row in highlight_rows:
        print(
            f"{row['trait']:36s} "
            f"{float(row['red']):12.4f} "
            f"{float(row['blue']):12.4f} "
            f"{float(row['red_minus_blue']):12.4f}"
        )

    effective_rows: list[dict[str, object]] = []
    print("\nROUND-ENTRY EFFECTIVE FSR — MEAN ACROSS PATHS THAT REACHED ROUND")
    print("Only striking_power is fatigue-sensitive in the current locked candidate.")
    print(
        f"{'round':>5s} {'side':>6s} {'paths':>8s} "
        f"{'stamina':>10s} {'fatigue_pen':>12s} {'eff_power':>11s}"
    )
    print("-" * 61)
    for rnd in (1, 2, 3):
        for fighter, side in ((0, "RED"), (1, "BLUE")):
            samples = effective_samples[rnd][fighter]
            count = len(samples["stamina_fraction"])
            stamina_mean = float(np.mean(samples["stamina_fraction"])) if count else np.nan
            penalty_mean = float(np.mean(samples["fatigue_penalty"])) if count else np.nan
            power_mean = float(np.mean(samples["effective_striking_power"])) if count else np.nan
            print(
                f"{rnd:5d} {side:>6s} {count:8,d} "
                f"{stamina_mean:10.4f} {penalty_mean:12.4f} {power_mean:11.4f}"
            )
            effective_rows.append(
                {
                    "round": rnd,
                    "side": side.lower(),
                    "paths_reached": count,
                    "stamina_fraction_mean": stamina_mean,
                    "fatigue_penalty_mean": penalty_mean,
                    "effective_striking_power_mean": power_mean,
                }
            )

    print("\nKO/TKO PROBABILITIES")
    print(f"RED  {summary['red_name']}: {p_red:.2%} ({red_ko:,}/{args.paths:,})")
    print(f"BLUE {summary['blue_name']}: {p_blue:.2%} ({blue_ko:,}/{args.paths:,})")
    print(f"ANY KO/TKO:              {p_any:.2%} ({any_ko:,}/{args.paths:,})")
    print(f"NO KO/TKO through R3:    {no_ko / args.paths:.2%} ({no_ko:,}/{args.paths:,})")
    print(f"predicted KO side:       {predicted_side}")

    print("\nROUND-BY-ROUND — CONDITIONAL ON REACHING ROUND")
    print(" round  paths_reached  sig_mean  kd_mean  ko_rate")
    for rnd in (1, 2, 3):
        reached = totals[rnd]["reached"]
        sig = totals[rnd]["sig"] / reached if reached else np.nan
        kd = totals[rnd]["kd"] / reached if reached else np.nan
        ko_rate = totals[rnd]["ko"] / reached if reached else np.nan
        print(f" {rnd:>5}  {reached:>13,}  {sig:>8.3f}  {kd:>7.4f}  {ko_rate:>7.2%}")

    print("\nFINISH MECHANISMS")
    print(f"terminal-collapse KO paths: {summary['terminal_collapse_finishes']:,}")
    print(f"direct-strike KO paths:     {summary['direct_strike_finishes']:,}")

    bout_dir = args.output_dir / bout_id
    bout_dir.mkdir(parents=True, exist_ok=True)
    summary_path = bout_dir / "summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    highlights_path = bout_dir / "fsr32_highlights.csv"
    pd.DataFrame(highlight_rows).to_csv(highlights_path, index=False)

    all_rows = _profile_rows(red, blue, _numeric_fsr_columns(red, blue))
    all_numeric_path = bout_dir / "fsr32_all_numeric.csv"
    pd.DataFrame(all_rows).to_csv(all_numeric_path, index=False)

    effective_path = bout_dir / "round_entry_effective_fsr.csv"
    pd.DataFrame(effective_rows).to_csv(effective_path, index=False)

    print(f"\nsummary:        {summary_path}")
    print(f"FSR highlights: {highlights_path}")
    print(f"all numeric FSR: {all_numeric_path}")
    print(f"effective FSR:  {effective_path}")

    if args.save_paths:
        paths_path = bout_dir / "paths.csv"
        paths.to_csv(paths_path, index=False)
        print(f"paths:          {paths_path}")

    print("Research-only audit; no production or stored FSR artifacts are modified.")


if __name__ == "__main__":
    main()
