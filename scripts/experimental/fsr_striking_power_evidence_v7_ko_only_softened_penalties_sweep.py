"""Research-only V7 fresh striking-power sweep using KO/TKO evidence only.

V7 keeps the V6 KO-only architecture and V4 career aggregation frozen. It
changes only the three per-fight discount rates to make Round-1 KO/TKO evidence
less aggressively penalized for later finish time, significant-strike volume,
and ground-strike accumulation.

V6 per-fight evidence:

    E_KO_V6 = 2.5 * I(R1 KO/TKO win)
                  * exp(-finish_seconds / 180)
                  * exp(-max(R1_sig_landed - 1, 0) / 20)
                  * exp(-R1_ground_landed / 10)

V7 per-fight evidence:

    E_KO_V7 = 2.5 * I(R1 KO/TKO win)
                  * exp(-finish_seconds / 240)
                  * exp(-max(R1_sig_landed - 1, 0) / 35)
                  * exp(-R1_ground_landed / 20)

There is no KD contribution. Quiet fights never subtract evidence.
Career aggregation is unchanged from V6/V4.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v4_aggregation_sweep as v4
from scripts.experimental import fsr_striking_power_evidence_v6_ko_only_sweep as v6

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v7_ko_only_softened_penalties_rankings.csv"
)

EARLY_KO_WEIGHT = 2.5
TIME_TAU_SECONDS = 240.0
SIG_LANDED_TAU = 35.0
GROUND_LANDED_TAU = 20.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def evidence_for_row_v7(row: pd.Series) -> float:
    if not bool(row["r1_ko_tko_win"]):
        return 0.0

    finish_sec = max(0.0, float(row["match_time_sec"]))
    sig = max(0.0, float(row["sig_str_landed"]))
    ground = max(0.0, float(row["ground_landed"]))

    freshness = exp(-finish_sec / TIME_TAU_SECONDS)
    low_volume = exp(-max(sig - 1.0, 0.0) / SIG_LANDED_TAU)
    low_ground = exp(-ground / GROUND_LANDED_TAU)
    return EARLY_KO_WEIGHT * freshness * low_volume * low_ground


def build_v7_rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    scored["ko_power_evidence_v7"] = scored.apply(evidence_for_row_v7, axis=1)
    scored["power_event"] = scored["ko_power_evidence_v7"].gt(0.0)

    agg = scored.groupby(["fighter_id", "fighter_name"], as_index=False).agg(
        ufc_r1_fights=("fight_id", "nunique"),
        r1_kds=("kd", "sum"),
        r1_ko_tko_wins=("r1_ko_tko_win", "sum"),
        power_event_fights=("power_event", "sum"),
        cumulative_ko_power_evidence_v7=("ko_power_evidence_v7", "sum"),
        max_single_fight_ko_evidence_v7=("ko_power_evidence_v7", "max"),
        total_r1_sig_landed=("sig_str_landed", "sum"),
        total_r1_ground_landed=("ground_landed", "sum"),
    )

    fights = agg["ufc_r1_fights"].astype(float).clip(lower=1.0)
    events = agg["power_event_fights"].astype(float)
    peak = agg["max_single_fight_ko_evidence_v7"].astype(float).clip(lower=0.0)

    agg["power_event_rate"] = (events / fights).clip(0.0, 1.0)
    agg["repeatability_term"] = 1.0 - np.exp(-events / v4.REPEATABILITY_TAU)
    agg["sample_confidence"] = 1.0 - np.exp(-fights / v4.CONFIDENCE_TAU)
    agg["adjusted_frequency_term"] = (
        np.sqrt(agg["power_event_rate"]) * agg["sample_confidence"]
    )
    agg["compressed_peak_term"] = 1.0 - np.exp(-peak / v4.PEAK_TAU)

    agg["v7_power_score"] = (
        v4.REPEATABILITY_WEIGHT * agg["repeatability_term"]
        + v4.FREQUENCY_WEIGHT * agg["adjusted_frequency_term"]
        + v4.PEAK_WEIGHT * agg["compressed_peak_term"]
    )

    s = agg["v7_power_score"].astype(float)
    agg["power_evidence_rating_v7"] = (
        50.0 + 40.0 * (1.0 - np.exp(-s / v4.DISPLAY_RATING_SATURATION))
    ).clip(50.0, 90.0)
    agg.loc[agg["power_event_fights"].eq(0), "power_evidence_rating_v7"] = 50.0

    # Direct V6 ranking comparison.
    v6_rankings, _ = v6.build_v6_rankings(frame)
    v6_rank = dict(zip(v6_rankings["fighter_id"], v6_rankings["rank_v6"]))

    agg = agg.sort_values(
        [
            "v7_power_score",
            "repeatability_term",
            "adjusted_frequency_term",
            "compressed_peak_term",
            "cumulative_ko_power_evidence_v7",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank_v7", np.arange(1, len(agg) + 1))
    agg["rank_v6"] = agg["fighter_id"].map(v6_rank).astype(int)
    agg["rank_change_vs_v6"] = agg["rank_v6"] - agg["rank_v7"]
    return agg, scored


def main() -> None:
    args = parse_args()
    if not args.master.exists():
        raise FileNotFoundError(args.master)
    if not args.round_stats.exists():
        raise FileNotFoundError(args.round_stats)

    master = pd.read_parquet(args.master)
    rounds = pd.read_parquet(args.round_stats)
    fighter_fights = v2.fighter_fight_frame(master, rounds)
    rankings, scored = build_v7_rankings(fighter_fights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    detail_path = args.output.with_name(args.output.stem + "_fighter_fight_detail.csv")
    scored.to_csv(detail_path, index=False)

    top = rankings.head(max(1, args.top)).copy()
    display_cols = [
        "rank_v7",
        "rank_v6",
        "rank_change_vs_v6",
        "fighter_name",
        "power_evidence_rating_v7",
        "v7_power_score",
        "r1_ko_tko_wins",
        "power_event_fights",
        "power_event_rate",
        "max_single_fight_ko_evidence_v7",
        "cumulative_ko_power_evidence_v7",
        "ufc_r1_fights",
        "r1_kds",
    ]

    print("\n" + "=" * 170)
    print("FRESH STRIKING POWER EVIDENCE V7 — KO/TKO ONLY, SOFTENED FINISH PENALTIES")
    print("=" * 170)
    print(f"fighter-fight R1 rows: {len(fighter_fights):,}")
    print(f"fighters ranked: {len(rankings):,}")
    print("\nPer-fight evidence:")
    print("  KD evidence = 0 (removed entirely)")
    print(
        "  E_KO_V7 = 2.5 * I(R1 KO/TKO win) * exp(-t/240) "
        "* exp(-max(R1_SIG_LANDED-1,0)/35) * exp(-R1_GROUND_LANDED/20)"
    )
    print("  V6 comparison taus: time=180, sig=20, ground=10")
    print("\nCareer aggregation: frozen from V6/V4")
    print("  S = 5*repeatability + 4*adjusted_frequency + 2*compressed_peak")

    print("\nTOP FIGHTERS")
    print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nrank_change_vs_v6: positive = fighter moved UP in V7; negative = moved DOWN")
    print(f"\nWrote rankings: {args.output}")
    print(f"Wrote fighter-fight evidence detail: {detail_path}")
    print("Research only: this script does not modify any FSR artifact.")


if __name__ == "__main__":
    main()
