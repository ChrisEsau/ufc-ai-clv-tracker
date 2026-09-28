"""Research-only V4 career aggregation sweep for fresh striking power.

Keeps the V2 per-fight evidence equation unchanged and changes only career
aggregation. V4 fixes the V3 failure mode where one extreme peak event could
dominate repeated demonstrations from established heavy hitters.

Per-fight evidence is imported unchanged from V2:

    E_KD = R1_KD

    E_EARLY_KO = 2.5 * I(R1 KO/TKO win)
                    * exp(-finish_seconds / 180)
                    * exp(-max(R1_sig_landed - 1, 0) / 20)
                    * exp(-R1_ground_landed / 10)

    E_fight = E_KD + E_EARLY_KO

V4 career aggregation:

    events = number of UFC Round-1 fights with E_fight > 0
    fights = UFC Round-1 fights
    event_rate = events / fights

    repeatability = 1 - exp(-events / 3)
    sample_confidence = 1 - exp(-fights / 5)
    adjusted_frequency = sqrt(event_rate) * sample_confidence
    compressed_peak = 1 - exp(-peak / 2)

    S_v4 = 5 * repeatability
         + 4 * adjusted_frequency
         + 2 * compressed_peak

Hierarchy:
1. repeatability is primary;
2. event frequency matters only with sample confidence;
3. peak severity is a compressed modifier, not the dominant signal.

Quiet fights never subtract from demonstrated power. They affect only the
frequency component. This script is research-only and does not modify FSR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v4_aggregation_rankings.csv"
)

REPEATABILITY_WEIGHT = 5.0
FREQUENCY_WEIGHT = 4.0
PEAK_WEIGHT = 2.0
REPEATABILITY_TAU = 3.0
CONFIDENCE_TAU = 5.0
PEAK_TAU = 2.0
DISPLAY_RATING_SATURATION = 6.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def build_v4_rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    pieces = scored.apply(v2.evidence_for_row, axis=1, result_type="expand")
    pieces.columns = ["kd_evidence", "early_ko_evidence", "fight_power_evidence"]
    scored = pd.concat([scored, pieces], axis=1)
    scored["power_event"] = scored["fight_power_evidence"].gt(0.0)

    agg = scored.groupby(["fighter_id", "fighter_name"], as_index=False).agg(
        ufc_r1_fights=("fight_id", "nunique"),
        r1_kds=("kd", "sum"),
        r1_ko_tko_wins=("r1_ko_tko_win", "sum"),
        power_event_fights=("power_event", "sum"),
        kd_evidence=("kd_evidence", "sum"),
        early_ko_evidence=("early_ko_evidence", "sum"),
        cumulative_power_evidence=("fight_power_evidence", "sum"),
        max_single_fight_evidence=("fight_power_evidence", "max"),
        total_r1_sig_landed=("sig_str_landed", "sum"),
        total_r1_ground_landed=("ground_landed", "sum"),
    )

    fights = agg["ufc_r1_fights"].astype(float).clip(lower=1.0)
    events = agg["power_event_fights"].astype(float)
    peak = agg["max_single_fight_evidence"].astype(float).clip(lower=0.0)

    agg["power_event_rate"] = (events / fights).clip(0.0, 1.0)
    agg["repeatability_term"] = 1.0 - np.exp(-events / REPEATABILITY_TAU)
    agg["sample_confidence"] = 1.0 - np.exp(-fights / CONFIDENCE_TAU)
    agg["adjusted_frequency_term"] = (
        np.sqrt(agg["power_event_rate"]) * agg["sample_confidence"]
    )
    agg["compressed_peak_term"] = 1.0 - np.exp(-peak / PEAK_TAU)

    agg["v4_power_score"] = (
        REPEATABILITY_WEIGHT * agg["repeatability_term"]
        + FREQUENCY_WEIGHT * agg["adjusted_frequency_term"]
        + PEAK_WEIGHT * agg["compressed_peak_term"]
    )

    s = agg["v4_power_score"].astype(float)
    agg["power_evidence_rating_v4"] = (
        50.0 + 40.0 * (1.0 - np.exp(-s / DISPLAY_RATING_SATURATION))
    ).clip(50.0, 90.0)

    # Reconstruct V2 ranking for direct movement comparison.
    cumulative = agg["cumulative_power_evidence"].astype(float)
    agg["power_evidence_rating_v2"] = (
        50.0 + 40.0 * (1.0 - np.exp(-cumulative / v2.EVIDENCE_SATURATION))
    ).clip(50.0, 90.0)
    v2_order = agg.sort_values(
        ["power_evidence_rating_v2", "cumulative_power_evidence", "power_event_fights"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    v2_rank = {
        fighter_id: rank
        for rank, fighter_id in enumerate(v2_order["fighter_id"], start=1)
    }

    agg = agg.sort_values(
        [
            "v4_power_score",
            "repeatability_term",
            "adjusted_frequency_term",
            "compressed_peak_term",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank_v4", np.arange(1, len(agg) + 1))
    agg["rank_v2"] = agg["fighter_id"].map(v2_rank).astype(int)
    agg["rank_change_vs_v2"] = agg["rank_v2"] - agg["rank_v4"]
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
    rankings, scored = build_v4_rankings(fighter_fights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    detail_path = args.output.with_name(args.output.stem + "_fighter_fight_detail.csv")
    scored.to_csv(detail_path, index=False)

    top = rankings.head(max(1, args.top)).copy()
    display_cols = [
        "rank_v4",
        "rank_v2",
        "rank_change_vs_v2",
        "fighter_name",
        "power_evidence_rating_v4",
        "v4_power_score",
        "repeatability_term",
        "adjusted_frequency_term",
        "compressed_peak_term",
        "max_single_fight_evidence",
        "power_event_fights",
        "power_event_rate",
        "r1_kds",
        "r1_ko_tko_wins",
        "ufc_r1_fights",
    ]

    print("\n" + "=" * 170)
    print("FRESH STRIKING POWER EVIDENCE V4 — REPEATABILITY + CONFIDENCE-ADJUSTED FREQUENCY + COMPRESSED PEAK")
    print("=" * 170)
    print(f"fighter-fight R1 rows: {len(fighter_fights):,}")
    print(f"fighters ranked: {len(rankings):,}")
    print("\nPer-fight evidence: unchanged from V2")
    print("  E_KD = R1_KD")
    print(
        "  E_EARLY_KO = 2.5 * I(R1 KO/TKO win) * exp(-t/180) "
        "* exp(-max(R1_SIG_LANDED-1,0)/20) * exp(-R1_GROUND_LANDED/10)"
    )
    print("\nV4 career aggregation:")
    print("  repeatability = 1 - exp(-power_event_fights/3)")
    print("  sample_confidence = 1 - exp(-UFC_R1_fights/5)")
    print("  adjusted_frequency = sqrt(power_event_rate) * sample_confidence")
    print("  compressed_peak = 1 - exp(-peak/2)")
    print("  S_v4 = 5*repeatability + 4*adjusted_frequency + 2*compressed_peak")
    print("  provisional rating = 50 + 40*(1-exp(-S_v4/6))")
    print("  hierarchy: repeatability > confidence-adjusted frequency > peak modifier")

    print("\nTOP FIGHTERS")
    print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nrank_change_vs_v2: positive = fighter moved UP in V4; negative = moved DOWN")
    print(f"\nWrote rankings: {args.output}")
    print(f"Wrote fighter-fight evidence detail: {detail_path}")
    print("Research only: this script does not modify any FSR artifact.")


if __name__ == "__main__":
    main()
