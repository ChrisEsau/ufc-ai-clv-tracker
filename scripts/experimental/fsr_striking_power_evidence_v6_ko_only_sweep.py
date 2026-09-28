"""Research-only V6 fresh striking-power sweep using KO/TKO evidence only.

V6 deliberately removes knockdowns from the power signal entirely.  The goal
is to ask a narrower face-validity question: if fresh striking power is inferred
only from Round-1 KO/TKO demonstrations, do the resulting fighter names better
match the latent physical trait we want for the simulator?

Per-fight evidence:

    E_KO = 2.5 * I(R1 KO/TKO win)
               * exp(-finish_seconds / 180)
               * exp(-max(R1_sig_landed - 1, 0) / 20)
               * exp(-R1_ground_landed / 10)

    E_fight = E_KO

There is no KD term and quiet fights never subtract evidence.

Career aggregation keeps the V4 structure:

    events = number of UFC Round-1 KO/TKO evidence fights
    fights = UFC Round-1 fights
    event_rate = events / fights

    repeatability = 1 - exp(-events / 3)
    sample_confidence = 1 - exp(-fights / 5)
    adjusted_frequency = sqrt(event_rate) * sample_confidence
    compressed_peak = 1 - exp(-peak / 2)

    S_v6 = 5 * repeatability
         + 4 * adjusted_frequency
         + 2 * compressed_peak

Research only.  This script does not modify any FSR artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v4_aggregation_sweep as v4

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v6_ko_only_rankings.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def evidence_for_row_v6(row: pd.Series) -> float:
    # Reuse the already-reviewed V2 early-R1 KO/TKO evidence equation exactly,
    # while discarding the KD term entirely.
    _, early_ko_evidence, _ = v2.evidence_for_row(row)
    return float(early_ko_evidence)


def build_v6_rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    scored["ko_power_evidence_v6"] = scored.apply(evidence_for_row_v6, axis=1)
    scored["power_event"] = scored["ko_power_evidence_v6"].gt(0.0)

    agg = scored.groupby(["fighter_id", "fighter_name"], as_index=False).agg(
        ufc_r1_fights=("fight_id", "nunique"),
        r1_kds=("kd", "sum"),
        r1_ko_tko_wins=("r1_ko_tko_win", "sum"),
        power_event_fights=("power_event", "sum"),
        cumulative_ko_power_evidence=("ko_power_evidence_v6", "sum"),
        max_single_fight_ko_evidence=("ko_power_evidence_v6", "max"),
        total_r1_sig_landed=("sig_str_landed", "sum"),
        total_r1_ground_landed=("ground_landed", "sum"),
    )

    fights = agg["ufc_r1_fights"].astype(float).clip(lower=1.0)
    events = agg["power_event_fights"].astype(float)
    peak = agg["max_single_fight_ko_evidence"].astype(float).clip(lower=0.0)

    agg["power_event_rate"] = (events / fights).clip(0.0, 1.0)
    agg["repeatability_term"] = 1.0 - np.exp(-events / v4.REPEATABILITY_TAU)
    agg["sample_confidence"] = 1.0 - np.exp(-fights / v4.CONFIDENCE_TAU)
    agg["adjusted_frequency_term"] = (
        np.sqrt(agg["power_event_rate"]) * agg["sample_confidence"]
    )
    agg["compressed_peak_term"] = 1.0 - np.exp(-peak / v4.PEAK_TAU)

    agg["v6_power_score"] = (
        v4.REPEATABILITY_WEIGHT * agg["repeatability_term"]
        + v4.FREQUENCY_WEIGHT * agg["adjusted_frequency_term"]
        + v4.PEAK_WEIGHT * agg["compressed_peak_term"]
    )

    s = agg["v6_power_score"].astype(float)
    agg["power_evidence_rating_v6"] = (
        50.0 + 40.0 * (1.0 - np.exp(-s / v4.DISPLAY_RATING_SATURATION))
    ).clip(50.0, 90.0)

    # Preserve fighters with no R1 KO/TKO evidence at neutral 50 for visibility.
    agg.loc[agg["power_event_fights"].eq(0), "power_evidence_rating_v6"] = 50.0

    agg = agg.sort_values(
        [
            "v6_power_score",
            "repeatability_term",
            "adjusted_frequency_term",
            "compressed_peak_term",
            "cumulative_ko_power_evidence",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank_v6", np.arange(1, len(agg) + 1))
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
    rankings, scored = build_v6_rankings(fighter_fights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    detail_path = args.output.with_name(args.output.stem + "_fighter_fight_detail.csv")
    scored.to_csv(detail_path, index=False)

    top = rankings.head(max(1, args.top)).copy()
    display_cols = [
        "rank_v6",
        "fighter_name",
        "power_evidence_rating_v6",
        "v6_power_score",
        "r1_ko_tko_wins",
        "power_event_fights",
        "power_event_rate",
        "max_single_fight_ko_evidence",
        "cumulative_ko_power_evidence",
        "ufc_r1_fights",
        "r1_kds",
    ]

    print("\n" + "=" * 155)
    print("FRESH STRIKING POWER EVIDENCE V6 — ROUND-1 KO/TKO ONLY")
    print("=" * 155)
    print(f"fighter-fight R1 rows: {len(fighter_fights):,}")
    print(f"fighters ranked: {len(rankings):,}")
    print("\nPer-fight evidence:")
    print("  KD evidence = 0 (removed entirely)")
    print(
        "  E_KO = 2.5 * I(R1 KO/TKO win) * exp(-t/180) "
        "* exp(-max(R1_SIG_LANDED-1,0)/20) * exp(-R1_GROUND_LANDED/10)"
    )
    print("\nCareer aggregation: V4 structure")
    print("  repeatability = 1 - exp(-R1_KO_event_fights/3)")
    print("  sample_confidence = 1 - exp(-UFC_R1_fights/5)")
    print("  adjusted_frequency = sqrt(R1_KO_event_rate) * sample_confidence")
    print("  compressed_peak = 1 - exp(-peak_KO_evidence/2)")
    print("  S_v6 = 5*repeatability + 4*adjusted_frequency + 2*compressed_peak")
    print("  quiet fights never subtract demonstrated power")

    print("\nTOP FIGHTERS")
    print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote rankings: {args.output}")
    print(f"Wrote fighter-fight evidence detail: {detail_path}")
    print("Research only: this script does not modify any FSR artifact.")


if __name__ == "__main__":
    main()
