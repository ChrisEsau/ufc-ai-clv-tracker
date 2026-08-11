"""Research-only V5 fresh striking-power sweep.

V5 freezes the V4 career aggregation and changes only the per-fight Round-1
knockdown evidence so that knockdowns produced with fewer landed significant
strikes count as stronger demonstrations of fresh acute power.

Early KO/TKO evidence is unchanged from V2.

Per-fight evidence:

    kd_efficiency_bonus = 1 + alpha * exp(-R1_sig_landed / tau)
    E_KD_V5 = R1_KD * kd_efficiency_bonus

    E_EARLY_KO = 2.5 * I(R1 KO/TKO win)
                    * exp(-finish_seconds / 180)
                    * exp(-max(R1_sig_landed - 1, 0) / 20)
                    * exp(-R1_ground_landed / 10)

    E_fight_V5 = E_KD_V5 + E_EARLY_KO

Career aggregation is unchanged from V4:

    events = number of UFC Round-1 fights with E_fight_V5 > 0
    fights = UFC Round-1 fights
    event_rate = events / fights

    repeatability = 1 - exp(-events / 3)
    sample_confidence = 1 - exp(-fights / 5)
    adjusted_frequency = sqrt(event_rate) * sample_confidence
    compressed_peak = 1 - exp(-peak / 2)

    S_v5 = 5 * repeatability
         + 4 * adjusted_frequency
         + 2 * compressed_peak

The alpha/tau KD-efficiency constants are intentionally exposed near the top
for face-validity research. This script does not modify any FSR artifact.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v4_aggregation_sweep as v4

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v5_kd_efficiency_rankings.csv"
)

# V5-only per-fight KD evidence parameters.
KD_EFFICIENCY_ALPHA = 1.0
KD_EFFICIENCY_TAU = 20.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def evidence_for_row_v5(row: pd.Series) -> tuple[float, float, float, float]:
    kd = max(0.0, float(row["kd"]))
    sig = max(0.0, float(row["sig_str_landed"]))

    kd_efficiency_bonus = 1.0 + KD_EFFICIENCY_ALPHA * exp(-sig / KD_EFFICIENCY_TAU)
    kd_evidence = kd * kd_efficiency_bonus

    # Keep V2 early-KO evidence exactly unchanged.
    _, early_ko_evidence, _ = v2.evidence_for_row(row)
    total = kd_evidence + early_ko_evidence
    return kd_evidence, early_ko_evidence, total, kd_efficiency_bonus


def build_v5_rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    pieces = scored.apply(evidence_for_row_v5, axis=1, result_type="expand")
    pieces.columns = [
        "kd_evidence_v5",
        "early_ko_evidence",
        "fight_power_evidence_v5",
        "kd_efficiency_bonus",
    ]
    scored = pd.concat([scored, pieces], axis=1)
    scored["power_event"] = scored["fight_power_evidence_v5"].gt(0.0)

    agg = scored.groupby(["fighter_id", "fighter_name"], as_index=False).agg(
        ufc_r1_fights=("fight_id", "nunique"),
        r1_kds=("kd", "sum"),
        r1_ko_tko_wins=("r1_ko_tko_win", "sum"),
        power_event_fights=("power_event", "sum"),
        kd_evidence_v5=("kd_evidence_v5", "sum"),
        early_ko_evidence=("early_ko_evidence", "sum"),
        cumulative_power_evidence_v5=("fight_power_evidence_v5", "sum"),
        max_single_fight_evidence_v5=("fight_power_evidence_v5", "max"),
        mean_kd_efficiency_bonus=("kd_efficiency_bonus", "mean"),
        total_r1_sig_landed=("sig_str_landed", "sum"),
        total_r1_ground_landed=("ground_landed", "sum"),
    )

    fights = agg["ufc_r1_fights"].astype(float).clip(lower=1.0)
    events = agg["power_event_fights"].astype(float)
    peak = agg["max_single_fight_evidence_v5"].astype(float).clip(lower=0.0)

    agg["power_event_rate"] = (events / fights).clip(0.0, 1.0)
    agg["repeatability_term"] = 1.0 - np.exp(-events / v4.REPEATABILITY_TAU)
    agg["sample_confidence"] = 1.0 - np.exp(-fights / v4.CONFIDENCE_TAU)
    agg["adjusted_frequency_term"] = (
        np.sqrt(agg["power_event_rate"]) * agg["sample_confidence"]
    )
    agg["compressed_peak_term"] = 1.0 - np.exp(-peak / v4.PEAK_TAU)

    agg["v5_power_score"] = (
        v4.REPEATABILITY_WEIGHT * agg["repeatability_term"]
        + v4.FREQUENCY_WEIGHT * agg["adjusted_frequency_term"]
        + v4.PEAK_WEIGHT * agg["compressed_peak_term"]
    )

    s = agg["v5_power_score"].astype(float)
    agg["power_evidence_rating_v5"] = (
        50.0 + 40.0 * (1.0 - np.exp(-s / v4.DISPLAY_RATING_SATURATION))
    ).clip(50.0, 90.0)

    # Build the V4 ranking using V2 per-fight evidence for direct comparison.
    v4_rankings, _ = v4.build_v4_rankings(frame)
    v4_rank = dict(zip(v4_rankings["fighter_id"], v4_rankings["rank_v4"]))

    agg = agg.sort_values(
        [
            "v5_power_score",
            "repeatability_term",
            "adjusted_frequency_term",
            "compressed_peak_term",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank_v5", np.arange(1, len(agg) + 1))
    agg["rank_v4"] = agg["fighter_id"].map(v4_rank).astype(int)
    agg["rank_change_vs_v4"] = agg["rank_v4"] - agg["rank_v5"]
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
    rankings, scored = build_v5_rankings(fighter_fights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    detail_path = args.output.with_name(args.output.stem + "_fighter_fight_detail.csv")
    scored.to_csv(detail_path, index=False)

    top = rankings.head(max(1, args.top)).copy()
    display_cols = [
        "rank_v5",
        "rank_v4",
        "rank_change_vs_v4",
        "fighter_name",
        "power_evidence_rating_v5",
        "v5_power_score",
        "max_single_fight_evidence_v5",
        "power_event_fights",
        "power_event_rate",
        "r1_kds",
        "r1_ko_tko_wins",
        "ufc_r1_fights",
        "kd_evidence_v5",
    ]

    print("\n" + "=" * 170)
    print("FRESH STRIKING POWER EVIDENCE V5 — OPPORTUNITY-ADJUSTED R1 KD EVIDENCE")
    print("=" * 170)
    print(f"fighter-fight R1 rows: {len(fighter_fights):,}")
    print(f"fighters ranked: {len(rankings):,}")
    print("\nV5 per-fight KD evidence:")
    print(
        "  E_KD_V5 = R1_KD * (1 + alpha*exp(-R1_SIG_LANDED/tau)); "
        f"alpha={KD_EFFICIENCY_ALPHA:.1f}, tau={KD_EFFICIENCY_TAU:.1f}"
    )
    print("  early-KO evidence: unchanged from V2")
    print("\nCareer aggregation: frozen from V4")
    print("  repeatability = 1 - exp(-power_event_fights/3)")
    print("  sample_confidence = 1 - exp(-UFC_R1_fights/5)")
    print("  adjusted_frequency = sqrt(power_event_rate) * sample_confidence")
    print("  compressed_peak = 1 - exp(-peak/2)")
    print("  S_v5 = 5*repeatability + 4*adjusted_frequency + 2*compressed_peak")

    print("\nTOP FIGHTERS")
    print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nrank_change_vs_v4: positive = fighter moved UP in V5; negative = moved DOWN")
    print(f"\nWrote rankings: {args.output}")
    print(f"Wrote fighter-fight evidence detail: {detail_path}")
    print("Research only: this script does not modify any FSR artifact.")


if __name__ == "__main__":
    main()
