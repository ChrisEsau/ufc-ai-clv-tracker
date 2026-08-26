"""Research-only distribution audit for V7 fresh striking power.

Builds the full V7 KO/TKO-only ranking, then prints:
- distribution summary and selected percentiles for power_evidence_rating_v7;
- counts in 5-point rating bins;
- representative fighters around the median;
- representative bottom fighters.

This does not modify any FSR artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v7_ko_only_softened_penalties_sweep as v7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--median-n", type=int, default=15)
    p.add_argument("--bottom-n", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    master = pd.read_parquet(args.master)
    rounds = pd.read_parquet(args.round_stats)
    frame = v2.fighter_fight_frame(master, rounds)
    rankings, _ = v7.build_v7_rankings(frame)

    rating = rankings["power_evidence_rating_v7"].astype(float)
    evidence_positive = rankings["power_event_fights"].gt(0)
    positive = rankings.loc[evidence_positive].copy()

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(rating, percentiles)

    print("\n" + "=" * 110)
    print("V7 FRESH STRIKING POWER — FULL DISTRIBUTION AUDIT")
    print("=" * 110)
    print(f"fighters ranked: {len(rankings):,}")
    print(f"fighters with >=1 R1 KO/TKO power event: {len(positive):,} ({len(positive)/len(rankings):.2%})")
    print(f"fighters with zero R1 KO/TKO evidence / neutral 50: {(~evidence_positive).sum():,} ({(~evidence_positive).mean():.2%})")
    print(f"mean rating:   {rating.mean():.3f}")
    print(f"median rating: {rating.median():.3f}")
    print(f"std dev:       {rating.std(ddof=0):.3f}")
    print(f"min / max:     {rating.min():.3f} / {rating.max():.3f}")

    print("\nPERCENTILES — ALL FIGHTERS")
    for p, value in zip(percentiles, q):
        print(f"  p{p:>3}: {value:7.3f}")

    if not positive.empty:
        pr = positive["power_evidence_rating_v7"].astype(float)
        pq = np.percentile(pr, percentiles)
        print("\nPERCENTILES — FIGHTERS WITH >=1 R1 KO/TKO POWER EVENT")
        for p, value in zip(percentiles, pq):
            print(f"  p{p:>3}: {value:7.3f}")

    bins = [50, 55, 60, 65, 70, 75, 80, 85, 90.0001]
    labels = ["50-<55", "55-<60", "60-<65", "65-<70", "70-<75", "75-<80", "80-<85", "85-90"]
    cats = pd.cut(rating, bins=bins, labels=labels, right=False, include_lowest=True)
    counts = cats.value_counts(sort=False)
    print("\nRATING DISTRIBUTION BINS")
    for label in labels:
        n = int(counts.get(label, 0))
        print(f"  {label:>7}: {n:4d}  ({n/len(rankings):6.2%})")

    display = [
        "rank_v7", "fighter_name", "power_evidence_rating_v7", "v7_power_score",
        "r1_ko_tko_wins", "power_event_rate", "max_single_fight_ko_evidence_v7", "ufc_r1_fights"
    ]

    # Median sample is centered on the full-population median rating, with stable tie-breaking by rank.
    median_rating = float(rating.median())
    med = rankings.assign(_dist=(rating - median_rating).abs()).sort_values(
        ["_dist", "rank_v7"], ascending=[True, True]
    ).head(max(1, args.median_n))
    print("\nREPRESENTATIVE FIGHTERS AROUND FULL-POPULATION MEDIAN")
    print(med[display].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    bottom = rankings.tail(max(1, args.bottom_n)).sort_values("rank_v7", ascending=True)
    print("\nBOTTOM FIGHTERS")
    print(bottom[display].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not positive.empty:
        pos_median = float(positive["power_evidence_rating_v7"].median())
        pos_med = positive.assign(
            _dist=(positive["power_evidence_rating_v7"].astype(float) - pos_median).abs()
        ).sort_values(["_dist", "rank_v7"]).head(max(1, args.median_n))
        print("\nREPRESENTATIVE FIGHTERS AROUND MEDIAN AMONG POSITIVE-EVIDENCE FIGHTERS")
        print(pos_med[display].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
