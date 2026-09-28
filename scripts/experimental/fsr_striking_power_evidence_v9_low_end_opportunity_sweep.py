"""Research-only V9 low-end sweep for fresh striking power.

V8 is frozen for every fighter with positive power evidence. V9 changes only
fighters whose V8 rating is exactly 50 (no observed KO/TKO or R1 KD power event).

Concept
-------
A fighter with no positive power event should not be called low-power merely
because evidence is absent. But repeated *landed* fresh significant strikes
without ever producing a KO/TKO or R1 KD are legitimate negative evidence about
acute damage ceiling.

We therefore accumulate an opportunity measure only for V8-neutral fighters:

    opportunity_i = 1 - exp(-R1_sig_landed_i / 20)
    O = sum(opportunity_i)

This saturates each individual fight so one high-volume round cannot dominate.
Then candidate low-end ratings are:

    rating = 50 - L * (1 - exp(-O / tau))

where L is the maximum downward penalty and tau controls how quickly repeated
opportunity becomes informative.

Positive V8 fighters are NEVER penalized. This preserves the non-degrading
'demonstrated power' rule.

The script compares multiple candidate L/tau settings and prints distribution
statistics plus representative neutral fighters. It does not modify any FSR
artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v8_hierarchical_ko_kd_sweep as v8

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v9_low_end_opportunity_sweep.csv"
)

OPPORTUNITY_SIG_TAU = 20.0
CANDIDATES = [
    (10.0, 4.0),
    (15.0, 4.0),
    (15.0, 6.0),
    (20.0, 6.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--show", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    master = pd.read_parquet(args.master)
    rounds = pd.read_parquet(args.round_stats)

    rankings, scored = v8.build_v8_rankings(master, rounds)

    # V8 currently groups by fighter_id + fighter_name, so historical name
    # variants can create multiple rows for one fighter_id. V9 is a fighter-
    # level audit, therefore normalize both sides of the merge to fighter_id.
    rankings = (
        rankings.sort_values(
            ["fighter_id", "power_evidence_rating_v8", "ufc_r1_fights"],
            ascending=[True, False, False],
        )
        .drop_duplicates("fighter_id", keep="first")
        .reset_index(drop=True)
    )

    # V8 scored is one row per fighter-fight/name variant, sourced from R1.
    # Use landed R1 significant strikes as actual clean-strike opportunity to
    # demonstrate fresh power. Aggregate ONLY by fighter_id so name aliases do
    # not create duplicate merge keys or split opportunity evidence.
    detail = scored[["fighter_id", "fighter_name", "fight_id", "sig_str_landed"]].copy()
    detail["sig_str_landed"] = (
        pd.to_numeric(detail["sig_str_landed"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    detail["opportunity"] = 1.0 - np.exp(
        -detail["sig_str_landed"] / OPPORTUNITY_SIG_TAU
    )

    opp = detail.groupby("fighter_id", as_index=False).agg(
        ufc_r1_fights_opp=("fight_id", "nunique"),
        total_r1_sig_landed=("sig_str_landed", "sum"),
        opportunity_score=("opportunity", "sum"),
        mean_r1_sig_landed=("sig_str_landed", "mean"),
    )

    if rankings["fighter_id"].duplicated().any():
        raise RuntimeError("V9 rankings normalization failed: duplicate fighter_id")
    if opp["fighter_id"].duplicated().any():
        raise RuntimeError("V9 opportunity normalization failed: duplicate fighter_id")

    base = rankings.merge(
        opp[[
            "fighter_id", "total_r1_sig_landed", "opportunity_score",
            "mean_r1_sig_landed",
        ]],
        on="fighter_id",
        how="left",
        validate="one_to_one",
    )
    for c in ("total_r1_sig_landed", "opportunity_score", "mean_r1_sig_landed"):
        base[c] = base[c].fillna(0.0)

    neutral_mask = np.isclose(base["power_evidence_rating_v8"], 50.0)
    neutral = base.loc[neutral_mask].copy()

    print("\n" + "=" * 145)
    print("V9 FRESH STRIKING POWER — LOW-END OPPORTUNITY SWEEP")
    print("=" * 145)
    print(f"fighters after fighter_id normalization: {len(base):,}")
    print(f"V8 neutral fighters: {len(neutral):,}")
    print("\nNegative evidence is applied ONLY to V8-neutral fighters.")
    print("opportunity_i = 1 - exp(-R1_SIG_LANDED/20)")
    print("O = sum(opportunity_i)")
    print("candidate rating = 50 - L*(1-exp(-O/tau))")

    print("\nNEUTRAL-FIGHTER OPPORTUNITY DISTRIBUTION")
    for col in ("ufc_r1_fights", "total_r1_sig_landed", "opportunity_score"):
        vals = neutral[col].astype(float)
        print(f"\n{col}:")
        for p in (0, 10, 25, 50, 75, 90, 95, 99, 100):
            print(f"  p{p:>3}: {np.percentile(vals, p):8.3f}")

    out = base.copy()
    for L, tau in CANDIDATES:
        name = f"v9_L{int(L)}_tau{int(tau)}"
        candidate = 50.0 - L * (1.0 - np.exp(-out["opportunity_score"] / tau))
        out[name] = out["power_evidence_rating_v8"]
        out.loc[neutral_mask, name] = candidate.loc[neutral_mask]

        vals = out[name].astype(float)
        neutral_vals = out.loc[neutral_mask, name].astype(float)
        print("\n" + "-" * 145)
        print(f"CANDIDATE {name}: max penalty L={L:.0f}, opportunity tau={tau:.0f}")
        print(f"  all fighters mean/median: {vals.mean():.3f} / {vals.median():.3f}")
        print(f"  neutral-only mean/median: {neutral_vals.mean():.3f} / {neutral_vals.median():.3f}")
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"  all p{p:02d}: {np.percentile(vals, p):.3f}")

        bottom = out.loc[neutral_mask].nsmallest(max(1, args.show), name)
        print("\n  LOWEST-RATED V8-NEUTRAL FIGHTERS")
        print(bottom[[
            "fighter_name", name, "ufc_r1_fights", "total_r1_sig_landed",
            "mean_r1_sig_landed", "opportunity_score",
        ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        uncertain = out.loc[neutral_mask].copy()
        uncertain["distance_to_49"] = (uncertain[name] - 49.0).abs()
        uncertain = uncertain.nsmallest(max(1, min(args.show, 12)), "distance_to_49")
        print("\n  REPRESENTATIVE STILL-UNCERTAIN FIGHTERS (~49)")
        print(uncertain[[
            "fighter_name", name, "ufc_r1_fights", "total_r1_sig_landed",
            "opportunity_score",
        ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote sweep: {args.output}")
    print("Research only: V8 and all FSR artifacts remain untouched.")


if __name__ == "__main__":
    main()
