"""Audit how historical UFCStats CTRL aligns with observable ground evidence.

Research-only diagnostic. UFCStats CTRL is not phase-split, so this script does
NOT claim to recover true clinch-vs-ground control. Instead it partitions
fighter-round control seconds by increasingly strong observable evidence that the
fighter had ground activity in that same round.

Ground-evidence proxies use only UFCStats round fields available in our pipeline:
- landed takedown
- ground strike attempt
- submission attempt
- reversal

This provides a sanity check for the simulator's phase allocation without
pretending CTRL itself is timestamped by phase.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH
from pipeline.round_stats import build_round_fighter_wrestling as wrestle
from scripts.experimental import fsr_mature_2020plus_full_cohort_ko_validation_r3_d60_s0 as full

ROUNDS = (1, 2, 3)


def _pct(n: float, d: float) -> float:
    return n / d if d > 0 else np.nan


def _describe(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {k: np.nan for k in ("mean", "median", "p75", "p90")}
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p75": float(x.quantile(0.75)),
        "p90": float(x.quantile(0.90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit historical mature-cohort CTRL by observable ground evidence")
    p.add_argument("--max-bouts", type=int, default=None, help="optional quick-run limit")
    args = p.parse_args()

    cohort, _ = full._build_full_cohort()
    if args.max_bouts is not None:
        cohort = cohort.head(args.max_bouts).reset_index(drop=True)
    wanted = set(cohort["bout_id"].astype(str))

    raw = pd.read_parquet(ROUND_STATS_PATH)
    df = wrestle.standardize_round_stats(raw)
    df["fight_id"] = df["fight_id"].astype(str)
    df = df[df["fight_id"].isin(wanted)].copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df[df["round"].isin(ROUNDS)].copy()
    df["round"] = df["round"].astype(int)

    numeric = ["control_seconds", "td_landed", "ground_attempted", "ground_landed", "sub_att", "rev"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Increasingly broad evidence that this fighter occupied an offensive or
    # controlling ground context during the round.
    df["ev_td"] = df["td_landed"] > 0
    df["ev_ground_offense"] = (df["ground_attempted"] > 0) | (df["ground_landed"] > 0)
    df["ev_ground_attack"] = df["ev_td"] | df["ev_ground_offense"] | (df["sub_att"] > 0)
    df["ev_any_ground"] = df["ev_ground_attack"] | (df["rev"] > 0)

    total_ctrl = float(df["control_seconds"].sum())
    positive = df[df["control_seconds"] > 0].copy()

    print("\n" + "=" * 118)
    print("MATURE 2020+ HISTORICAL UFCSTATS CTRL — GROUND-EVIDENCE PROXY AUDIT")
    print("=" * 118)
    print(f"cohort bouts: {len(cohort):,}")
    print(f"fighter-round rows: {len(df):,}")
    print(f"fighter-rounds with CTRL > 0: {len(positive):,}")
    print(f"total historical fighter CTRL seconds: {total_ctrl:,.0f}s")
    print("IMPORTANT: these are proxy buckets, NOT a true historical clinch/ground phase split.")

    print("\nCONTROL SECONDS ASSOCIATED WITH SAME-FIGHTER GROUND EVIDENCE")
    print(" proxy                         rows+evidence   ctrl_seconds   share_all_CTRL   mean_CTRL/evid_row")
    proxies = [
        ("landed TD", "ev_td"),
        ("ground strike activity", "ev_ground_offense"),
        ("TD/ground strike/sub", "ev_ground_attack"),
        ("TD/ground/sub/reversal", "ev_any_ground"),
    ]
    for label, col in proxies:
        g = df[df[col]]
        ctrl = float(g["control_seconds"].sum())
        mean = float(g["control_seconds"].mean()) if len(g) else np.nan
        print(f" {label:28s} {len(g):13,d} {ctrl:14,.0f} {_pct(ctrl,total_ctrl):16.2%} {mean:19.2f}")

    no_ground = df[~df["ev_any_ground"]].copy()
    no_ground_ctrl = float(no_ground["control_seconds"].sum())
    print(f" {'NO observable ground evidence':28s} {len(no_ground):13,d} {no_ground_ctrl:14,.0f} {_pct(no_ground_ctrl,total_ctrl):16.2%} {float(no_ground['control_seconds'].mean()) if len(no_ground) else np.nan:19.2f}")

    print("\nPOSITIVE-CTRL FIGHTER-ROUND DISTRIBUTIONS")
    print(" bucket                         n      mean    median      p75      p90")
    buckets = [
        ("all CTRL-positive", positive),
        ("any ground evidence", positive[positive["ev_any_ground"]]),
        ("no ground evidence", positive[~positive["ev_any_ground"]]),
    ]
    for label, g in buckets:
        d = _describe(g["control_seconds"])
        print(f" {label:28s} {len(g):6,d} {d['mean']:9.2f} {d['median']:9.2f} {d['p75']:8.2f} {d['p90']:8.2f}")

    print("\nBY ROUND — SHARE OF TOTAL HISTORICAL CTRL ON ROWS WITH ANY GROUND EVIDENCE")
    print(" rnd   fighter_rows    total_CTRL    ground_evid_CTRL    share")
    for r in ROUNDS:
        g = df[df["round"] == r]
        total = float(g["control_seconds"].sum())
        evid = float(g.loc[g["ev_any_ground"], "control_seconds"].sum())
        print(f" R{r} {len(g):13,d} {total:13,.0f} {evid:19,.0f} {_pct(evid,total):9.2%}")

    print("\nINTERPRETATION")
    print("Rows with ground evidence can still include clinch/cage CTRL, and rows without recorded ground evidence can still contain ground control with no TD/ground strike/sub/reversal stat. Therefore the evidence share is not a literal ground-control percentage. Use it as a plausibility band for the simulator's phase allocation, not as a direct calibration target.")
    print("Research-only audit; no FSR or simulator physics modified.")


if __name__ == "__main__":
    main()
