"""Research sweep for a non-degrading fresh striking-power evidence model.

This script DOES NOT modify the FSR database.  It ranks fighters from historical
UFC evidence using only Round-1 knockdowns plus early Round-1 KO/TKO evidence.
The intent is to inspect face validity before replacing the current FSR
``striking_power`` construction.

Concept
-------
Stored striking power is treated as a latent fresh physical ceiling.  Quiet
fights do not subtract evidence.  Demonstrations accumulate, with diminishing
returns only when cumulative evidence is mapped onto a provisional 50-90
rating scale.

Per fighter-fight evidence:

    E_KD = R1_KD

    E_EARLY_KO = 2.5 * I(R1 KO/TKO win)
                    * exp(-finish_seconds / 180)
                    * exp(-max(R1_sig_landed - 1, 0) / 20)
                    * exp(-R1_ground_landed / 10)

    E_fight = E_KD + E_EARLY_KO

Cumulative evidence:

    S = sum(E_fight)

Provisional face-validity rating (NOT yet an approved FSR equation):

    power_evidence_rating = 50 + 40 * (1 - exp(-S / 4))

The early-finish term rewards freshness and low-volume acute stoppages while
penalizing high-volume and ground-accumulation finishes.  R1 KDs remain direct
positive demonstrations.  No fight can reduce S.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_PATH = Path("data/master/ufc_master.parquet")
ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v2_rankings.csv"
)

EARLY_KO_WEIGHT = 2.5
TIME_TAU_SECONDS = 180.0
SIG_LANDED_TAU = 20.0
GROUND_LANDED_TAU = 10.0
EVIDENCE_SATURATION = 4.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def is_ko_tko(method: object) -> bool:
    text = str(method).strip().upper()
    return "KO" in text or "TKO" in text


def fighter_fight_frame(master: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    r1 = rounds.loc[pd.to_numeric(rounds["round"], errors="coerce").eq(1)].copy()
    required_round = {
        "fight_id", "fighter_id", "fighter_name", "kd", "sig_str_landed", "ground_landed"
    }
    missing = sorted(required_round - set(r1.columns))
    if missing:
        raise RuntimeError(f"Round stats missing required columns: {missing}")

    r1["fight_id"] = r1["fight_id"].astype(str)
    r1["fighter_id"] = r1["fighter_id"].astype(str)
    for c in ("kd", "sig_str_landed", "ground_landed"):
        r1[c] = pd.to_numeric(r1[c], errors="coerce").fillna(0.0)

    fight_cols = [
        "fight_id", "date", "method", "finish_round", "match_time_sec",
        "winner_id", "r_id", "b_id", "r_name", "b_name",
    ]
    missing_master = [c for c in fight_cols if c not in master.columns]
    if missing_master:
        raise RuntimeError(f"Master missing required columns: {missing_master}")

    fights = master[fight_cols].copy()
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["winner_id"] = fights["winner_id"].astype(str)
    fights["finish_round"] = pd.to_numeric(fights["finish_round"], errors="coerce")
    fights["match_time_sec"] = pd.to_numeric(fights["match_time_sec"], errors="coerce")

    out = r1.merge(
        fights[["fight_id", "date", "method", "finish_round", "match_time_sec", "winner_id"]],
        on="fight_id",
        how="inner",
        validate="many_to_one",
    )
    out["r1_ko_tko_win"] = (
        out["fighter_id"].eq(out["winner_id"])
        & out["finish_round"].eq(1)
        & out["method"].map(is_ko_tko)
    )
    return out


def evidence_for_row(row: pd.Series) -> tuple[float, float, float]:
    kd_evidence = max(0.0, float(row["kd"]))
    early_ko_evidence = 0.0

    if bool(row["r1_ko_tko_win"]):
        finish_sec = max(0.0, float(row["match_time_sec"]))
        sig = max(0.0, float(row["sig_str_landed"]))
        ground = max(0.0, float(row["ground_landed"]))

        freshness = exp(-finish_sec / TIME_TAU_SECONDS)
        low_volume = exp(-max(sig - 1.0, 0.0) / SIG_LANDED_TAU)
        low_ground = exp(-ground / GROUND_LANDED_TAU)
        early_ko_evidence = EARLY_KO_WEIGHT * freshness * low_volume * low_ground

    return kd_evidence, early_ko_evidence, kd_evidence + early_ko_evidence


def build_rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    pieces = scored.apply(evidence_for_row, axis=1, result_type="expand")
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

    s = agg["cumulative_power_evidence"].astype(float)
    agg["power_evidence_rating"] = 50.0 + 40.0 * (1.0 - np.exp(-s / EVIDENCE_SATURATION))
    agg["power_evidence_rating"] = agg["power_evidence_rating"].clip(50.0, 90.0)
    agg = agg.sort_values(
        ["power_evidence_rating", "cumulative_power_evidence", "power_event_fights"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    return agg, scored


def main() -> None:
    args = parse_args()
    if not args.master.exists():
        raise FileNotFoundError(args.master)
    if not args.round_stats.exists():
        raise FileNotFoundError(args.round_stats)

    master = pd.read_parquet(args.master)
    rounds = pd.read_parquet(args.round_stats)
    fighter_fights = fighter_fight_frame(master, rounds)
    rankings, scored = build_rankings(fighter_fights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    detail_path = args.output.with_name(args.output.stem + "_fighter_fight_detail.csv")
    scored.to_csv(detail_path, index=False)

    top = rankings.head(max(1, args.top)).copy()
    display_cols = [
        "rank", "fighter_name", "power_evidence_rating", "cumulative_power_evidence",
        "r1_kds", "r1_ko_tko_wins", "power_event_fights", "max_single_fight_evidence",
        "ufc_r1_fights",
    ]

    print("\n" + "=" * 118)
    print("FRESH STRIKING POWER EVIDENCE V2 — ALL-FIGHTER FACE-VALIDITY SWEEP")
    print("=" * 118)
    print(f"fighter-fight R1 rows: {len(fighter_fights):,}")
    print(f"fighters ranked: {len(rankings):,}")
    print("\nEquation:")
    print("  E_KD = R1_KD")
    print(
        "  E_EARLY_KO = 2.5 * I(R1 KO/TKO win) * exp(-t/180) "
        "* exp(-max(R1_SIG_LANDED-1,0)/20) * exp(-R1_GROUND_LANDED/10)"
    )
    print("  S = sum(E_KD + E_EARLY_KO); no negative updates")
    print("  provisional rating = 50 + 40*(1-exp(-S/4))")
    print("\nTOP FIGHTERS")
    print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote rankings: {args.output}")
    print(f"Wrote fighter-fight evidence detail: {detail_path}")
    print("Research only: this script does not modify any FSR artifact.")


if __name__ == "__main__":
    main()
