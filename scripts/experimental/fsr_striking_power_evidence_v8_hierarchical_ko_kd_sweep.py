"""Research-only V8 fresh striking-power sweep.

V8 keeps the V7 Round-1 KO/TKO evidence as the primary power signal, then adds
back two weaker evidence sources:

1. later-round KO/TKO wins, discounted by finish round;
2. Round-1 knockdowns, heavily discounted so they cannot dominate the trait.

Hierarchy:
    R1 KO/TKO  >>  later-round KO/TKO  >  R1 KD

R1 KO evidence is unchanged from V7:

    E_R1_KO = 2.5 * I(R1 KO/TKO win)
                   * exp(-finish_seconds / 240)
                   * exp(-max(R1_sig_landed - 1, 0) / 35)
                   * exp(-R1_ground_landed / 20)

Later-round KO evidence uses the same within-round severity modifiers but is
multiplied by a round discount:

    R2 = 0.60
    R3 = 0.40
    R4 = 0.30
    R5+ = 0.20

Round-1 KD evidence is deliberately small:

    E_R1_KD = 0.20 * R1_KD * (1 + exp(-R1_sig_landed / 20))

The career aggregation remains the V4/V7 structure so this experiment isolates
the effect of restoring these secondary evidence channels. Quiet fights never
subtract demonstrated power. Research only; no FSR artifact is modified.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as v2
from scripts.experimental import fsr_striking_power_evidence_v4_aggregation_sweep as v4
from scripts.experimental import fsr_striking_power_evidence_v7_ko_only_softened_penalties_sweep as v7

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_evidence_v8_hierarchical_ko_kd_rankings.csv"
)

# Primary R1 KO evidence is frozen from V7.
EARLY_KO_WEIGHT = v7.EARLY_KO_WEIGHT
TIME_TAU_SECONDS = v7.TIME_TAU_SECONDS
SIG_LANDED_TAU = v7.SIG_LANDED_TAU
GROUND_LANDED_TAU = v7.GROUND_LANDED_TAU

# Secondary evidence controls.
KD_WEIGHT = 0.20
KD_EFFICIENCY_TAU = 20.0
ROUND_KO_WEIGHTS = {
    1: 1.00,
    2: 0.60,
    3: 0.40,
    4: 0.30,
}
ROUND_5PLUS_KO_WEIGHT = 0.20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=v2.MASTER_PATH)
    p.add_argument("--round-stats", type=Path, default=v2.ROUND_STATS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def _round_weight(round_number: int) -> float:
    if round_number >= 5:
        return ROUND_5PLUS_KO_WEIGHT
    return ROUND_KO_WEIGHTS.get(round_number, 0.0)


def _within_round_finish_seconds(match_time_sec: float, finish_round: int) -> float:
    """Support either total-elapsed or within-round master timing conventions."""
    t = max(0.0, float(match_time_sec))
    if finish_round > 1 and t > 300.0:
        t = t - 300.0 * (finish_round - 1)
    return float(np.clip(t, 0.0, 300.0))


def _ko_severity_evidence(
    finish_round: int,
    match_time_sec: float,
    sig_landed: float,
    ground_landed: float,
) -> float:
    round_weight = _round_weight(int(finish_round))
    if round_weight <= 0.0:
        return 0.0

    within_round_sec = _within_round_finish_seconds(match_time_sec, int(finish_round))
    sig = max(0.0, float(sig_landed))
    ground = max(0.0, float(ground_landed))

    freshness = exp(-within_round_sec / TIME_TAU_SECONDS)
    low_volume = exp(-max(sig - 1.0, 0.0) / SIG_LANDED_TAU)
    low_ground = exp(-ground / GROUND_LANDED_TAU)
    return EARLY_KO_WEIGHT * round_weight * freshness * low_volume * low_ground


def build_all_round_frame(master: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    required_round = {
        "fight_id", "fighter_id", "fighter_name", "round", "kd",
        "sig_str_landed", "ground_landed",
    }
    missing = sorted(required_round - set(rounds.columns))
    if missing:
        raise RuntimeError(f"Round stats missing required columns: {missing}")

    required_master = {
        "fight_id", "method", "finish_round", "match_time_sec", "winner_id"
    }
    missing_master = sorted(required_master - set(master.columns))
    if missing_master:
        raise RuntimeError(f"Master missing required columns: {missing_master}")

    r = rounds[list(required_round)].copy()
    r["fight_id"] = r["fight_id"].astype(str)
    r["fighter_id"] = r["fighter_id"].astype(str)
    r["round"] = pd.to_numeric(r["round"], errors="coerce")
    for c in ("kd", "sig_str_landed", "ground_landed"):
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0.0)

    f = master[list(required_master)].copy()
    f["fight_id"] = f["fight_id"].astype(str)
    f["winner_id"] = f["winner_id"].astype(str)
    f["finish_round"] = pd.to_numeric(f["finish_round"], errors="coerce")
    f["match_time_sec"] = pd.to_numeric(f["match_time_sec"], errors="coerce")
    f["is_ko_tko"] = f["method"].map(v2.is_ko_tko)

    out = r.merge(f, on="fight_id", how="inner", validate="many_to_one")
    out["is_finish_round"] = out["round"].eq(out["finish_round"])
    out["is_ko_tko_win_finish_round"] = (
        out["fighter_id"].eq(out["winner_id"])
        & out["is_ko_tko"]
        & out["is_finish_round"]
    )
    return out


def build_v8_rankings(master: pd.DataFrame, rounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rounds = build_all_round_frame(master, rounds)

    # One row per fighter-fight is the career evidence grain. R1 is also the
    # source of the small KD term and the sample/opportunity denominator.
    r1 = all_rounds.loc[all_rounds["round"].eq(1)].copy()
    if r1.empty:
        raise RuntimeError("No Round-1 rows available")

    r1["r1_kd_evidence_v8"] = (
        KD_WEIGHT
        * r1["kd"].clip(lower=0.0)
        * (1.0 + np.exp(-r1["sig_str_landed"].clip(lower=0.0) / KD_EFFICIENCY_TAU))
    )

    # KO evidence can come from the actual finishing round.
    finish_rows = all_rounds.loc[all_rounds["is_ko_tko_win_finish_round"]].copy()
    finish_rows["ko_evidence_v8"] = finish_rows.apply(
        lambda row: _ko_severity_evidence(
            int(row["finish_round"]),
            float(row["match_time_sec"]),
            float(row["sig_str_landed"]),
            float(row["ground_landed"]),
        ),
        axis=1,
    )
    finish_rows["r1_ko_evidence_v8"] = np.where(
        finish_rows["finish_round"].eq(1), finish_rows["ko_evidence_v8"], 0.0
    )
    finish_rows["later_ko_evidence_v8"] = np.where(
        finish_rows["finish_round"].gt(1), finish_rows["ko_evidence_v8"], 0.0
    )

    ko_by_fight = finish_rows[[
        "fight_id", "fighter_id", "ko_evidence_v8", "r1_ko_evidence_v8",
        "later_ko_evidence_v8", "finish_round",
    ]].copy()

    scored = r1.merge(
        ko_by_fight,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    for c in ("ko_evidence_v8", "r1_ko_evidence_v8", "later_ko_evidence_v8"):
        scored[c] = scored[c].fillna(0.0)
    scored["ko_finish_round_v8"] = scored["finish_round_y"].fillna(0.0)
    scored["fight_power_evidence_v8"] = (
        scored["ko_evidence_v8"] + scored["r1_kd_evidence_v8"]
    )
    scored["power_event"] = scored["fight_power_evidence_v8"].gt(0.0)
    scored["r1_ko_event"] = scored["r1_ko_evidence_v8"].gt(0.0)
    scored["later_ko_event"] = scored["later_ko_evidence_v8"].gt(0.0)

    agg = scored.groupby(["fighter_id", "fighter_name"], as_index=False).agg(
        ufc_r1_fights=("fight_id", "nunique"),
        r1_kds=("kd", "sum"),
        r1_ko_tko_wins=("r1_ko_event", "sum"),
        later_ko_tko_wins=("later_ko_event", "sum"),
        power_event_fights=("power_event", "sum"),
        r1_ko_evidence_v8=("r1_ko_evidence_v8", "sum"),
        later_ko_evidence_v8=("later_ko_evidence_v8", "sum"),
        r1_kd_evidence_v8=("r1_kd_evidence_v8", "sum"),
        cumulative_power_evidence_v8=("fight_power_evidence_v8", "sum"),
        max_single_fight_evidence_v8=("fight_power_evidence_v8", "max"),
    )

    fights = agg["ufc_r1_fights"].astype(float).clip(lower=1.0)
    events = agg["power_event_fights"].astype(float)
    peak = agg["max_single_fight_evidence_v8"].astype(float).clip(lower=0.0)

    agg["power_event_rate"] = (events / fights).clip(0.0, 1.0)
    agg["repeatability_term"] = 1.0 - np.exp(-events / v4.REPEATABILITY_TAU)
    agg["sample_confidence"] = 1.0 - np.exp(-fights / v4.CONFIDENCE_TAU)
    agg["adjusted_frequency_term"] = (
        np.sqrt(agg["power_event_rate"]) * agg["sample_confidence"]
    )
    agg["compressed_peak_term"] = 1.0 - np.exp(-peak / v4.PEAK_TAU)

    agg["v8_power_score"] = (
        v4.REPEATABILITY_WEIGHT * agg["repeatability_term"]
        + v4.FREQUENCY_WEIGHT * agg["adjusted_frequency_term"]
        + v4.PEAK_WEIGHT * agg["compressed_peak_term"]
    )
    s = agg["v8_power_score"].astype(float)
    agg["power_evidence_rating_v8"] = (
        50.0 + 40.0 * (1.0 - np.exp(-s / v4.DISPLAY_RATING_SATURATION))
    ).clip(50.0, 90.0)
    agg.loc[agg["power_event_fights"].eq(0), "power_evidence_rating_v8"] = 50.0

    agg = agg.sort_values(
        ["v8_power_score", "repeatability_term", "adjusted_frequency_term", "compressed_peak_term"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank_v8", np.arange(1, len(agg) + 1))
    return agg, scored


def main() -> None:
    args = parse_args()
    if not args.master.exists():
        raise FileNotFoundError(args.master)
    if not args.round_stats.exists():
        raise FileNotFoundError(args.round_stats)

    master = pd.read_parquet(args.master)
    rounds = pd.read_parquet(args.round_stats)
    rankings, scored = build_v8_rankings(master, rounds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    detail_path = args.output.with_name(args.output.stem + "_fighter_fight_detail.csv")
    scored.to_csv(detail_path, index=False)

    top = rankings.head(max(1, args.top)).copy()
    display_cols = [
        "rank_v8", "fighter_name", "power_evidence_rating_v8", "v8_power_score",
        "r1_ko_tko_wins", "later_ko_tko_wins", "r1_kds",
        "r1_ko_evidence_v8", "later_ko_evidence_v8", "r1_kd_evidence_v8",
        "power_event_fights", "power_event_rate", "ufc_r1_fights",
    ]

    print("\n" + "=" * 175)
    print("FRESH STRIKING POWER EVIDENCE V8 — HIERARCHICAL KO/TKO + SECONDARY R1 KD")
    print("=" * 175)
    print(f"fighters ranked: {len(rankings):,}")
    print("\nEvidence hierarchy:")
    print("  R1 KO/TKO: full V7 evidence")
    print("  Later KO/TKO round weights: R2=.60, R3=.40, R4=.30, R5+=.20")
    print("  R1 KD: 0.20 * KD * (1 + exp(-R1_SIG_LANDED/20))")
    print("  quiet fights never subtract demonstrated power")
    print("\nCareer aggregation: frozen V4/V7 structure")

    print("\nTOP FIGHTERS")
    print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    neutral = rankings.loc[np.isclose(rankings["power_evidence_rating_v8"], 50.0)]
    print("\nDISTRIBUTION SNAPSHOT")
    print(f"neutral 50 fighters: {len(neutral):,} / {len(rankings):,} ({len(neutral)/max(len(rankings),1)*100:.2f}%)")
    print(f"mean rating: {rankings['power_evidence_rating_v8'].mean():.3f}")
    print(f"median rating: {rankings['power_evidence_rating_v8'].median():.3f}")
    for p in (10, 25, 50, 75, 90, 95, 99):
        print(f"p{p:02d}: {np.percentile(rankings['power_evidence_rating_v8'], p):.3f}")

    print(f"\nWrote rankings: {args.output}")
    print(f"Wrote fighter-fight evidence detail: {detail_path}")
    print("Research only: this script does not modify any FSR artifact.")


if __name__ == "__main__":
    main()
