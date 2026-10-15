"""Rank research-only candidate phase initiative traits for the mature cohort.

Purpose
-------
Use the raw leakage-safe profiles created by
``audit_mature_phase_initiative_raw_profiles.py`` to construct simple, interpretable
candidate initiative scores for:

- distance initiative
- clinch initiative
- takedown initiative
- ground-offense initiative

These are NOT promoted FSR traits and are NOT simulator parameters. They are
sanity-check candidates intended to answer whether the rankings resemble known
fighter styles before any architecture change.

Scoring philosophy
------------------
Each candidate score combines three kinds of behavior evidence:

1. volume: how often the fighter attempts the action per observed round;
2. share: how much of the fighter's recorded offense occurs in that phase/action;
3. persistence: how consistently prior fights/rounds contain that action.

All component values are converted to empirical percentiles across the latest
mature-cohort fighter states, then blended and mapped to a 10-90 display scale.
No opponent-quality or success metric is included: initiative is about what a
fighter chooses to do, not whether it works.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("data/experimental/mature_phase_initiative_raw")
COHORT_PATH = INPUT_DIR / "cohort_prefight_profiles.csv"
LATEST_PATH = INPUT_DIR / "latest_cohort_fighter_profiles.csv"
OUTPUT_DIR = Path("data/experimental/mature_phase_initiative_candidates")
RANKINGS_PATH = OUTPUT_DIR / "latest_candidate_initiative_rankings.csv"

DOBER_FREVOLA_BOUT_ID = "3f8b4aeb3baf4724"


def _pct(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    # rank(pct=True) is stable for ties and preserves missing values.
    return x.rank(method="average", pct=True)


def _score(*parts: tuple[float, pd.Series]) -> pd.Series:
    out = None
    weight_sum = 0.0
    for weight, values in parts:
        weighted = float(weight) * values
        out = weighted if out is None else out.add(weighted, fill_value=0.0)
        weight_sum += float(weight)
    if out is None or weight_sum <= 0:
        raise ValueError("candidate score requires positive-weight inputs")
    unit = out / weight_sum
    return 10.0 + 80.0 * unit


def _ensure_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def build_scores(latest: pd.DataFrame) -> pd.DataFrame:
    df = latest.copy()

    required = [
        "fighter_id",
        "fighter_name",
        "prior_fights",
        "prior_rounds_observed",
        "prior_distance_attempts_per_round",
        "prior_clinch_attempts_per_round",
        "prior_ground_attempts_per_round",
        "prior_td_attempts_per_round",
        "prior_distance_strike_attempt_share",
        "prior_clinch_strike_attempt_share",
        "prior_ground_strike_attempt_share",
        "prior_td_active_fight_rate",
        "prior_td_active_round_rate",
        "prior_clinch_active_fight_rate",
        "prior_clinch_active_round_rate",
        "prior_distance_active_fight_rate",
        "prior_distance_active_round_rate",
        "prior_ground_active_fight_rate",
        "prior_ground_active_round_rate",
    ]
    _ensure_columns(df, required, "latest raw profiles")

    # Component percentiles. Success metrics are intentionally excluded.
    d_vol = _pct(df["prior_distance_attempts_per_round"])
    d_share = _pct(df["prior_distance_strike_attempt_share"])
    d_fight = _pct(df["prior_distance_active_fight_rate"])
    d_round = _pct(df["prior_distance_active_round_rate"])

    c_vol = _pct(df["prior_clinch_attempts_per_round"])
    c_share = _pct(df["prior_clinch_strike_attempt_share"])
    c_fight = _pct(df["prior_clinch_active_fight_rate"])
    c_round = _pct(df["prior_clinch_active_round_rate"])

    td_vol = _pct(df["prior_td_attempts_per_round"])
    td_fight = _pct(df["prior_td_active_fight_rate"])
    td_round = _pct(df["prior_td_active_round_rate"])

    g_vol = _pct(df["prior_ground_attempts_per_round"])
    g_share = _pct(df["prior_ground_strike_attempt_share"])
    g_fight = _pct(df["prior_ground_active_fight_rate"])
    g_round = _pct(df["prior_ground_active_round_rate"])

    # Candidate weights emphasize observed volume but require persistence so one
    # explosive fight does not dominate the ranking. Shares distinguish fighters
    # with similar volume but different style allocation.
    df["distance_initiative"] = _score(
        (0.45, d_vol), (0.30, d_share), (0.15, d_fight), (0.10, d_round)
    )
    df["clinch_initiative"] = _score(
        (0.40, c_vol), (0.30, c_share), (0.15, c_fight), (0.15, c_round)
    )
    df["takedown_initiative"] = _score(
        (0.55, td_vol), (0.25, td_fight), (0.20, td_round)
    )
    df["ground_offense_initiative"] = _score(
        (0.40, g_vol), (0.30, g_share), (0.15, g_fight), (0.15, g_round)
    )

    score_cols = [
        "distance_initiative",
        "clinch_initiative",
        "takedown_initiative",
        "ground_offense_initiative",
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(10.0, 90.0)

    return df


def _print_top(df: pd.DataFrame, score: str, label: str, n: int = 20) -> None:
    cols = [
        "fighter_name",
        "prior_fights",
        "prior_rounds_observed",
        score,
    ]
    print(f"\nTOP {n} — {label}")
    print(
        df.sort_values(score, ascending=False)[cols]
        .head(n)
        .to_string(index=False, formatters={score: lambda x: f"{x:.2f}"})
    )


def _print_bottom(df: pd.DataFrame, score: str, label: str, n: int = 10) -> None:
    cols = ["fighter_name", "prior_fights", "prior_rounds_observed", score]
    print(f"\nBOTTOM {n} — {label}")
    print(
        df.sort_values(score, ascending=True)[cols]
        .head(n)
        .to_string(index=False, formatters={score: lambda x: f"{x:.2f}"})
    )


def _dober_frevola_exact_snapshot(cohort: pd.DataFrame, scored_latest: pd.DataFrame) -> pd.DataFrame:
    if "bout_id" not in cohort.columns:
        return pd.DataFrame()
    exact = cohort[cohort["bout_id"].astype(str) == DOBER_FREVOLA_BOUT_ID].copy()
    if exact.empty:
        return exact

    # Candidate percentiles are defined against the latest cohort distribution.
    # Reconstruct each exact snapshot's percentile location against that same
    # reference population so the scores are directly comparable with rankings.
    ref = scored_latest.copy()

    def percentile_against_ref(value: float, col: str) -> float:
        vals = pd.to_numeric(ref[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0 or pd.isna(value):
            return np.nan
        return float(np.mean(vals <= float(value)))

    raw_map = {
        "d_vol": "prior_distance_attempts_per_round",
        "d_share": "prior_distance_strike_attempt_share",
        "d_fight": "prior_distance_active_fight_rate",
        "d_round": "prior_distance_active_round_rate",
        "c_vol": "prior_clinch_attempts_per_round",
        "c_share": "prior_clinch_strike_attempt_share",
        "c_fight": "prior_clinch_active_fight_rate",
        "c_round": "prior_clinch_active_round_rate",
        "td_vol": "prior_td_attempts_per_round",
        "td_fight": "prior_td_active_fight_rate",
        "td_round": "prior_td_active_round_rate",
        "g_vol": "prior_ground_attempts_per_round",
        "g_share": "prior_ground_strike_attempt_share",
        "g_fight": "prior_ground_active_fight_rate",
        "g_round": "prior_ground_active_round_rate",
    }

    rows = []
    for _, r in exact.iterrows():
        p = {k: percentile_against_ref(r.get(col), col) for k, col in raw_map.items()}
        row = r.to_dict()
        row["distance_initiative"] = 10 + 80 * (
            0.45*p["d_vol"] + 0.30*p["d_share"] + 0.15*p["d_fight"] + 0.10*p["d_round"]
        )
        row["clinch_initiative"] = 10 + 80 * (
            0.40*p["c_vol"] + 0.30*p["c_share"] + 0.15*p["c_fight"] + 0.15*p["c_round"]
        )
        row["takedown_initiative"] = 10 + 80 * (
            0.55*p["td_vol"] + 0.25*p["td_fight"] + 0.20*p["td_round"]
        )
        row["ground_offense_initiative"] = 10 + 80 * (
            0.40*p["g_vol"] + 0.30*p["g_share"] + 0.15*p["g_fight"] + 0.15*p["g_round"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not LATEST_PATH.exists() or not COHORT_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/experimental/audit_mature_phase_initiative_raw_profiles.py first"
        )

    latest = pd.read_csv(LATEST_PATH)
    cohort = pd.read_csv(COHORT_PATH)
    scored = build_scores(latest)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(RANKINGS_PATH, index=False)

    print("\n" + "=" * 118)
    print("MATURE COHORT — CANDIDATE PHASE INITIATIVE RANKINGS")
    print("=" * 118)
    print(f"fighters ranked: {len(scored):,}")
    print("Research-only candidate scores. No FSR-32 or simulator changes.")
    print("10-90 scale is percentile-derived, NOT an Elo ability scale.")
    print("Initiative intentionally excludes accuracy, TD completion, control payoff, and other success metrics.")

    _print_top(scored, "distance_initiative", "DISTANCE INITIATIVE")
    _print_top(scored, "clinch_initiative", "CLINCH INITIATIVE")
    _print_top(scored, "takedown_initiative", "TAKEDOWN INITIATIVE")
    _print_top(scored, "ground_offense_initiative", "GROUND-OFFENSE INITIATIVE")

    _print_bottom(scored, "takedown_initiative", "TAKEDOWN INITIATIVE")

    exact = _dober_frevola_exact_snapshot(cohort, scored)
    if not exact.empty:
        print("\nDOBER / FREVOLA — EXACT PRE-FIGHT SNAPSHOT FOR MAY 6, 2023 BOUT")
        cols = [
            "fighter_name",
            "event_date",
            "prior_fights",
            "prior_rounds_observed",
            "prior_distance_attempts_per_round",
            "prior_clinch_attempts_per_round",
            "prior_ground_attempts_per_round",
            "prior_td_attempts_per_round",
            "distance_initiative",
            "clinch_initiative",
            "takedown_initiative",
            "ground_offense_initiative",
        ]
        cols = [c for c in cols if c in exact.columns]
        print(exact[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\nDober/Frevola exact bout snapshot not found in cohort CSV.")

    print("\nSaved:")
    print(f"  {RANKINGS_PATH}")
    print("\nNEXT")
    print("Sanity-check the names at the top and bottom of each ranking. If the archetypes make sense, then test candidate scores for forward prediction of next-fight action rates before promoting any trait.")


if __name__ == "__main__":
    main()
