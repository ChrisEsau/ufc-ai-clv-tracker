"""Build leakage-safe raw phase/action initiative profiles for the mature 2020+ cohort.

Research-only audit. This script does NOT create new FSR traits and does NOT
modify simulator physics. It preserves raw historical action evidence so that
new initiative/propensity traits can be designed from first principles.

For every fighter-fight in the aligned mature cohort, the snapshot contains only
UFCStats rounds from fights strictly before that fight date / fight row.

Outputs
-------
data/experimental/mature_phase_initiative_raw/
    cohort_prefight_profiles.csv
        Leakage-safe pre-fight profiles for every fighter-side in the mature cohort.
    latest_cohort_fighter_profiles.csv
        Most recent leakage-safe pre-fight snapshot for each fighter appearing in
        the mature cohort. Intended for later ranking / sanity checks.

Raw initiative evidence retained
---------------------------------
- distance / clinch / ground significant-strike attempts
- takedown attempts
- per-round rates
- significant-strike phase shares
- action-presence rates by prior fight and prior round

Important boundary
------------------
UFCStats does not provide exact seconds spent at distance, clinch, or ground.
These metrics describe observed action tendency, not exact phase occupancy.
Ground strike attempts are downstream of reaching ground and therefore should not
be interpreted as ground-entry intent without additional modeling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH
from pipeline.round_stats import build_round_fighter_phase_baseline as phase
from scripts.experimental import fsr_32_historical_cohort as cohort32

OUTPUT_DIR = Path("data/experimental/mature_phase_initiative_raw")
PREFIGHT_PATH = OUTPUT_DIR / "cohort_prefight_profiles.csv"
LATEST_PATH = OUTPUT_DIR / "latest_cohort_fighter_profiles.csv"

COUNT_COLS = {
    "sig": "sig_str_attempted",
    "distance": "distance_attempted",
    "clinch": "clinch_attempted",
    "ground": "ground_attempted",
    "td": "td_attempted",
}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _build_fighter_fights(rounds: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter-fight with raw action counts and activity indicators."""
    group_cols = [
        "fight_id",
        "fighter_id",
        "fighter_name",
        "date",
    ]
    fight = (
        rounds.groupby(group_cols, dropna=False)
        .agg(
            rounds_observed=("round", "nunique"),
            sig_attempts=("sig_str_attempted", "sum"),
            distance_attempts=("distance_attempted", "sum"),
            clinch_attempts=("clinch_attempted", "sum"),
            ground_attempts=("ground_attempted", "sum"),
            td_attempts=("td_attempted", "sum"),
            rounds_with_distance=("distance_attempted", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
            rounds_with_clinch=("clinch_attempted", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
            rounds_with_ground=("ground_attempted", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
            rounds_with_td=("td_attempted", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
        )
        .reset_index()
    )
    for action in ("distance", "clinch", "ground", "td"):
        fight[f"fight_has_{action}"] = (fight[f"{action}_attempts"] > 0).astype(int)
    return fight.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


def _add_prefight_cumulative_state(fight: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative totals shifted one fight so every state is leakage-safe."""
    df = fight.copy()
    g = df.groupby("fighter_id", sort=False, group_keys=False)

    # Prior exposure.
    df["prior_fights"] = g.cumcount()
    for col in [
        "rounds_observed",
        "sig_attempts",
        "distance_attempts",
        "clinch_attempts",
        "ground_attempts",
        "td_attempts",
        "rounds_with_distance",
        "rounds_with_clinch",
        "rounds_with_ground",
        "rounds_with_td",
        "fight_has_distance",
        "fight_has_clinch",
        "fight_has_ground",
        "fight_has_td",
    ]:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[f"prior_{col}"] = (
            values.groupby(df["fighter_id"])
            .transform(lambda s: s.shift(1, fill_value=0.0).cumsum())
        )

    # Per-round rates: direct historical behavior, not FSR ratings.
    for action in ("sig", "distance", "clinch", "ground", "td"):
        base = "sig_attempts" if action == "sig" else f"{action}_attempts"
        df[f"prior_{action}_attempts_per_round"] = _safe_div(
            df[f"prior_{base}"], df["prior_rounds_observed"]
        )

    # Within-significant-striking phase shares.
    for action in ("distance", "clinch", "ground"):
        df[f"prior_{action}_strike_attempt_share"] = _safe_div(
            df[f"prior_{action}_attempts"], df["prior_sig_attempts"]
        )

    # How often the fighter actually demonstrates each action across opportunities.
    for action in ("distance", "clinch", "ground", "td"):
        df[f"prior_{action}_active_fight_rate"] = _safe_div(
            df[f"prior_fight_has_{action}"], df["prior_fights"]
        )
        df[f"prior_{action}_active_round_rate"] = _safe_div(
            df[f"prior_rounds_with_{action}"], df["prior_rounds_observed"]
        )

    # Helpful interpretable TD frequency relative to strike activity. This is an
    # audit diagnostic only; it is NOT proposed as the final initiative formula.
    df["prior_td_attempts_per_100_sig_attempts"] = 100.0 * _safe_div(
        df["prior_td_attempts"], df["prior_sig_attempts"]
    )

    return df


def _cohort_side_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, id_col in (("red", "r_id"), ("blue", "b_id")):
        part = cohort[["bout_id", "event_date", id_col]].copy()
        part = part.rename(columns={id_col: "fighter_id"})
        part["side"] = side
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["bout_id"] = out["bout_id"].astype(str)
    out["fighter_id"] = out["fighter_id"].astype(str)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort, _ = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    cohort["event_date"] = pd.to_datetime(cohort["event_date"], errors="coerce")

    raw = pd.read_parquet(ROUND_STATS_PATH)
    rounds = phase.standardize_round_stats(raw)
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)

    fight = _build_fighter_fights(rounds)
    states = _add_prefight_cumulative_state(fight)

    # Exact point-in-time state for each cohort fighter-side: match the fighter's
    # row for that same bout. Since all cumulative features are shifted, the
    # current fight contributes zero information to its own snapshot.
    state_cols = [
        c for c in states.columns
        if c.startswith("prior_")
    ]
    state_lookup = states[[
        "fight_id", "fighter_id", "fighter_name", "date", *state_cols
    ]].rename(columns={"fight_id": "bout_id", "date": "state_date"})

    sides = _cohort_side_rows(cohort)
    profiles = sides.merge(
        state_lookup,
        on=["bout_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    missing = int(profiles["fighter_name"].isna().sum())
    if missing:
        raise RuntimeError(f"Missing point-in-time phase state for {missing} cohort fighter-sides")

    profiles = profiles.sort_values(["event_date", "bout_id", "side"]).reset_index(drop=True)
    profiles.to_csv(PREFIGHT_PATH, index=False)

    # Most recent leakage-safe cohort snapshot for each fighter. Require the same
    # >=3-prior-fight maturity that defines the aligned cohort.
    latest = (
        profiles.sort_values(["event_date", "bout_id"])
        .groupby("fighter_id", as_index=False, sort=False)
        .tail(1)
        .sort_values("fighter_name")
        .reset_index(drop=True)
    )
    latest.to_csv(LATEST_PATH, index=False)

    print("\n" + "=" * 118)
    print("MATURE 2020+ RAW PHASE / ACTION INITIATIVE PROFILE AUDIT")
    print("=" * 118)
    print(f"aligned cohort bouts: {len(cohort):,}")
    print(f"cohort fighter-side snapshots: {len(profiles):,}")
    print(f"unique cohort fighters: {latest['fighter_id'].nunique():,}")
    print("All profile values are leakage-safe and use only fights before each cohort bout.")
    print("No new FSR ratings or MC parameters are created by this audit.")

    metrics = [
        ("DISTANCE attempts/round", "prior_distance_attempts_per_round"),
        ("CLINCH attempts/round", "prior_clinch_attempts_per_round"),
        ("GROUND strikes/round", "prior_ground_attempts_per_round"),
        ("TD attempts/round", "prior_td_attempts_per_round"),
        ("TD-active fight rate", "prior_td_active_fight_rate"),
        ("CLINCH-active fight rate", "prior_clinch_active_fight_rate"),
    ]

    for label, col in metrics:
        print(f"\nTOP 15 — {label}")
        show = latest.sort_values(col, ascending=False).head(15)[
            ["fighter_name", "prior_fights", "prior_rounds_observed", col]
        ]
        print(show.to_string(index=False, formatters={col: lambda x: f"{x:.3f}" if pd.notna(x) else "nan"}))

    # Specific matchup diagnostic that motivated the audit.
    names = latest[latest["fighter_name"].str.contains("Dober|Frevola", case=False, na=False)].copy()
    if not names.empty:
        print("\nDOBER / FREVOLA — MOST RECENT MATURE-COHORT PRE-FIGHT SNAPSHOTS")
        cols = [
            "fighter_name",
            "event_date",
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
        ]
        print(names[cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {PREFIGHT_PATH}")
    print(f"  {LATEST_PATH}")
    print("\nNEXT STEP")
    print("Use these raw distributions to define candidate initiative traits; then rank fighters and sanity-check archetypes before touching FSR-32 or the simulator.")


if __name__ == "__main__":
    main()
