"""Research-only phase-choice signal audit from leakage-safe FSR-26 snapshots.

Question
--------
Can the existing FSR profile recover a fighter's realized phase mix well enough
to seed Monte Carlo phase-choice logic without adding dedicated phase-choice
ratings?

This script does NOT modify ratings or simulator behavior. It derives candidate
pre-fight preference scores from FSR-26, joins them to the current fight's RFS
outcomes, and prints forward-looking bucket/correlation diagnostics.

Candidate families
------------------
A) pressure_only
   Distance = distance striking pressure
   Clinch   = clinch striking pressure
   Wrestling= wrestling entry

B) control_blend
   Distance = distance striking pressure
   Clinch   = 70% clinch striking pressure + 30% control imposition
   Wrestling= 70% wrestling entry + 30% control imposition

C) matchup_adjusted
   Starts from B and adjusts phase-entry desirability for opponent resistance:
   Clinch   -= 25% of opponent control resistance deviation from 50
   Wrestling-= 25% of opponent TD-defense deviation from 50

Within each candidate, the three scores are centered against their row mean.
This makes them RELATIVE phase preferences rather than absolute activity scores.

Shadow/research only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FSR26_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")

KEYS = ["fight_id", "fighter_id"]

FSR_COLUMNS = [
    *KEYS,
    "fighter_name",
    "distance_striking_pressure",
    "clinch_striking_pressure",
    "wrestling_entry",
    "control_imposition",
    "td_defense",
    "control_resistance",
]

RFS_COLUMNS = [
    *KEYS,
    "rfs_phase_base_fight_distance_attempt_share",
    "rfs_phase_base_fight_clinch_attempt_share",
    "rfs_phase_base_fight_ground_attempt_share",
    "rfs_phase_base_fight_td_attempts_per_round",
    "rfs_phase_base_fight_control_seconds_per_round",
]

OUTCOME_COLUMNS = {
    "distance": "rfs_phase_base_fight_distance_attempt_share",
    "clinch": "rfs_phase_base_fight_clinch_attempt_share",
    "ground": "rfs_phase_base_fight_ground_attempt_share",
    "td_rate": "rfs_phase_base_fight_td_attempts_per_round",
    "control_rate": "rfs_phase_base_fight_control_seconds_per_round",
}

CANDIDATES = ("pressure_only", "control_blend", "matchup_adjusted")
PHASES = ("distance", "clinch", "wrestling")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for key in KEYS:
        out[key] = out[key].astype(str)
    return out


def _attach_opponent_traits(fsr: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent defensive traits at exact fighter-fight grain."""
    base = _normalize_keys(fsr)
    opp = base[
        ["fight_id", "fighter_id", "td_defense", "control_resistance"]
    ].rename(
        columns={
            "fighter_id": "opponent_id",
            "td_defense": "opponent_td_defense",
            "control_resistance": "opponent_control_resistance",
        }
    )

    paired = base.merge(opp, on="fight_id", how="inner")
    paired = paired[paired["fighter_id"] != paired["opponent_id"]].copy()

    if paired.duplicated(KEYS).any():
        raise RuntimeError("opponent attachment violates fighter-fight grain")
    return paired


def _center_triplet(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{phase}_raw" for phase in PHASES]
    row_mean = frame[cols].mean(axis=1)
    for phase, col in zip(PHASES, cols):
        frame[f"{prefix}_{phase}_preference"] = frame[col] - row_mean
    return frame


def derive_candidate_preferences(fsr: pd.DataFrame) -> pd.DataFrame:
    """Return one row per fighter-fight with three candidate preference vectors."""
    missing = sorted(set(FSR_COLUMNS) - set(fsr.columns))
    if missing:
        raise ValueError(f"FSR-26 missing phase-preference inputs: {missing}")

    out = _attach_opponent_traits(fsr[FSR_COLUMNS])

    # Candidate A: direct relative phase/activity tendencies.
    out["pressure_only_distance_raw"] = out["distance_striking_pressure"]
    out["pressure_only_clinch_raw"] = out["clinch_striking_pressure"]
    out["pressure_only_wrestling_raw"] = out["wrestling_entry"]
    out = _center_triplet(out, "pressure_only")

    # Candidate B: general positional control contributes to clinch/wrestling
    # persistence/imposition, but not to distance preference.
    out["control_blend_distance_raw"] = out["distance_striking_pressure"]
    out["control_blend_clinch_raw"] = (
        0.70 * out["clinch_striking_pressure"]
        + 0.30 * out["control_imposition"]
    )
    out["control_blend_wrestling_raw"] = (
        0.70 * out["wrestling_entry"]
        + 0.30 * out["control_imposition"]
    )
    out = _center_triplet(out, "control_blend")

    # Candidate C: matchup-specific choice desirability. Opponent resistance is
    # deliberately a modest modifier; execution remains a separate MC layer.
    out["matchup_adjusted_distance_raw"] = out["distance_striking_pressure"]
    out["matchup_adjusted_clinch_raw"] = (
        out["control_blend_clinch_raw"]
        - 0.25 * (out["opponent_control_resistance"] - 50.0)
    )
    out["matchup_adjusted_wrestling_raw"] = (
        out["control_blend_wrestling_raw"]
        - 0.25 * (out["opponent_td_defense"] - 50.0)
    )
    out = _center_triplet(out, "matchup_adjusted")

    keep = [*KEYS, "fighter_name", "opponent_id"]
    keep += [
        f"{candidate}_{phase}_preference"
        for candidate in CANDIDATES
        for phase in PHASES
    ]
    return out[keep].reset_index(drop=True)


def build_research_frame(fsr: pd.DataFrame, rfs: pd.DataFrame) -> pd.DataFrame:
    prefs = derive_candidate_preferences(fsr)
    missing = sorted(set(RFS_COLUMNS) - set(rfs.columns))
    if missing:
        raise ValueError(f"RFS history missing phase-preference outcomes: {missing}")

    outcomes = _normalize_keys(rfs[RFS_COLUMNS])
    if outcomes.duplicated(KEYS).any():
        raise RuntimeError("RFS outcomes violate fighter-fight grain")

    merged = prefs.merge(outcomes, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(prefs):
        raise RuntimeError(
            f"phase-preference key mismatch: preferences={len(prefs):,}, merged={len(merged):,}"
        )

    for col in OUTCOME_COLUMNS.values():
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def _quantile_bucket(series: pd.Series, groups: int = 7) -> pd.Series:
    """Stable equal-count buckets even when a score has duplicate values."""
    rank = series.rank(method="first")
    return pd.qcut(rank, q=groups, labels=[f"Q{i}" for i in range(1, groups + 1)])


def _spearman(frame: pd.DataFrame, score: str, outcome: str) -> float:
    clean = frame[[score, outcome]].dropna()
    if len(clean) < 3:
        return float("nan")
    return float(clean.corr(method="spearman").iloc[0, 1])


def summarize_candidate(frame: pd.DataFrame, candidate: str) -> None:
    print("\n" + "=" * 96)
    print(f"CANDIDATE: {candidate}")
    print("=" * 96)

    mapping = {
        "distance": [OUTCOME_COLUMNS["distance"]],
        "clinch": [OUTCOME_COLUMNS["clinch"]],
        "wrestling": [OUTCOME_COLUMNS["td_rate"], OUTCOME_COLUMNS["control_rate"]],
    }

    for phase in PHASES:
        score = f"{candidate}_{phase}_preference"
        print(f"\n{phase.upper()} preference")
        for outcome in mapping[phase]:
            corr = _spearman(frame, score, outcome)
            print(f"  Spearman vs {outcome}: {corr:.4f}")

        work = frame[[score, *mapping[phase]]].dropna(subset=[score]).copy()
        work["bucket"] = _quantile_bucket(work[score])
        agg_spec: dict[str, tuple[str, str]] = {
            "rows": (score, "size"),
            "mean_preference": (score, "mean"),
        }
        for outcome in mapping[phase]:
            short = next(k for k, v in OUTCOME_COLUMNS.items() if v == outcome)
            agg_spec[f"mean_{short}"] = (outcome, "mean")

        summary = work.groupby("bucket", observed=False).agg(**agg_spec)
        print(summary.to_string())


def main() -> None:
    if not FSR26_PATH.exists():
        raise RuntimeError(f"FSR-26 database not found: {FSR26_PATH}")
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")

    print(f"[phase research] loading FSR-26 from {FSR26_PATH}", flush=True)
    fsr = pd.read_parquet(FSR26_PATH)
    print(f"[phase research] loaded {len(fsr):,} FSR-26 rows", flush=True)

    print(f"[phase research] loading RFS history from {RFS_PATH}", flush=True)
    rfs = pd.read_parquet(RFS_PATH)
    print(f"[phase research] loaded {len(rfs):,} RFS rows", flush=True)

    print("[phase research] deriving candidate phase-preference vectors", flush=True)
    frame = build_research_frame(fsr, rfs)
    print(f"[phase research] research grain: {len(frame):,} fighter-fight rows", flush=True)

    for candidate in CANDIDATES:
        summarize_candidate(frame, candidate)


if __name__ == "__main__":
    main()
