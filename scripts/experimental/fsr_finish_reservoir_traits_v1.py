"""Leakage-safe shadow builders for Damage Reservoir V1 finish traits.

This module implements the two newly researched defensive finish traits:

- ``knockdown_resistance``: resistance to acute knockdown shock.
- ``damage_durability``: tolerance of accumulated damaging exposure and the
  preferred driver of fighter-specific reservoir capacity.

The builder is deliberately shadow-only.  It does not remove or rename the
legacy ``chin_resistance`` / ``damage_resistance`` fields and does not modify
production simulator behavior.

Leakage guarantees
------------------
* Each pre-fight snapshot uses only observations from STRICTLY earlier dates.
* All fights on the same date are snapshotted before any rows from that date
  update fighter histories or population thresholds.
* Population thresholds / peer percentiles are therefore prior-date only.
* Fighters with no prior evidence start at the neutral FSR rating of 50.

The research studies used full-history thresholds/ranks for ontology discovery.
This implementation intentionally replaces those with prior-date equivalents so
it can be evaluated as a genuine pre-fight feature family.
"""

from __future__ import annotations

from collections import defaultdict
from math import exp, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")

SKILLS = (
    "knockdown_resistance",
    "damage_durability",
)

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
CONFIDENCE_FIGHTS = 3.0
KD_HIGH_EXPOSURE_QUANTILE = 0.67
DURABILITY_HIGH_EXPOSURE_QUANTILE = 0.70
MIN_POPULATION_OBSERVATIONS = 100

KD_COL = "rfs_finish_state_fight_knockdowns_absorbed"
SIG_ABS_COL = "rfs_finish_state_fight_sig_strikes_absorbed"
HEAD_ABS_COL = "rfs_finish_state_fight_head_strikes_absorbed"
GROUND_ABS_COL = "rfs_finish_state_fight_ground_strikes_absorbed"
OPP_CTRL_COL = "rfs_finish_state_fight_opponent_control_seconds"
ROUNDS_COL = "rfs_finish_state_fight_rounds_observed"
KO_LOSS_COL = "rfs_finish_state_fight_ko_tko_loss_indicator"

REQUIRED_COLUMNS = {
    "fight_id",
    "fighter_id",
    "date",
    KD_COL,
    SIG_ABS_COL,
    HEAD_ABS_COL,
    GROUND_ABS_COL,
    OPP_CTRL_COL,
    ROUNDS_COL,
    KO_LOSS_COL,
}


def _finite(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if isfinite(out) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rating_from_score(score: float, prior_fights: int) -> float:
    """Shrink one 0-1 evidence score toward neutral and map to 10-90."""
    if prior_fights <= 0 or not isfinite(score):
        return BASE_RATING
    confidence = 1.0 - exp(-float(prior_fights) / CONFIDENCE_FIGHTS)
    shrunk = 0.5 + confidence * (_clip01(score) - 0.5)
    return float(np.clip(MIN_RATING + 80.0 * shrunk, MIN_RATING, MAX_RATING))


def _percentile(value: float, population: list[float]) -> float | None:
    vals = [v for v in population if isfinite(v)]
    if not vals or not isfinite(value):
        return None
    arr = np.asarray(vals, dtype=float)
    return float((np.sum(arr < value) + 0.5 * np.sum(arr == value)) / len(arr))


def _quantile(values: list[float], q: float) -> float | None:
    vals = [v for v in values if isfinite(v)]
    if len(vals) < MIN_POPULATION_OBSERVATIONS:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), q))


def _damage_exposure(row: pd.Series) -> float:
    rounds = max(1.0, _finite(row.get(ROUNDS_COL), 1.0))
    kd = _finite(row.get(KD_COL)) / rounds
    head = _finite(row.get(HEAD_ABS_COL)) / rounds
    ground = _finite(row.get(GROUND_ABS_COL)) / rounds
    opp_ctrl = _finite(row.get(OPP_CTRL_COL)) / (rounds * 60.0)
    return float(np.mean([kd, head, ground, opp_ctrl]))


def _new_state() -> dict[str, float]:
    return {
        "fights": 0.0,
        "kd_absorbed": 0.0,
        "sig_absorbed": 0.0,
        "kd_free_fights": 0.0,
        "kd_high_exposure_fights": 0.0,
        "kd_free_high_exposure": 0.0,
        "dur_high_exposure_fights": 0.0,
        "dur_high_survivals": 0.0,
        "dur_high_exposure_sum": 0.0,
        "dur_high_survived_exposure_sum": 0.0,
        "survived_exposure_sum": 0.0,
        "survived_fights": 0.0,
        "ko_losses": 0.0,
    }


def _kd_raw_components(state: dict[str, float]) -> tuple[float, float, float | None]:
    # Higher is better.  Smoothing prevents tiny histories from creating
    # unbounded KD-per-strike ratios.
    avoidance = -((state["kd_absorbed"] + 0.5) / (state["sig_absorbed"] + 50.0))
    kd_free_rate = state["kd_free_fights"] / state["fights"]
    high_rate = None
    if state["kd_high_exposure_fights"] > 0:
        high_rate = state["kd_free_high_exposure"] / state["kd_high_exposure_fights"]
    return float(avoidance), float(kd_free_rate), None if high_rate is None else float(high_rate)


def _kd_score(
    state: dict[str, float],
    peer_avoidance: list[float],
    peer_free_rate: list[float],
    peer_high_rate: list[float],
) -> float:
    avoidance, free_rate, high_rate = _kd_raw_components(state)
    parts: list[float] = []
    p = _percentile(avoidance, peer_avoidance)
    if p is not None:
        parts.append(p)
    p = _percentile(free_rate, peer_free_rate)
    if p is not None:
        parts.append(p)
    if high_rate is not None:
        p = _percentile(high_rate, peer_high_rate)
        if p is not None:
            parts.append(p)
    return float(np.mean(parts)) if parts else 0.5


def _durability_score(state: dict[str, float], threshold: float | None) -> float:
    parts: list[tuple[float, float]] = []

    if state["dur_high_exposure_fights"] > 0:
        high_survival = state["dur_high_survivals"] / state["dur_high_exposure_fights"]
        parts.append((0.35, high_survival))

        denom = state["dur_high_exposure_sum"]
        if denom > 0:
            sev_survival = state["dur_high_survived_exposure_sum"] / denom
            parts.append((0.30, sev_survival))

    if threshold is not None and threshold > 0 and state["survived_fights"] > 0:
        avg_survived = state["survived_exposure_sum"] / state["survived_fights"]
        punishment_tolerance = np.clip(avg_survived / threshold, 0.0, 2.0) / 2.0
        parts.append((0.20, float(punishment_tolerance)))

    if state["fights"] > 0:
        overall_survival = 1.0 - state["ko_losses"] / state["fights"]
        parts.append((0.15, overall_survival))

    if not parts:
        return 0.5
    weight = sum(w for w, _ in parts)
    return float(sum(w * v for w, v in parts) / weight)


def build_prefight_snapshots(rfs: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe pre-fight snapshots for both reservoir traits."""
    missing = sorted(REQUIRED_COLUMNS - set(rfs.columns))
    if missing:
        raise ValueError(f"RFS missing required reservoir-trait columns: {missing}")

    work = rfs.copy()
    work["fight_id"] = work["fight_id"].astype(str)
    work["fighter_id"] = work["fighter_id"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if work["date"].isna().any():
        raise ValueError("RFS contains invalid dates")
    if work.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("RFS violates fighter-fight grain")

    for col in (KD_COL, SIG_ABS_COL, HEAD_ABS_COL, GROUND_ABS_COL, OPP_CTRL_COL, ROUNDS_COL, KO_LOSS_COL):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["_damage_exposure"] = work.apply(_damage_exposure, axis=1)

    states: dict[str, dict[str, float]] = defaultdict(_new_state)
    prior_sig_exposures: list[float] = []
    prior_damage_exposures: list[float] = []
    output: list[dict[str, Any]] = []

    dates = sorted(work["date"].dropna().unique())
    for date_index, fight_date in enumerate(dates, start=1):
        date_rows = work[work["date"] == fight_date].copy()

        kd_high_threshold = _quantile(prior_sig_exposures, KD_HIGH_EXPOSURE_QUANTILE)
        dur_high_threshold = _quantile(prior_damage_exposures, DURABILITY_HIGH_EXPOSURE_QUANTILE)

        # Build prior-date peer populations once for the entire date.  This is
        # the key same-date simultaneity guarantee for the peer-percentile KD trait.
        peer_avoidance: list[float] = []
        peer_free: list[float] = []
        peer_high: list[float] = []
        for state in states.values():
            if state["fights"] <= 0:
                continue
            avoidance, free_rate, high_rate = _kd_raw_components(state)
            peer_avoidance.append(avoidance)
            peer_free.append(free_rate)
            if high_rate is not None:
                peer_high.append(high_rate)

        # Snapshot every fight on this date BEFORE applying any of today's rows.
        for _, row in date_rows.iterrows():
            fighter_id = str(row["fighter_id"])
            state = states[fighter_id]
            prior_fights = int(state["fights"])

            if prior_fights <= 0:
                kd_score = 0.5
                durability_score = 0.5
            else:
                kd_score = _kd_score(state, peer_avoidance, peer_free, peer_high)
                durability_score = _durability_score(state, dur_high_threshold)

            output.append(
                {
                    "fight_id": str(row["fight_id"]),
                    "fighter_id": fighter_id,
                    "date": pd.Timestamp(fight_date),
                    "knockdown_resistance": _rating_from_score(kd_score, prior_fights),
                    "knockdown_resistance_updates": prior_fights,
                    "damage_durability": _rating_from_score(durability_score, prior_fights),
                    "damage_durability_updates": prior_fights,
                    "knockdown_resistance_evidence_score": float(kd_score * 100.0),
                    "damage_durability_evidence_score": float(durability_score * 100.0),
                }
            )

        # Apply today's observations only after every same-date snapshot exists.
        for _, row in date_rows.iterrows():
            fighter_id = str(row["fighter_id"])
            state = states[fighter_id]
            kd_abs = max(0.0, _finite(row.get(KD_COL)))
            sig_abs = max(0.0, _finite(row.get(SIG_ABS_COL)))
            exposure = max(0.0, _finite(row.get("_damage_exposure")))
            ko_loss = 1.0 if _finite(row.get(KO_LOSS_COL)) >= 0.5 else 0.0

            state["fights"] += 1.0
            state["kd_absorbed"] += kd_abs
            state["sig_absorbed"] += sig_abs
            if kd_abs <= 0:
                state["kd_free_fights"] += 1.0

            if kd_high_threshold is not None and sig_abs >= kd_high_threshold:
                state["kd_high_exposure_fights"] += 1.0
                if kd_abs <= 0:
                    state["kd_free_high_exposure"] += 1.0

            if dur_high_threshold is not None and exposure >= dur_high_threshold:
                state["dur_high_exposure_fights"] += 1.0
                state["dur_high_exposure_sum"] += exposure
                if ko_loss < 0.5:
                    state["dur_high_survivals"] += 1.0
                    state["dur_high_survived_exposure_sum"] += exposure

            if ko_loss < 0.5:
                state["survived_fights"] += 1.0
                state["survived_exposure_sum"] += exposure
            else:
                state["ko_losses"] += 1.0

            prior_sig_exposures.append(sig_abs)
            prior_damage_exposures.append(exposure)

        if date_index % 100 == 0 or date_index == len(dates):
            print(
                f"[reservoir traits] processed {date_index:,}/{len(dates):,} dates; "
                f"snapshots={len(output):,}",
                flush=True,
            )

    snapshots = pd.DataFrame(output)
    if len(snapshots) != len(work):
        raise RuntimeError(
            f"reservoir snapshot row mismatch: expected {len(work):,}, got {len(snapshots):,}"
        )

    ratings = snapshots[list(SKILLS)].apply(pd.to_numeric, errors="coerce")
    if ratings.isna().any().any():
        raise RuntimeError("reservoir traits contain missing ratings")
    if ((ratings < MIN_RATING) | (ratings > MAX_RATING)).any().any():
        raise RuntimeError("reservoir traits contain out-of-range ratings")

    return snapshots.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def main() -> None:
    if not RFS_PATH.exists():
        raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    rfs = pd.read_parquet(RFS_PATH)
    snapshots = build_prefight_snapshots(rfs)
    out = Path(
        "data/simulation/rfs_mc_v2_shared_state/fsr_reservoir_traits_v1/"
        "fsr_reservoir_traits_prefight_snapshots.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(out, index=False)
    print(f"Wrote {len(snapshots):,} reservoir-trait pre-fight rows to {out}", flush=True)


if __name__ == "__main__":
    main()
