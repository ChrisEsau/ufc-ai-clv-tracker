"""Research-only prefight hazard bundle for KO V3 from scratch.

This module converts the Stage-2 promoted raw-history models into matchup-time
hazards without reading any FSR physical traits:

* KD hazard per landed significant strike:
  EWM95 attacker KD creation + defender KD susceptibility, strength 200,
  exposure terms, attacker/defender age, division.
* Direct KO/TKO proxy hazard per landed significant strike:
  EWM95 attacker direct-finish + defender direct-finish susceptibility,
  strength 400, exposure terms, attacker/defender age, division.
* Post-KD finishing-sequence hazard per KD:
  age + division only. Fighter-specific conversion/recovery history is excluded
  because it failed the Stage-1b out-of-sample ablation.

Every feature is constructed same-date delayed from raw UFC round statistics.
Models are fit only on rows strictly before the target event date. The module
contains no hurt-state magnitude or decay assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1
from pipeline.research import ko_v3_from_scratch_stage2 as s2

KD_DECAY = 0.95
KD_PRIOR_STRENGTH = 200.0
DIRECT_DECAY = 0.95
DIRECT_PRIOR_STRENGTH = 400.0


@dataclass(frozen=True)
class FighterKOV3Hazards:
    fighter_id: str
    fighter_name: str
    kd_per_landed: float
    direct_finish_per_landed: float
    post_kd_sequence_per_kd: float
    kd_population_hazard: float
    direct_population_hazard: float
    raw_audit: dict

    @property
    def total_finish_per_landed(self) -> float:
        """Sequential direct-or-KD-sequence finish probability per landed strike."""
        p_direct = float(self.direct_finish_per_landed)
        p_kd = float(self.kd_per_landed)
        p_seq = float(self.post_kd_sequence_per_kd)
        return float(p_direct + (1.0 - p_direct) * p_kd * p_seq)


def build_raw_matchup_frame(
    round_path: Path = ROUND_STATS_PATH,
    master_path: Path = MASTER_PATH,
) -> tuple[pd.DataFrame, dict]:
    ff, audit = s1.load_raw_fighter_fights(round_path, master_path)
    states = s1.build_prefight_states(ff)
    return s1.build_matchup_frame(states), audit


def _fit_one_hazard(
    frame: pd.DataFrame,
    target: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
    kind: str,
    kcol: str,
    ncol: str,
    decay: float,
    strength: float,
) -> tuple[np.ndarray, float]:
    train = frame[
        (frame["event_date"] < target_date)
        & frame[ncol].gt(0)
        & frame[kcol].ge(0)
        & frame[kcol].le(frame[ncol])
    ].copy()
    if len(train) < 500:
        raise RuntimeError(f"insufficient pre-event {kind} training rows: {len(train)}")
    total_n = float(train[ncol].sum())
    if total_n <= 0:
        raise RuntimeError(f"no pre-event {kind} opportunities")
    p0 = float(train[kcol].sum() / total_n)

    tr = s2.add_shrunken(
        train,
        kind=kind,
        decay=decay,
        strength=strength,
        p0=p0,
    )
    te = s2.add_shrunken(
        target,
        kind=kind,
        decay=decay,
        strength=strength,
        p0=p0,
    )
    cols = [
        "shr_att",
        "shr_def",
        "attacker_age",
        "defender_age",
        "shr_att_log_exp",
        "shr_def_log_exp",
    ]
    trw, y, weights = s2.weighted_rows(tr, kcol, ncol)
    if np.unique(y).size < 2:
        raise RuntimeError(f"pre-event {kind} training target has one class")
    p, _, _ = s2.fit_logit(
        trw,
        te,
        cols,
        ["division_cat"],
        y,
        weights,
    )
    return np.asarray(p, dtype=float), p0


def _fit_sequence(
    frame: pd.DataFrame,
    target: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
) -> np.ndarray:
    train = frame[
        (frame["event_date"] < target_date)
        & frame["post_kd_opportunity"].gt(0)
        & frame["kd_scored"].gt(0)
    ].copy()
    if len(train) < 200:
        raise RuntimeError(f"insufficient pre-event post-KD training rows: {len(train)}")
    model = s2.fit_sequence_model(train)
    return np.asarray(model.hazard(target), dtype=float)


def fit_prefight_hazards(
    *,
    fight_id: str,
    round_path: Path = ROUND_STATS_PATH,
    master_path: Path = MASTER_PATH,
) -> dict[str, FighterKOV3Hazards]:
    """Fit promoted KO V3 hazards using only information available prefight."""
    frame, audit = build_raw_matchup_frame(round_path, master_path)
    frame = frame.copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    target = frame[frame["fight_id"].eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected two raw matchup rows for fight {fight_id}, found {len(target)}")
    dates = pd.to_datetime(target["event_date"]).dt.normalize().unique()
    if len(dates) != 1:
        raise RuntimeError(f"fight {fight_id} has inconsistent event dates")
    target_date = pd.Timestamp(dates[0]).normalize()

    p_kd, kd_p0 = _fit_one_hazard(
        frame,
        target,
        target_date=target_date,
        kind="kd",
        kcol="kd_scored",
        ncol="sig_landed",
        decay=KD_DECAY,
        strength=KD_PRIOR_STRENGTH,
    )
    p_direct, direct_p0 = _fit_one_hazard(
        frame,
        target,
        target_date=target_date,
        kind="direct",
        kcol="direct_ko_win",
        ncol="sig_landed",
        decay=DIRECT_DECAY,
        strength=DIRECT_PRIOR_STRENGTH,
    )
    p_seq = _fit_sequence(frame, target, target_date=target_date)

    out: dict[str, FighterKOV3Hazards] = {}
    for i, row in enumerate(target.itertuples(index=False)):
        fid = str(row.fighter_id)
        out[fid] = FighterKOV3Hazards(
            fighter_id=fid,
            fighter_name=str(row.fighter_name),
            kd_per_landed=float(p_kd[i]),
            direct_finish_per_landed=float(p_direct[i]),
            post_kd_sequence_per_kd=float(p_seq[i]),
            kd_population_hazard=kd_p0,
            direct_population_hazard=direct_p0,
            raw_audit=dict(audit),
        )
    return out
