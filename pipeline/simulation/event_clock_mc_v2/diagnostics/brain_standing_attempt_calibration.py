"""Historical calibration screen for Brain-owned standing attempt cadence.

Research only. Production Brain, FSR, mechanics, submissions, KO/KD and judging
are unchanged.

The intent-rate shadow restored fighter-specific ordering by letting the FSR
matchup-effective standing tendency drive the Brain clock, but the Leavitt-Brito
sanity check produced too many standing attempts.  This diagnostic estimates a
global translation scale empirically instead of tuning that matchup.

For a deterministic recent mature-UFC cohort it compares:

* actual UFCStats distance significant-strike attempts per 15 minutes of total
  fight exposure; and
* simulated STAND_ATTACK attempts per 15 minutes of simulated total fight
  exposure under the intent-rate Brain.

Only the standing-strike rate returned by the research _standing_rates helper is
scaled. TD, clinch, reset, non-standing policy, landing accuracy, damage,
finishes and judging remain frozen. Candidate scales use matched path seeds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import (
    BrainIntentPriors,
    action_probabilities_with_intent_priors,
)
from pipeline.simulation.event_clock_mc_v2.brain.memory import decision_context
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext, sample_next_action_delay
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import (
    EngineConfig,
    EngineFunctions,
    EngineInputs,
    FighterEngineInputs,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.judging import Event2JudgeModel
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    age_years_on_date,
    fighter_mechanics_from_prefight,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import _capabilities
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_intent_rate_shadow import _standing_rates
from pipeline.simulation.event_clock_mc_v2.calibration.config import load_override_file
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION

REFERENCE_CUTOFF = pd.Timestamp("2023-02-04")
TARGET_BEFORE = pd.Timestamp("2026-06-06")
COHORT_FIGHTS = 40
PATHS_PER_FIGHT = 15
SCALES = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.00)
MIN_PRIOR_UFC_FIGHTS = 3
ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
EPS = 1e-12


def _column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"none of {candidates} found; columns={list(frame.columns)}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def actual_distance_attempts(round_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate UFCStats distance significant-strike attempts by bout and side."""
    fight_col = _column(round_stats, ("fight_id", "bout_id"))
    r_sig = _column(round_stats, ("r_sig_str_att", "red_sig_str_att", "r_sig_att"))
    b_sig = _column(round_stats, ("b_sig_str_att", "blue_sig_str_att", "b_sig_att"))
    r_clinch = _column(round_stats, ("r_clinch_att", "red_clinch_att"))
    b_clinch = _column(round_stats, ("b_clinch_att", "blue_clinch_att"))
    r_ground = _column(round_stats, ("r_ground_att", "red_ground_att"))
    b_ground = _column(round_stats, ("b_ground_att", "blue_ground_att"))

    work = pd.DataFrame({
        "fight_id": round_stats[fight_col].astype(str),
        "red_distance_attempts": (
            _numeric(round_stats[r_sig])
            - _numeric(round_stats[r_clinch])
            - _numeric(round_stats[r_ground])
        ).clip(lower=0.0),
        "blue_distance_attempts": (
            _numeric(round_stats[b_sig])
            - _numeric(round_stats[b_clinch])
            - _numeric(round_stats[b_ground])
        ).clip(lower=0.0),
    })
    return work.groupby("fight_id", as_index=False).sum(numeric_only=True)


def elapsed_seconds(row: pd.Series) -> float:
    """Correct UFC elapsed exposure: completed rounds + final-round clock."""
    finish_round = float(row.get("finish_round", np.nan))
    match_time = float(row.get("match_time_sec", np.nan))
    if not np.isfinite(finish_round) or not np.isfinite(match_time):
        return float("nan")
    return max(0.0, (finish_round - 1.0) * 300.0 + match_time)


def prior_counts(master: pd.DataFrame, fight: pd.Series) -> tuple[int, int]:
    date = pd.Timestamp(fight["date"])
    prior = master[pd.to_datetime(master["date"]) < date]
    r_id, b_id = str(fight.r_id), str(fight.b_id)
    red = int(((prior.r_id.astype(str) == r_id) | (prior.b_id.astype(str) == r_id)).sum())
    blue = int(((prior.r_id.astype(str) == b_id) | (prior.b_id.astype(str) == b_id)).sum())
    return red, blue


class ScaledIntentRateBrain:
    """Research intent-rate Brain with one global standing cadence scale."""

    def __init__(self, inputs: EngineInputs, priors: dict[Side, BrainIntentPriors], horizon: float, scale: float):
        self.inputs = inputs
        self.priors = priors
        self.horizon = float(horizon)
        self.scale = float(scale)
        self.side_by_timing_context_id = {
            id(inputs.red.timing_context): Side.RED,
            id(inputs.blue.timing_context): Side.BLUE,
        }

    def _rates(self, state, actor, capabilities, context, config):
        rates, pressure = _standing_rates(
            state, actor, capabilities, context, self.priors[actor], config
        )
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] = max(
            rates[ActionFamily.STAND_ATTACK] * self.scale, EPS
        )
        return rates, pressure

    def timing_sampler(self, state, timing_context, rng, timing_config):
        if state.phase is not Phase.STANDING:
            return sample_next_action_delay(state, timing_context, rng, timing_config)
        side = self.side_by_timing_context_id[id(timing_context)]
        fighter = self.inputs.fighter(side)
        context = decision_context(state, side, fighter.decision_context, self.horizon)
        rates, _ = self._rates(
            state, side, fighter.capabilities, context, self.inputs.policy_config
        )
        mean_delay = 900.0 / max(sum(rates.values()), EPS)
        mean_delay = float(np.clip(
            mean_delay,
            timing_config.minimum_delay_seconds,
            timing_config.maximum_delay_seconds,
        ))
        sampled = rng.gamma(
            shape=timing_config.gamma_shape,
            scale=mean_delay / timing_config.gamma_shape,
        )
        return float(np.clip(
            sampled,
            timing_config.minimum_delay_seconds,
            timing_config.maximum_delay_seconds,
        ))

    def action_chooser(self, state, actor, capabilities, context, rng, config):
        if state.phase is not Phase.STANDING:
            rows = action_probabilities_with_intent_priors(
                state, actor, capabilities, context, self.priors[actor], config
            )
            probs = [row.probability for row in rows]
            return rows[int(rng.choice(len(rows), p=probs))].action_family
        rates, _ = self._rates(state, actor, capabilities, context, config)
        actions = tuple(rates)
        weights = np.asarray([rates[a] for a in actions], dtype=float)
        probs = weights / weights.sum()
        return actions[int(rng.choice(len(actions), p=probs))]


def common_setup():
    snapshots = load_prefight_snapshots()
    snapshots["fight_id"] = snapshots.fight_id.astype(str)
    reference = CapabilityReference.from_prefight_before(snapshots, REFERENCE_CUTOFF)
    mechanics_config, _ = load_override_file(Path("configs/event_clock_v2/calibration/default.yaml"))
    bundle = joblib.load(DEFAULT_BUNDLE_PATH)
    context = bundle["context"]
    submission_offset = float(context["conversion_offset"])

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master.fight_id.astype(str)
    master["r_id"] = master.r_id.astype(str)
    master["b_id"] = master.b_id.astype(str)
    master["date"] = pd.to_datetime(master.date)

    fsr_source = context["fsr_all"].copy()
    valid_source_ids = set(
        fsr_source.groupby(fsr_source.fight_id.astype(str)).size().loc[lambda x: x == 2].index
    )
    source_train = master.assign(event_date=pd.to_datetime(master.date))
    source_train = source_train[
        (source_train.event_date < pd.Timestamp("2025-03-22"))
        & source_train.fight_id.isin(valid_source_ids)
        & source_train.total_rounds.isin([3, 5])
        & source_train.match_time_sec.notna()
    ].sort_values(["event_date", "fight_id"]).tail(3000)
    training_decisions = int(
        source_train.method.fillna("").astype(str).str.lower().str.contains("decision").sum()
    )
    judge_model = Event2JudgeModel.from_sklearn(
        context["judge_model"], training_decisions=training_decisions
    )
    return snapshots, reference, mechanics_config, context, submission_offset, master, judge_model


def build_fight_setup(fight, snapshots, reference, mechanics_config, submission_offset, judge_model):
    date = pd.Timestamp(fight.date)
    red, blue = historical_fighter_rows(
        snapshots,
        event_date=date,
        fight_id=str(fight.fight_id),
        fighter_ids=(str(fight.r_id), str(fight.b_id)),
    )
    rc, rr = _capabilities(red, blue, reference)
    bc, br = _capabilities(blue, red, reference)
    red_mech = fighter_mechanics_from_prefight(
        red,
        rr,
        age_years=age_years_on_date(fight.get("r_dob"), date),
        submission_conversion_offset=submission_offset,
    )
    blue_mech = fighter_mechanics_from_prefight(
        blue,
        br,
        age_years=age_years_on_date(fight.get("b_dob"), date),
        submission_conversion_offset=submission_offset,
    )
    inputs = EngineInputs(
        FighterEngineInputs(rc, BrainTimingContext(), replace(rc and __import__('pipeline.simulation.event_clock_mc_v2.brain.policy', fromlist=['BrainDecisionContext']).BrainDecisionContext()), red_mech),
        FighterEngineInputs(bc, BrainTimingContext(), replace(bc and __import__('pipeline.simulation.event_clock_mc_v2.brain.policy', fromlist=['BrainDecisionContext']).BrainDecisionContext()), blue_mech),
        mechanics_calibration=mechanics_config,
        ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
        judge_model=judge_model,
    )
    priors = {
        Side.RED: BrainIntentPriors(rr.standing_rate_15m, rr.takedown_rate_15m, 0.06, 3.0, 0.3),
        Side.BLUE: BrainIntentPriors(br.standing_rate_15m, br.takedown_rate_15m, 0.06, 3.0, 0.3),
    }
    horizon = float(int(fight.total_rounds) * 300)
    cfg = EngineConfig(number_of_rounds=int(fight.total_rounds))
    return inputs, priors, horizon, cfg


def choose_cohort(master: pd.DataFrame, snapshots: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    valid_snapshot_ids = set(
        snapshots.groupby("fight_id").size().loc[lambda x: x == 2].index.astype(str)
    )
    actual_ids = set(actual.fight_id.astype(str))
    candidates = master[
        (master.date < TARGET_BEFORE)
        & (master.date >= pd.Timestamp("2025-01-01"))
        & master.total_rounds.isin([3, 5])
        & master.match_time_sec.notna()
        & master.fight_id.isin(valid_snapshot_ids)
        & master.fight_id.isin(actual_ids)
    ].sort_values(["date", "fight_id"], ascending=[False, False])

    rows = []
    for _, fight in candidates.iterrows():
        rp, bp = prior_counts(master, fight)
        if rp < MIN_PRIOR_UFC_FIGHTS or bp < MIN_PRIOR_UFC_FIGHTS:
            continue
        elapsed = elapsed_seconds(fight)
        if not np.isfinite(elapsed) or elapsed <= 0:
            continue
        row = fight.copy()
        row["elapsed_seconds"] = elapsed
        row["red_prior"] = rp
        row["blue_prior"] = bp
        rows.append(row)
        if len(rows) >= COHORT_FIGHTS:
            break
    if len(rows) < COHORT_FIGHTS:
        raise RuntimeError(f"only {len(rows)} mature fights available for calibration")
    return pd.DataFrame(rows).sort_values(["date", "fight_id"]).reset_index(drop=True)


def run_scale(scale, cohort, actual_lookup, shared):
    snapshots, reference, mechanics_config, context, submission_offset, master, judge_model = shared
    aggregate_attempts = Counter()
    aggregate_seconds = 0.0
    fighter_rows = []

    for _, fight in cohort.iterrows():
        inputs, priors, horizon, cfg = build_fight_setup(
            fight, snapshots, reference, mechanics_config, submission_offset, judge_model
        )
        sim_attempts = {Side.RED: 0, Side.BLUE: 0}
        sim_seconds = 0.0
        for path_id in range(PATHS_PER_FIGHT):
            brain = ScaledIntentRateBrain(inputs, priors, horizon, scale)
            funcs = EngineFunctions(
                timing_sampler=brain.timing_sampler,
                action_chooser=brain.action_chooser,
            )
            seed = derive_path_seed(SEED_SET_VERSION, str(fight.fight_id), path_id)
            out = run_causal_path(
                inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs
            )
            sim_seconds += float(out.reported_through_seconds)
            for ev in out.events:
                if ev.selected_action is ActionFamily.STAND_ATTACK:
                    sim_attempts[ev.actor] += 1

        actual_row = actual_lookup.loc[str(fight.fight_id)]
        actual_seconds = float(fight.elapsed_seconds)
        for side, prefix in ((Side.RED, "red"), (Side.BLUE, "blue")):
            actual_attempts = float(actual_row[f"{prefix}_distance_attempts"])
            actual_rate = actual_attempts * 900.0 / actual_seconds
            sim_rate = sim_attempts[side] * 900.0 / sim_seconds if sim_seconds else 0.0
            fighter_rows.append({
                "fight_id": str(fight.fight_id),
                "fighter": str(fight.r_name if side is Side.RED else fight.b_name),
                "side": side.value,
                "actual_rate_15m_total_exposure": actual_rate,
                "sim_rate_15m_total_exposure": sim_rate,
                "fsr_effective_standing_rate_15m": priors[side].standing_attempt_rate_15m,
            })
            aggregate_attempts["actual"] += actual_attempts
            aggregate_attempts["sim"] += sim_attempts[side]
        aggregate_seconds += actual_seconds * 2.0

    frame = pd.DataFrame(fighter_rows)
    # For simulated aggregate exposure, each side shares the same path time; use
    # the summed per-fight path exposure twice.
    sim_seconds_total = 0.0
    for _, fight in cohort.iterrows():
        # Recover aggregate simulated seconds from rates/attempts would be lossy;
        # use fighter-level mean rate comparison for selection instead.
        pass

    actual_mean = float(frame.actual_rate_15m_total_exposure.mean())
    sim_mean = float(frame.sim_rate_15m_total_exposure.mean())
    mae = float(np.mean(np.abs(frame.sim_rate_15m_total_exposure - frame.actual_rate_15m_total_exposure)))
    rmse = float(np.sqrt(np.mean((frame.sim_rate_15m_total_exposure - frame.actual_rate_15m_total_exposure) ** 2)))
    corr = float(frame[["sim_rate_15m_total_exposure", "actual_rate_15m_total_exposure"]].corr().iloc[0, 1])
    fsr_corr_actual = float(frame[["fsr_effective_standing_rate_15m", "actual_rate_15m_total_exposure"]].corr().iloc[0, 1])
    return {
        "scale": float(scale),
        "fighters": int(len(frame)),
        "actual_mean_rate_15m_total_exposure": actual_mean,
        "sim_mean_rate_15m_total_exposure": sim_mean,
        "mean_ratio_sim_to_actual": sim_mean / actual_mean if actual_mean else None,
        "mae_rate_15m": mae,
        "rmse_rate_15m": rmse,
        "sim_actual_correlation": corr,
        "fsr_actual_correlation": fsr_corr_actual,
        "fighter_rows": frame.to_dict(orient="records"),
    }


def main():
    if not ROUND_STATS_PATH.exists():
        raise FileNotFoundError(ROUND_STATS_PATH)
    round_stats = pd.read_parquet(ROUND_STATS_PATH)
    actual = actual_distance_attempts(round_stats)
    shared = common_setup()
    cohort = choose_cohort(shared[5], shared[0], actual)
    lookup = actual.set_index("fight_id")

    screens = []
    for scale in SCALES:
        print(f"[attempt calibration] scale={scale:.2f}", flush=True)
        screens.append(run_scale(scale, cohort, lookup, shared))

    best = min(screens, key=lambda row: row["mae_rate_15m"])
    compact = [
        {k: v for k, v in row.items() if k != "fighter_rows"}
        for row in screens
    ]
    payload = {
        "diagnostic": "Brain standing-attempt historical calibration screen",
        "production_changed": False,
        "mechanics_changed": False,
        "judging_changed": False,
        "fsr_changed": False,
        "seed_set": SEED_SET_VERSION,
        "reference_cutoff": str(REFERENCE_CUTOFF.date()),
        "target_before": str(TARGET_BEFORE.date()),
        "cohort_fights": int(len(cohort)),
        "paths_per_fight": PATHS_PER_FIGHT,
        "candidate_scales": list(SCALES),
        "actual_target": "UFCStats distance significant-strike attempts per 15m total fight exposure",
        "simulated_target": "Brain STAND_ATTACK attempts per 15m simulated total fight exposure",
        "cohort": cohort[["fight_id", "date", "r_name", "b_name", "red_prior", "blue_prior", "elapsed_seconds"]].astype({"fight_id": str}).to_dict(orient="records"),
        "screen": compact,
        "best_by_mae": {k: v for k, v in best.items() if k != "fighter_rows"},
        "best_fighter_rows": best["fighter_rows"],
    }
    print("BRAIN_STANDING_ATTEMPT_CALIBRATION")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
