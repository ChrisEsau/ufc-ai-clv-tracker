"""Matched-seed Leavitt-Brito dynamic-pressure shadow.

Research only. Production Brain policy/mechanics are unchanged.
The only intervention is standing PRESSURE handling:
- pressure becomes a dynamic context-derived multiplier;
- PRESSURE itself no longer consumes a standing action opportunity;
- dynamic pressure amplifies strike/clinch/TD intent and suppresses reset intent.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
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
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import FinishMethod
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    age_years_on_date,
    fighter_mechanics_from_prefight,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import _capabilities
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_intent_prior_shadow import IntentPriorChooser
from pipeline.simulation.event_clock_mc_v2.calibration.config import load_override_file
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION

FIGHT_ID = "44cfbb8c3c356c65"
PATHS = 500
REFERENCE_CUTOFF = pd.Timestamp("2023-02-04")
STAND = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
GROUND = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}
STRIKES = STAND | GROUND | {ActionFamily.CLINCH_STRIKE}
EPS = 1e-12


def dynamic_pressure(capabilities, context: BrainDecisionContext) -> float:
    """Research-only live pressure state derived from current causal context.

    Baseline style comes from the existing pressure capability. Recent success,
    opponent vulnerability and score urgency can raise it; trouble, hurt,
    fatigue and failed wrestling can lower it. All inputs already exist in the
    production decision context, so this adds no new future information.
    """
    p = float(capabilities.pressure)
    p += 0.35 * max(context.striking_edge, 0.0)
    p += 0.30 * context.opponent_hurt
    p += 0.15 * context.td_defense_success_recent
    p += 0.10 * context.control_success_recent
    p += 0.15 * context.late_urgency * max(-context.score_state, 0.0)
    p -= 0.35 * max(-context.striking_edge, 0.0)
    p -= 0.35 * context.own_hurt
    p -= 0.20 * context.fatigue
    p -= 0.15 * context.td_failure_recent
    return float(np.clip(p, 0.0, 1.0))


def _renormalize(rows, multipliers):
    weights = np.asarray(
        [max(row.probability, EPS) * multipliers.get(row.action_family, 1.0) for row in rows],
        dtype=float,
    )
    weights /= weights.sum()
    return [(row.action_family, float(prob)) for row, prob in zip(rows, weights, strict=True)]


class DynamicPressureChooser:
    """Shadow chooser that changes only standing PRESSURE semantics."""

    def __init__(self, priors: dict[Side, BrainIntentPriors]) -> None:
        self.priors = priors
        self.pressure_sum = defaultdict(float)
        self.pressure_n = Counter()
        self.pressure_min = {Side.RED: 1.0, Side.BLUE: 1.0}
        self.pressure_max = {Side.RED: 0.0, Side.BLUE: 0.0}

    def __call__(self, state, actor, capabilities, context, rng, config):
        rows = action_probabilities_with_intent_priors(
            state, actor, capabilities, context, self.priors[actor], config
        )
        if state.phase is not Phase.STANDING:
            probs = [row.probability for row in rows]
            return rows[int(rng.choice(len(rows), p=probs))].action_family

        p = dynamic_pressure(capabilities, context)
        self.pressure_sum[actor] += p
        self.pressure_n[actor] += 1
        self.pressure_min[actor] = min(self.pressure_min[actor], p)
        self.pressure_max[actor] = max(self.pressure_max[actor], p)

        # Pressure is a live modifier, not a clock-consuming action.
        # Moderate multipliers are intentionally structural, not tuned to this fight.
        strike_mult = 0.75 + 1.00 * p      # 0.75 .. 1.75
        entry_mult = 0.90 + 0.40 * p       # 0.90 .. 1.30
        reset_mult = 1.25 - 0.75 * p       # 1.25 .. 0.50
        multipliers = {
            ActionFamily.STAND_ATTACK: strike_mult,
            ActionFamily.STAND_COUNTER: strike_mult,
            ActionFamily.CLINCH_ENTRY: entry_mult,
            ActionFamily.TAKEDOWN_ENTRY: entry_mult,
            ActionFamily.RESET_RANGE: reset_mult,
            ActionFamily.PRESSURE: EPS,
        }
        adjusted = _renormalize(rows, multipliers)
        probs = [prob for _, prob in adjusted]
        return adjusted[int(rng.choice(len(adjusted), p=probs))][0]

    def summary(self, side: Side) -> dict[str, float]:
        n = self.pressure_n[side]
        return {
            "mean": self.pressure_sum[side] / n if n else 0.0,
            "min": self.pressure_min[side] if n else 0.0,
            "max": self.pressure_max[side] if n else 0.0,
            "standing_decisions": int(n),
        }


def build_setup():
    snapshots = load_prefight_snapshots()
    reference = CapabilityReference.from_prefight_before(snapshots, REFERENCE_CUTOFF)
    mechanics_config, _ = load_override_file(Path("configs/event_clock_v2/calibration/default.yaml"))
    bundle = joblib.load(DEFAULT_BUNDLE_PATH)
    context = bundle["context"]
    submission_offset = float(context["conversion_offset"])

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    master["fight_id"] = master.fight_id.astype(str)
    fight = master.set_index("fight_id").loc[FIGHT_ID]
    date = pd.Timestamp(fight["date"])

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

    red, blue = historical_fighter_rows(
        snapshots,
        event_date=date,
        fight_id=FIGHT_ID,
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
        FighterEngineInputs(rc, BrainTimingContext(), BrainDecisionContext(), red_mech),
        FighterEngineInputs(bc, BrainTimingContext(), BrainDecisionContext(), blue_mech),
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
    return fight, inputs, priors, horizon, cfg


def run_condition(name, fight, inputs, priors, horizon, cfg, shadow: bool):
    chooser = DynamicPressureChooser(priors) if shadow else IntentPriorChooser(priors)
    funcs = EngineFunctions(action_chooser=chooser)
    totals = {s: Counter() for s in Side}
    control = {s: 0.0 for s in Side}
    wins = Counter()
    methods = Counter()
    decision_wins = Counter()

    for path_id in range(PATHS):
        seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, path_id)
        out = run_causal_path(
            inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs
        )
        for seg in out.timeline_segments:
            if seg.phase is Phase.GROUND and seg.controller in (Side.RED, Side.BLUE):
                control[seg.controller] += seg.duration
        for ev in out.events:
            side = ev.actor
            action = ev.selected_action
            totals[side][f"action::{action.value}"] += 1
            if action in TD:
                totals[side]["td_attempts"] += 1
                if ev.resulting_phase is Phase.GROUND and ev.resulting_controller is side:
                    totals[side]["td_success"] += 1
            if action in STRIKES:
                totals[side]["strike_attempts"] += 1
                if ev.outcome.value == "landed":
                    totals[side]["strikes_landed"] += 1
            if action in GROUND:
                totals[side]["ground_strike_attempts"] += 1
                if ev.outcome.value == "landed":
                    totals[side]["ground_strikes_landed"] += 1
            if ev.submission_attempt:
                totals[side]["sub_attempts"] += 1
                if ev.submission_success:
                    totals[side]["sub_success"] += 1
            if ev.knockdown:
                totals[side]["kd"] += 1

        if out.termination is None:
            continue
        winner = out.termination.winner
        method = out.termination.finish_method.value
        wins[winner] += 1
        methods[method] += 1
        if out.termination.finish_method is FinishMethod.DECISION:
            decision_wins[winner] += 1

    def side_summary(side, fighter_name):
        t = totals[side]
        action_counts = {
            key.split("::", 1)[1]: value / PATHS
            for key, value in sorted(t.items())
            if key.startswith("action::")
        }
        row = {
            "fighter": fighter_name,
            "wins": wins[side],
            "win_probability": wins[side] / PATHS,
            "decision_wins": decision_wins[side],
            "td_attempts_per_path": t["td_attempts"] / PATHS,
            "td_success_per_path": t["td_success"] / PATHS,
            "td_success_rate": t["td_success"] / t["td_attempts"] if t["td_attempts"] else 0.0,
            "ground_control_seconds_per_path": control[side] / PATHS,
            "sub_attempts_per_path": t["sub_attempts"] / PATHS,
            "sub_conversion": t["sub_success"] / t["sub_attempts"] if t["sub_attempts"] else 0.0,
            "strike_attempts_per_path": t["strike_attempts"] / PATHS,
            "strikes_landed_per_path": t["strikes_landed"] / PATHS,
            "ground_strike_attempts_per_path": t["ground_strike_attempts"] / PATHS,
            "ground_strikes_landed_per_path": t["ground_strikes_landed"] / PATHS,
            "knockdowns_per_path": t["kd"] / PATHS,
            "actions_per_path": action_counts,
        }
        if shadow:
            row["dynamic_pressure"] = chooser.summary(side)
        return row

    return {
        "condition": name,
        "red": side_summary(Side.RED, str(fight.r_name)),
        "blue": side_summary(Side.BLUE, str(fight.b_name)),
        "methods": dict(methods),
    }


def main():
    fight, inputs, priors, horizon, cfg = build_setup()
    baseline = run_condition("production_intent_policy", fight, inputs, priors, horizon, cfg, False)
    shadow = run_condition("dynamic_pressure_shadow", fight, inputs, priors, horizon, cfg, True)
    payload = {
        "diagnostic": "Leavitt-Brito dynamic pressure matched-seed shadow",
        "fight_id": FIGHT_ID,
        "paths_each": PATHS,
        "seed_set": SEED_SET_VERSION,
        "reference_cutoff": str(REFERENCE_CUTOFF.date()),
        "production_changed": False,
        "intervention": {
            "pressure_is_dynamic": True,
            "pressure_action_consumes_clock": False,
            "counter_logic_changed": False,
            "mechanics_changed": False,
            "timing_changed": False,
        },
        "baseline": baseline,
        "shadow": shadow,
        "delta_blue_brito_win_probability": (
            shadow["blue"]["win_probability"] - baseline["blue"]["win_probability"]
        ),
    }
    print("LEAVITT_BRITO_DYNAMIC_PRESSURE_SHADOW")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
