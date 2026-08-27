"""Schnell-Costa KO V3 shadow with explicit event-driven hurt follow-up state.

Research only. No production files are changed.

Key behavior:
- a KD NEVER finishes on the same strike;
- every KD starts a persistent hurt/follow-up sequence;
- the validated post-KD sequence probability is sampled once when the KD occurs;
- if that sequence is destined to finish, the next landed strike by the KD scorer
  ends the fight by KO/TKO;
- the hurt state is cleared by a clear defensive recovery event (landing back,
  successful clinch/takedown/escape/reversal/disengage, reset range) or a new round;
- Brain action selection sees own_hurt/opponent_hurt=1 while the sequence is active;
- there is no clock half-life, no old FSR power/durability/KD-resistance input,
  and no legacy acute-vulnerability increment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import shutil

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics import ko_kd_empirical as ko_mod
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import ActionOutcome
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action
from pipeline.simulation.event_clock_mc_v2.diagnostics import schnell_costa_ko_v3_from_scratch_shadow as base

RECOVERY_TRANSITION_ACTIONS = frozenset({
    ActionFamily.CLINCH_ENTRY,
    ActionFamily.TAKEDOWN_ENTRY,
    ActionFamily.CLINCH_TAKEDOWN,
    ActionFamily.ESCAPE_STAND,
    ActionFamily.REVERSAL,
    ActionFamily.DISENGAGE,
})
STRIKES = frozenset({
    ActionFamily.STAND_ATTACK,
    ActionFamily.STAND_COUNTER,
    ActionFamily.CLINCH_STRIKE,
    ActionFamily.GROUND_STRIKE,
    ActionFamily.BOTTOM_STRIKE,
})


class HurtFollowupResolver:
    def __init__(self, hazards_by_side):
        self.hazards_by_side = hazards_by_side
        self.active = {side: False for side in Side}
        self.will_finish = {side: False for side in Side}
        self.start_round = {side: 0 for side in Side}
        self.landed = Counter()
        self.direct_finishes = Counter()
        self.knockdowns = Counter()
        self.sequences_started = Counter()
        self.sequence_finish_intents = Counter()
        self.followup_finishes = Counter()
        self.recoveries = Counter()
        self.round_clears = Counter()

    def _clear(self, attacker_side: Side, *, recovery: bool = False, round_clear: bool = False):
        if self.active[attacker_side]:
            self.active[attacker_side] = False
            self.will_finish[attacker_side] = False
            if recovery:
                self.recoveries[attacker_side] += 1
            if round_clear:
                self.round_clears[attacker_side] += 1

    def clear_if_new_round(self, state):
        for side in Side:
            if self.active[side] and state.round_number > self.start_round[side]:
                self._clear(side, round_clear=True)

    def brain_context(self, state, actor, context):
        self.clear_if_new_round(state)
        if self.active[actor]:
            context = replace(context, opponent_hurt=max(context.opponent_hurt, 1.0))
        if self.active[actor.opponent]:
            context = replace(context, own_hurt=max(context.own_hurt, 1.0))
        return context

    def __call__(self, *, state, attacker_side, attacker, defender, rng):
        del attacker, defender
        self.clear_if_new_round(state)
        h = self.hazards_by_side[attacker_side]
        target = state.physiology.fighter(attacker_side.opponent)
        prior = int(target.knockdowns_suffered)
        self.landed[attacker_side] += 1

        # A sampled finishing sequence resolves only on a later landed follow-up.
        if self.active[attacker_side] and self.will_finish[attacker_side]:
            self.followup_finishes[attacker_side] += 1
            self._clear(attacker_side)
            return ko_mod.EmpiricalKOKDResult(
                float(h.post_kd_sequence_per_kd), True, 0.0, False, prior
            )

        p_direct = float(h.direct_finish_per_landed)
        p_kd = float(h.kd_per_landed)
        if bool(rng.random() < p_direct):
            self.direct_finishes[attacker_side] += 1
            self._clear(attacker_side)
            return ko_mod.EmpiricalKOKDResult(p_direct, True, 0.0, False, prior)

        kd = bool(rng.random() < p_kd)
        if not kd:
            return ko_mod.EmpiricalKOKDResult(p_direct, False, p_kd, False, prior)

        self.knockdowns[attacker_side] += 1
        self.sequences_started[attacker_side] += 1
        self.active[attacker_side] = True
        self.start_round[attacker_side] = int(state.round_number)
        self.will_finish[attacker_side] = bool(
            rng.random() < float(h.post_kd_sequence_per_kd)
        )
        if self.will_finish[attacker_side]:
            self.sequence_finish_intents[attacker_side] += 1
        # KD itself never terminates the fight in this architecture.
        return ko_mod.EmpiricalKOKDResult(p_direct, False, p_kd, True, prior)

    def observe_resolution(self, event, resolution):
        """Clear hurt only when the hurt fighter demonstrates a recovery event."""
        hurt_attacker = event.actor.opponent
        if not self.active[hurt_attacker]:
            return
        recovered = False
        if event.action_family in STRIKES and resolution.outcome is ActionOutcome.LANDED:
            recovered = True
        elif event.action_family is ActionFamily.RESET_RANGE:
            recovered = True
        elif event.action_family in RECOVERY_TRANSITION_ACTIONS and resolution.transition is not None:
            recovered = True
        if recovered:
            self._clear(hurt_attacker, recovery=True)

    def summary(self, side):
        return {
            "landed_strike_resolutions": int(self.landed[side]),
            "direct_finishes": int(self.direct_finishes[side]),
            "knockdowns": int(self.knockdowns[side]),
            "hurt_sequences_started": int(self.sequences_started[side]),
            "sampled_finish_sequences": int(self.sequence_finish_intents[side]),
            "followup_finishes": int(self.followup_finishes[side]),
            "defender_recoveries": int(self.recoveries[side]),
            "round_boundary_clears": int(self.round_clears[side]),
        }


def main():
    fight_id = base.resolve_fight_id()
    hazards_by_id = base.fit_prefight_hazards(fight_id=fight_id)

    canonical = base.pd.read_parquet(base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = base.pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    ewm50 = base.build_pure_ewm50_snapshot(canonical)

    shutil.copy2(base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH, base.BACKUP_PATH)
    original_standing_rates = base.intent_mod._standing_rates
    original_empirical_resolver = base.physiology_mod.resolve_empirical_ko_kd
    original_hurt_increment = base.physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    try:
        ewm50.to_parquet(base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        base.pressure_mod.FIGHT_ID = fight_id
        base.pressure_mod.PATHS = base.PATHS
        base.intent_mod.FIGHT_ID = fight_id
        base.intent_mod.PATHS = base.PATHS

        def calibrated_standing_rates(state, actor, capabilities, context, priors, config):
            rates, pressure = original_standing_rates(state, actor, capabilities, context, priors, config)
            rates = dict(rates)
            rates[ActionFamily.STAND_ATTACK] *= base.STANDING_ATTEMPT_SCALE
            return rates, pressure

        base.intent_mod._standing_rates = calibrated_standing_rates
        fight, inputs, priors, horizon, cfg = base.pressure_mod.build_setup()
        side_to_id = {Side.RED: str(fight.r_id), Side.BLUE: str(fight.b_id)}
        hazards_by_side = {side: hazards_by_id[fid] for side, fid in side_to_id.items()}
        resolver = HurtFollowupResolver(hazards_by_side)
        base.physiology_mod.resolve_empirical_ko_kd = resolver
        base.physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0

        brain = base.intent_mod.IntentRateBrain(inputs, priors, horizon)

        def action_chooser(state, actor, capabilities, context, rng, config):
            context = resolver.brain_context(state, actor, context)
            return brain.action_chooser(state, actor, capabilities, context, rng, config)

        def mechanics_resolver(event, state, mechanics_inputs, rng, placeholders, ko_kd_rng, submission_rng):
            resolver.clear_if_new_round(state)
            out = resolve_action(
                event,
                state,
                mechanics_inputs,
                rng,
                placeholders,
                ko_kd_rng,
                submission_rng,
            )
            resolver.observe_resolution(event, out)
            return out

        funcs = EngineFunctions(
            timing_sampler=brain.timing_sampler,
            action_chooser=action_chooser,
            mechanics_resolver=mechanics_resolver,
        )
        names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
        wins = Counter(); sixway = Counter()
        for path_id in range(base.PATHS):
            seed = base.derive_path_seed(base.SEED_SET_VERSION, fight_id, path_id)
            out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
            if out.termination is None:
                continue
            winner = out.termination.winner
            method = out.termination.finish_method.value
            wins[winner] += 1
            sixway[(winner.value, method)] += 1

        fighter_methods = {}
        hazard_audit = {}
        for side in Side:
            h = hazards_by_side[side]
            counts = {m: int(sixway[(side.value, m)]) for m in ("ko_tko", "submission", "decision")}
            fighter_methods[names[side]] = {
                "wins": int(wins[side]),
                "win_probability": wins[side] / base.PATHS,
                "ko_tko": counts["ko_tko"] / base.PATHS,
                "submission": counts["submission"] / base.PATHS,
                "decision": counts["decision"] / base.PATHS,
                "counts": counts,
            }
            hazard_audit[names[side]] = {
                "fighter_id": h.fighter_id,
                "kd_per_landed": h.kd_per_landed,
                "direct_finish_per_landed": h.direct_finish_per_landed,
                "post_kd_sequence_per_kd": h.post_kd_sequence_per_kd,
                "resolver_counts": resolver.summary(side),
            }

        payload = {
            "diagnostic": "Schnell-Costa KO V3 persistent hurt follow-up shadow",
            "fight_id": fight_id,
            "paths": base.PATHS,
            "production_changed": False,
            "ko_v3_uses_fsr_physical_traits": False,
            "old_acute_hurt_disabled": True,
            "clock_hurt_decay_used": False,
            "hurt_state_model": "event-driven until recovery/reset/new round",
            "kd_can_finish_same_strike": False,
            "brain_action_selection_sees_hurt": True,
            "brain_timing_cadence_changed_by_hurt": False,
            "hazard_audit": hazard_audit,
            "fighter_methods": fighter_methods,
        }
        print("SCHNELL_COSTA_KO_V3_HURT_FOLLOWUP_SHADOW")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        base.physiology_mod.resolve_empirical_ko_kd = original_empirical_resolver
        base.physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt_increment
        base.intent_mod._standing_rates = original_standing_rates
        shutil.move(base.BACKUP_PATH, base.FSR_V3_PREFIGHT_SNAPSHOTS_PATH)


if __name__ == "__main__":
    main()
