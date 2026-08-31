"""Trace one frozen Event Clock V2 path with Standard Fighter V1 in shadow mode.

The brain never changes the event stream, RNG, mechanics, or result.  It reads
only state that the frozen path exposes plus explicitly labelled coarse rolling
summaries.  Because frozen V1 does not maintain a full phase/cage timeline, the
``shadow_phase`` used here is inferred from the most recent mechanical event and
is diagnostic only.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v1.diagnostics_full_fight_predictive_replay_shadow_ko_kd import (
    add_budget_events,
    add_submission_events,
    apply_delta,
    event_clock_shadow_ko_kd_profiles,
)
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import select_target
from pipeline.simulation.event_mc_v1.calibration import DEFAULT_RESOLVER
from pipeline.simulation.event_mc_v1.components.profiles import Side
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.physiology import PhysiologyTimeAdvanceModel
from pipeline.simulation.event_mc_v1.stamina import StaminaModel
from pipeline.simulation.event_mc_v1.state import FightState as MechanicalFightState

from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_latest_profiles,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.inference import predict_target_v3
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    _draw_budgets,
    _submission_inputs,
    load_frozen_context,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH
from .capability_translation import CapabilityReference, prior_snapshot_count, translate_capability
from .policy import Action, FightState as BrainFightState, Phase as BrainPhase, action_probabilities

DEFAULT_FIGHT_ID = "eb8753ed0476b3d1"  # Aleksandar Rakic vs Marcin Tybura
DEFAULT_SEED = 20260824


def _clip(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _signed_clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _sample_action(probs: dict[Action, float], rng: np.random.Generator) -> Action:
    actions = list(probs)
    p = np.asarray([probs[a] for a in actions], dtype=float)
    return actions[int(rng.choice(len(actions), p=p / p.sum()))]


def _top(probs: dict[Action, float], n: int = 3) -> str:
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return " ".join(f"{a.value}:{p:.0%}" for a, p in ranked)


def _recent_td_signal(td_events, side_name: str, now: float, own: bool) -> tuple[float, float]:
    # Exponentially weighted recent TD outcomes. Diagnostic state translation only.
    total_w = success_w = 0.0
    for t, attacker, landed in td_events:
        if now - t > 120.0:
            continue
        match = attacker == side_name if own else attacker != side_name
        if not match:
            continue
        w = float(np.exp(-(now - t) / 45.0))
        total_w += w
        success_w += w * float(landed)
    if total_w <= 0:
        return 0.0, 0.0
    success = success_w / total_w
    return _clip(success), _clip(1.0 - success)


def _brain_state(
    mech: MechanicalFightState,
    side_name: str,
    shadow_phase: dict[str, BrainPhase],
    recent_landed,
    td_events,
    stats,
    kds,
    horizon: float,
) -> BrainFightState:
    opp = "blue" if side_name == "red" else "red"
    now = float(mech.fight_time_seconds)

    own_recent = sum(1 for t, s in recent_landed if s == side_name and now - t <= 45.0)
    opp_recent = sum(1 for t, s in recent_landed if s == opp and now - t <= 45.0)
    striking_edge = _signed_clip((own_recent - opp_recent) / 4.0)

    own_td_success, own_td_failure = _recent_td_signal(td_events, side_name, now, own=True)
    opp_td_success, opp_td_failure = _recent_td_signal(td_events, side_name, now, own=False)

    own_stamina = float(getattr(mech, f"{side_name}_stamina"))
    own_acute = float(getattr(mech, f"{side_name}_acute_vulnerability"))
    opp_acute = float(getattr(mech, f"{opp}_acute_vulnerability"))
    own_trauma = float(getattr(mech, f"{side_name}_cumulative_trauma"))
    opp_trauma = float(getattr(mech, f"{opp}_cumulative_trauma"))

    own_sig = stats[side_name]["standing_strike_landed"] + stats[side_name]["ground_strike_landed"]
    opp_sig = stats[opp]["standing_strike_landed"] + stats[opp]["ground_strike_landed"]
    score_proxy = (
        (own_sig - opp_sig)
        + 3.0 * (kds[side_name] - kds[opp])
        + 1.5 * (stats[side_name]["td_landed"] - stats[opp]["td_landed"])
    ) / 20.0

    phase = shadow_phase[side_name]
    return BrainFightState(
        phase=phase,
        striking_edge=striking_edge,
        damage_edge=_signed_clip((opp_trauma - own_trauma) / 10.0),
        own_hurt=_clip(own_acute),
        opponent_hurt=_clip(opp_acute),
        td_success_recent=own_td_success,
        td_failure_recent=own_td_failure,
        td_defense_success_recent=opp_td_failure,
        control_success_recent=0.0,
        fatigue=_clip(1.0 - own_stamina),
        score_state=_signed_clip(score_proxy),
        late_fight=_clip(now / max(horizon, 1.0)),
        bad_bottom_position=1.0 if phase == BrainPhase.GROUND_BOTTOM else 0.0,
        dominant_top_position=0.65 if phase == BrainPhase.GROUND_TOP else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fight-id", default=DEFAULT_FIGHT_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH))
    args = parser.parse_args()

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()
    row = master[master["fight_id"] == str(args.fight_id)]
    if len(row) != 1:
        raise RuntimeError(f"expected exactly one fight for {args.fight_id}, found {len(row)}")
    master_row = row.iloc[0]

    context = load_frozen_context(pd.io.common.Path(args.bundle) if False else __import__('pathlib').Path(args.bundle))
    fsr = load_prefight_snapshots()
    latest = load_latest_profiles()
    event_date = pd.Timestamp(master_row["event_date"]).normalize()
    fight_id = str(master_row["fight_id"])

    target = row.copy()
    mean_test, mean_pair = predict_target_v3(
        target, fsr, context["inference_models"], context["submission_scale"], context["conversion_offset"]
    )
    pair_info = mean_pair.iloc[0]
    submission_rates, conversion_probability = _submission_inputs(mean_test)
    budgets = _draw_budgets(mean_test, pair_info, context, np.random.default_rng(args.seed))

    fight = _fight(master_row, context["fsr_all"])
    red_row, blue_row = historical_fighter_rows(
        fsr,
        event_date=event_date,
        fight_id=fight_id,
        fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
    )
    reference = CapabilityReference.from_latest(latest)
    red_prior = prior_snapshot_count(fsr, fighter_id=str(master_row["r_id"]), event_date=event_date)
    blue_prior = prior_snapshot_count(fsr, fighter_id=str(master_row["b_id"]), event_date=event_date)
    red_trans = translate_capability(red_row, blue_row, reference, prior_ufc_fights=red_prior)
    blue_trans = translate_capability(blue_row, red_row, reference, prior_ufc_fights=blue_prior)
    caps = {"red": red_trans.capability, "blue": blue_trans.capability}

    mech_rng = np.random.default_rng(args.seed + DETAILED_PATH_SEED_OFFSET)
    brain_rng = np.random.default_rng(args.seed + 900_000_000)
    horizon = float(fight.rounds * 300)
    key = fight.division if fight.division in DEFAULT_RESOLVER.weight_classes else None
    calibration = DEFAULT_RESOLVER.for_weight_class(key)
    shadow_profiles = event_clock_shadow_ko_kd_profiles(fight)
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    modifiers = DynamicModifierProvider(calibration)
    ko_kd = EventClockShadowKOKDModel(shadow_profiles)
    time_advance = PhysiologyTimeAdvanceModel(stamina, calibration)
    mech = MechanicalFightState()

    events = []
    for side_name in ("red", "blue"):
        side = Side(side_name)
        add_budget_events(events, side, "standing_strike", budgets[f"{side_name}_standing_attempted"], budgets[f"{side_name}_standing_landed"], horizon, mech_rng)
        add_budget_events(events, side, "ground_strike", budgets[f"{side_name}_ground_attempted"], budgets[f"{side_name}_ground_landed"], horizon, mech_rng)
        add_budget_events(events, side, "takedown", budgets[f"{side_name}_td_attempted"], budgets[f"{side_name}_td_landed"], horizon, mech_rng)
        add_submission_events(events, side, submission_rates[side_name], horizon, mech_rng)
    events.sort(key=lambda x: x[0])

    stats = {"red": Counter(), "blue": Counter()}
    kds = {"red": 0, "blue": 0}
    recent_landed = deque()
    td_events = deque()
    shadow_phase = {"red": BrainPhase.STANDING, "blue": BrainPhase.STANDING}
    next_boundary = 300.0

    def advance_to(target_time: float) -> None:
        nonlocal next_boundary
        while next_boundary < target_time and next_boundary < horizon:
            dt = next_boundary - mech.fight_time_seconds
            apply_delta(mech, time_advance.advance(mech, None, dt))
            mech.fight_time_seconds = next_boundary
            apply_delta(mech, stamina.recovery_delta(mech))
            next_boundary += 300.0
        dt = target_time - mech.fight_time_seconds
        if dt > 0:
            apply_delta(mech, time_advance.advance(mech, None, dt))
            mech.fight_time_seconds = target_time

    names = {"red": str(master_row["r_name"]), "blue": str(master_row["b_name"])}
    print("=" * 150)
    print("STANDARD FIGHTER V1 — SINGLE-PATH SHADOW TRACE")
    print("=" * 150)
    print(f"fight: {names['red']} vs {names['blue']} | fight_id={fight_id} | seed={args.seed}")
    print("brain is SHADOW ONLY: separate RNG; cannot change events, mechanics, or outcome")
    print("shadow_phase is inferred from event family because frozen detailed path does not maintain a full phase timeline")
    print(f"RED capability: standing={caps['red'].standing:.3f} TD={caps['red'].takedown:.3f} ground={caps['red'].ground_top:.3f} prior={red_prior}")
    print(f"BLUE capability: standing={caps['blue'].standing:.3f} TD={caps['blue'].takedown:.3f} ground={caps['blue'].ground_top:.3f} prior={blue_prior}")

    # Initial brain read before any event.
    for side_name in ("red", "blue"):
        bs = _brain_state(mech, side_name, shadow_phase, recent_landed, td_events, stats, kds, horizon)
        probs = action_probabilities(bs, caps[side_name])
        chosen = _sample_action(probs, brain_rng)
        print(f"BRAIN t=   0.0 {names[side_name]:20s} phase={bs.phase.value:13s} state=edge {bs.striking_edge:+.2f} hurt {bs.own_hurt:.2f} fat {bs.fatigue:.2f} -> {chosen.value} | {_top(probs)}")

    for event_index, (event_time, side, family, landed) in enumerate(events, start=1):
        if mech.finished:
            break
        advance_to(float(event_time))
        side_name = side.value
        opp = "blue" if side_name == "red" else "red"
        stats[side_name][f"{family}_attempted"] += 1
        profile = fight.profiles.fighter(side)
        modifiers.modifiers(profile, mech, side)
        apply_delta(mech, stamina.action_delta(mech, side, family))

        consequence_text = ""
        meaningful = False
        if family == "submission_attempt":
            meaningful = True
            if mech_rng.random() < conversion_probability:
                mech.finished = True
                mech.finish_reason = mech.finish_method = "SUB"
                mech.winner = side_name
                consequence_text = " -> SUB FINISH"
        elif family == "takedown":
            meaningful = True
            td_events.append((float(event_time), side_name, bool(landed)))
            if landed:
                stats[side_name]["td_landed"] += 1
                shadow_phase[side_name] = BrainPhase.GROUND_TOP
                shadow_phase[opp] = BrainPhase.GROUND_BOTTOM
                consequence_text = " -> LANDED"
            else:
                shadow_phase[side_name] = shadow_phase[opp] = BrainPhase.STANDING
                consequence_text = " -> STUFFED"
        else:
            if family == "standing_strike":
                shadow_phase[side_name] = shadow_phase[opp] = BrainPhase.STANDING
            elif family == "ground_strike":
                shadow_phase[side_name] = BrainPhase.GROUND_TOP
                shadow_phase[opp] = BrainPhase.GROUND_BOTTOM
            if landed:
                meaningful = True
                stats[side_name][f"{family}_landed"] += 1
                recent_landed.append((float(event_time), side_name))
                consequence = ko_kd.resolve_landed_strike(
                    state=mech, attacker=side, prior_defender_kds=kds[side_name], rng=mech_rng
                )
                if consequence.ko_tko:
                    mech.finished = True
                    mech.finish_reason = mech.finish_method = "KO_TKO"
                    mech.winner = side_name
                    consequence_text = f" -> LANDED KO/TKO (pKO={consequence.ko_probability:.3f})"
                elif consequence.knockdown:
                    kds[side_name] += 1
                    consequence_text = f" -> LANDED KD (pKD={consequence.kd_probability:.3f})"
                else:
                    consequence_text = f" -> LANDED (pKD={consequence.kd_probability:.3f})"
            else:
                consequence_text = " -> miss"

        while recent_landed and float(event_time) - recent_landed[0][0] > 120.0:
            recent_landed.popleft()
        while td_events and float(event_time) - td_events[0][0] > 180.0:
            td_events.popleft()

        print(f"EVENT {event_index:03d} t={event_time:6.1f} {names[side_name]:20s} {family:18s}{consequence_text}")
        if meaningful:
            for brain_side in (side_name, opp):
                bs = _brain_state(mech, brain_side, shadow_phase, recent_landed, td_events, stats, kds, horizon)
                probs = action_probabilities(bs, caps[brain_side])
                chosen = _sample_action(probs, brain_rng)
                print(
                    f"BRAIN       t={event_time:6.1f} {names[brain_side]:20s} phase={bs.phase.value:13s} "
                    f"edge={bs.striking_edge:+.2f} hurt={bs.own_hurt:.2f} opphurt={bs.opponent_hurt:.2f} "
                    f"fat={bs.fatigue:.2f} TDok={bs.td_success_recent:.2f} TDfail={bs.td_failure_recent:.2f} "
                    f"-> {chosen.value} | {_top(probs)}"
                )
        if mech.finished:
            break

    if not mech.finished:
        advance_to(horizon)
        red_sig = stats["red"]["standing_strike_landed"] + stats["red"]["ground_strike_landed"]
        blue_sig = stats["blue"]["standing_strike_landed"] + stats["blue"]["ground_strike_landed"]
        decision_row = {
            "sig_diff": red_sig - blue_sig,
            "kd_diff": kds["red"] - kds["blue"],
            "td_diff": stats["red"]["td_landed"] - stats["blue"]["td_landed"],
            "sub_diff": stats["red"]["submission_attempt_attempted"] - stats["blue"]["submission_attempt_attempted"],
            "ctrl_diff": budgets["red_control"] - budgets["blue_control"],
        }
        p_red = float(context["judge_model"].predict_proba(pd.DataFrame([decision_row])[context["judge_features"]])[0, 1])
        mech.finished = True
        mech.finish_reason = mech.finish_method = "DEC"
        mech.winner = "red" if mech_rng.random() < p_red else "blue"
        mech.fight_time_seconds = horizon
        print(f"EVENT END t={horizon:.1f} DECISION p_red={p_red:.3f}")

    print("-" * 150)
    print(f"RESULT: winner={names[mech.winner]} method={mech.finish_method} elapsed={mech.fight_time_seconds:.1f}s")
    print(f"MECHANICS: red_sig={stats['red']['standing_strike_landed'] + stats['red']['ground_strike_landed']} blue_sig={stats['blue']['standing_strike_landed'] + stats['blue']['ground_strike_landed']} red_td={stats['red']['td_landed']} blue_td={stats['blue']['td_landed']} red_kd={kds['red']} blue_kd={kds['blue']}")
    print("SHADOW TRACE COMPLETE — brain had zero causal influence on this path")


if __name__ == "__main__":
    main()
