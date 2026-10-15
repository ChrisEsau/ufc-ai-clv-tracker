"""Codespaces-friendly historical single-fight EVENT MC sanity runner."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import time

import pandas as pd

from pipeline.common.paths import MASTER_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH

from .calibration import DEFAULT_RESOLVER
from .components.action_rates import FightFlowRateProvider, FSRV2ActionRateProvider
from .components.fsr_v2 import FSRV2FighterInput, FSRV2Matchup
from .components.actions import ActionAttempt, ActionOutcome
from .components.profiles import FighterProfile, MatchupProfiles
from .config import FightConfig
from .engine import SimulationEngine
from .events import ConsequenceEvent, FightFinished, PrimaryEvent, RoundEnded, RoundStarted
from .finishes import FinishOutcome, KOTKOFinishModel
from .flow_stats import FlowStatsSink
from .modifiers import DynamicModifierProvider
from .physiology import ImpactTraumaKnockdownModel, PhysiologyOutcome, PhysiologyTimeAdvanceModel
from .rng import RNGManager
from .sinks import FullTraceEventSink
from .stamina import StaminaModel
from .submission_finishes import SubmissionFinishModel, SubmissionFinishOutcome
from .judging import DeterministicJudgingModel, RoundScore

DEFAULT_SEED = 20260813


def fighter_age_years(dob, event_date) -> float:
    """Return age on fight date; unknown DOB uses neutral age 30."""
    if dob is None or pd.isna(dob):
        return 30.0

    dob = pd.Timestamp(dob)
    event_date = pd.Timestamp(event_date)

    age = (event_date - dob).days / 365.2425

    if age <= 0.0:
        raise ValueError(
            f"invalid DOB/event date: dob={dob}, event_date={event_date}"
        )

    return float(age)


@dataclass(frozen=True)
class HistoricalFight:
    fight_id: str
    date: str
    red_name: str
    blue_name: str
    division: str
    rounds: int
    profiles: MatchupProfiles
    fsr_v2_matchup: FSRV2Matchup | None = None


def fight_from_fsr_v2_rows(
    red_row,
    blue_row,
    *,
    fight_id="fsr-v2-fight",
    date="",
    division="unknown",
    rounds=3,
    red_age_years=30.0,
    blue_age_years=30.0,
) -> HistoricalFight:
    """Create an executable fight from two complete canonical FSR V2 rows."""
    red_input = dict(red_row)
    blue_input = dict(blue_row)

    red_input["age_years"] = float(red_age_years)
    blue_input["age_years"] = float(blue_age_years)

    matchup = FSRV2Matchup(
        FSRV2FighterInput.from_mapping(red_input),
        FSRV2FighterInput.from_mapping(blue_input),
    )
    profiles = matchup.physical_profiles()
    return HistoricalFight(str(fight_id), str(date), matchup.red.fighter_name,
                           matchup.blue.fighter_name, str(division), int(rounds),
                           profiles, matchup)


def resolve_fight(fight_id: str) -> HistoricalFight:
    master = pd.read_parquet(MASTER_PATH)
    match = master[master["fight_id"].astype(str) == str(fight_id)]
    if len(match) != 1:
        raise ValueError(f"fight_id must resolve exactly once: {fight_id!r} ({len(match)} rows)")
    row = match.iloc[0]
    fsr = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH)
    date = pd.Timestamp(row["date"])
    snapshots = fsr[
        (fsr["fight_id"].astype(str) == str(fight_id))
        & (pd.to_datetime(fsr["event_date"]) == date)
    ]
    def exact_fighter(fighter_id: object, corner: str) -> dict:
        matched = snapshots[snapshots["fighter_id"].astype(str) == str(fighter_id)]
        if len(matched) != 1:
            raise ValueError(
                f"canonical FSR V2 must resolve exactly one {corner} row for "
                f"fight={fight_id!r}, date={date.date()}, fighter_id={fighter_id!r}; "
                f"found {len(matched)}"
            )
        return matched.iloc[0].to_dict()
    return fight_from_fsr_v2_rows(
        exact_fighter(row["r_id"], "red"), exact_fighter(row["b_id"], "blue"),
        fight_id=row["fight_id"], date=date.date().isoformat(),
        division=row.get("division", "unknown"), rounds=row.get("total_rounds", 3),
        red_age_years=fighter_age_years(row.get("r_dob"), date),
        blue_age_years=fighter_age_years(row.get("b_dob"), date),
    )


def build_engine(fight: HistoricalFight, seed: int, sink):
    key = fight.division if fight.division in DEFAULT_RESOLVER.weight_classes else None
    calibration = DEFAULT_RESOLVER.for_weight_class(key)
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    rate_provider = (FSRV2ActionRateProvider(fight.fsr_v2_matchup, fight.profiles, stamina,
                                             DynamicModifierProvider(calibration), calibration)
                     if fight.fsr_v2_matchup is not None else
                     FightFlowRateProvider(fight.profiles, stamina, DynamicModifierProvider(calibration), calibration))
    return SimulationEngine(
        FightConfig(fight.rounds),
        rate_provider,
        PhysiologyTimeAdvanceModel(stamina, calibration), RNGManager(seed), sink,
        round_recovery_model=stamina,
        physiology_model=ImpactTraumaKnockdownModel(fight.profiles, calibration),
        finish_model=KOTKOFinishModel(fight.profiles, calibration),
        submission_finish_model=SubmissionFinishModel(fight.profiles, calibration, fight.fsr_v2_matchup),
        judging_model=DeterministicJudgingModel(calibration),
    ), calibration, key


def _aggregate_results(rows):
    """Return deterministic summary data separately from wall-clock reporting."""
    paths = len(rows)
    wins = Counter(result.state.winner or "scheduled_horizon" for result in rows)
    methods = Counter(result.state.finish_method or "scheduled_horizon" for result in rows)
    finish_times = tuple(
        result.state.fight_time_seconds
        for result in rows
        if result.state.finish_method == "KO_TKO"
    )
    finish_rounds = Counter(
        int(max(seconds - 1e-12, 0.0) // 300) + 1 for seconds in finish_times
    )
    knockdowns = tuple(
        sum(item.knockdown for item in result.sink_result["physiology"])
        for result in rows
    )
    side_stats = {}
    for side in ("red", "blue"):
        attempts = Counter()
        outcomes = Counter()
        for result in rows:
            attempts.update(result.sink_result["attempts"][side])
            outcomes.update(result.sink_result["outcomes"][side])
        td_attempts = attempts["takedown"] + attempts["clinch_takedown"]
        td_completions = outcomes["takedown_landed"] + outcomes["clinch_takedown_landed"]
        side_stats[side] = {
            "attempts": dict(attempts),
            "td_attempts": td_attempts,
            "td_completions": td_completions,
            "submission_attempts": attempts["submission_attempt"],
            "ko_wins": sum(result.state.winner == side and result.state.finish_method == "KO_TKO" for result in rows),
            "submission_finishes": sum(result.state.winner == side and result.state.finish_method == "SUB" for result in rows),
            "decision_wins": sum(result.state.winner == side and result.state.finish_method == "DEC" for result in rows),
        }
    return {
        "paths": paths,
        "outcomes": dict(wins),
        "scheduled_horizon": wins["scheduled_horizon"],
        "methods": dict(methods),
        "finish_times": finish_times,
        "finish_rounds": dict(finish_rounds),
        "knockdowns": knockdowns,
        "sides": side_stats,
    }


def run_summary(fight: HistoricalFight, paths: int, seed: int):
    started = time.perf_counter(); rows = []
    for index in range(paths):
        engine, calibration, key = build_engine(fight, seed + index, FlowStatsSink())
        rows.append(engine.run())
    elapsed = time.perf_counter() - started
    summary = _aggregate_results(rows)
    wins = summary["outcomes"]
    finishes = summary["finish_times"]
    kds = summary["knockdowns"]
    print_header(fight, calibration, key, seed, paths)
    print(f"Completed: {len(rows)}/{paths} | runtime={elapsed:.2f}s | {paths/elapsed:.2f} paths/s")
    print("Outcomes:", ", ".join(f"{name}={count} ({count/paths:.1%})" for name, count in sorted(wins.items())))
    print(f"Scheduled horizon={summary['scheduled_horizon']} ({summary['scheduled_horizon']/paths:.1%})")
    print(f"KO_TKO={summary['methods'].get('KO_TKO', 0)} ({summary['methods'].get('KO_TKO', 0)/paths:.1%}) | SUB={summary['methods'].get('SUB', 0)} ({summary['methods'].get('SUB', 0)/paths:.1%}) | DEC={summary['methods'].get('DEC', 0)} ({summary['methods'].get('DEC', 0)/paths:.1%})")
    total_sub_attempts = sum(summary["sides"][side]["submission_attempts"] for side in ("red", "blue"))
    total_sub_finishes = summary["methods"].get("SUB", 0)
    print(f"SUB attempts/path={total_sub_attempts/paths:.3f} | SUB finishes={total_sub_finishes} | P(SUB|attempt)={total_sub_finishes/total_sub_attempts:.1%}" if total_sub_attempts else "SUB attempts/path=0.000 | SUB finishes=0 | P(SUB|attempt)=n/a")
    print(f"Avg KO/TKO finish={sum(finishes)/max(len(finishes),1):.1f}s")
    rounds = ", ".join(f"R{round_no}={count} ({count/paths:.1%})" for round_no, count in sorted(summary["finish_rounds"].items())) or "none"
    print(f"Finish rounds: {rounds}")
    print("KO/TKO wins: " + " | ".join(f"{side}={summary['sides'][side]['ko_wins']} ({summary['sides'][side]['ko_wins']/paths:.1%})" for side in ("red", "blue")))
    print(f"KDs/path={sum(kds)/paths:.3f} | zero={sum(x==0 for x in kds)/paths:.1%} | >=1={sum(x>=1 for x in kds)/paths:.1%} | multi={sum(x>=2 for x in kds)/paths:.1%}")
    for side in ("red", "blue"):
        trauma = sum(getattr(result.state, f"{side}_cumulative_trauma") for result in rows)/paths
        stamina = sum(getattr(result.state, f"{side}_stamina") for result in rows)/paths
        control = sum(result.sink_result["clinch_control_seconds"][side] + result.sink_result["ground_control_seconds"][side] for result in rows)/paths
        print(f"{side.upper()}: final stamina={stamina:.3f} trauma={trauma:.2f} control={control:.1f}s")
        side_summary = summary["sides"][side]
        print(f"  TD attempts/completions={side_summary['td_attempts']/paths:.2f}/{side_summary['td_completions']/paths:.2f} per path | SUB attempts={side_summary['submission_attempts']/paths:.2f} per path")
        print(f"  SUB finishes={side_summary['submission_finishes']} | P(SUB|attempt)={side_summary['submission_finishes']/side_summary['submission_attempts']:.1%}" if side_summary["submission_attempts"] else "  SUB finishes=0 | P(SUB|attempt)=n/a")
        print(f"  DEC wins={side_summary['decision_wins']}")
        families = sorted(set().union(*(result.sink_result["attempts"][side] for result in rows)))
        print(f"  avg attempts: " + ", ".join(f"{family}={sum(result.sink_result['attempts'][side].get(family,0) for result in rows)/paths:.2f}" for family in families))
        landed = sum(sum(count for key, count in result.sink_result["outcomes"][side].items() if key.endswith("_landed")) for result in rows) / paths
        print(f"  avg landed strike/TD outcomes={landed:.2f}")
    print("Avg phase seconds:", " ".join(
        f"{phase}={sum(r.sink_result['phase_seconds'][phase] for r in rows)/paths:.1f}"
        for phase in ("standing", "ground")
    ))
    return summary


def run_trace(fight: HistoricalFight, seed: int):
    sink = FullTraceEventSink(); engine, calibration, key = build_engine(fight, seed, sink)
    print_header(fight, calibration, key, seed, 1)
    for side, profile in (("RED", fight.profiles.red), ("BLUE", fight.profiles.blue)):
        mods = DynamicModifierProvider(calibration).modifiers(profile, __import__("pipeline.simulation.event_mc_v1.state", fromlist=["FightState"]).FightState(), __import__("pipeline.simulation.event_mc_v1.components.profiles", fromlist=["Side"]).Side(side.lower()))
        print(f"{side} profile: power={profile.striking_power:.2f} durability={profile.damage_durability:.2f} KD-res={profile.knockdown_resistance:.2f} stamina-cap={profile.stamina_capacity:.2f} fresh modifiers={mods}")
    result = engine.run()
    print("\nCHRONOLOGICAL TRACE")
    for entry in result.sink_result:
        if entry.kind != "event": continue
        event = entry.payload; stamp = format_time(entry.timestamp_seconds)
        if isinstance(event, (RoundStarted, RoundEnded, FightFinished)):
            print(f"{stamp} {type(event).__name__}: {event}"); continue
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt):
            p=event.payload; print(f"{stamp} ACTION {p.side.value:4} {p.action_family:20} phase={entry.before.phase}->{entry.after.phase} controllers=clinch:{entry.before.clinch_controller}->{entry.after.clinch_controller},ground:{entry.before.ground_controller}->{entry.after.ground_controller} stamina={entry.before.red_stamina:.3f}/{entry.before.blue_stamina:.3f}->{entry.after.red_stamina:.3f}/{entry.after.blue_stamina:.3f} mods={p.dynamic_modifiers}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, ActionOutcome):
            print(f"{stamp} outcome {event.payload.side.value} {event.payload.action_family}={event.payload.outcome} phase={entry.after.phase}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, PhysiologyOutcome):
            p=event.payload; defender=p.defender.value; print(f"{stamp} IMPACT {p.attacker.value}->{defender} impact={p.impact:.3f} primary_trauma={p.primary_trauma:.3f} cumulative_trauma={getattr(entry.after, defender + '_cumulative_trauma'):.3f} acute_vulnerability={getattr(entry.after, defender + '_acute_vulnerability'):.3f} current_KD_resistance={p.current_resistance:.3f} pKD={p.knockdown_probability:.3%} KD={p.knockdown}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, FinishOutcome):
            p=event.payload; print(f"{stamp} FINISH CHECK current_finish_resistance={p.current_finish_resistance:.3f} impact_ratio={p.impact_ratio:.3f} pKO_TKO={p.finish_probability:.3%} finished={p.finished}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, SubmissionFinishOutcome):
            p=event.payload; print(f"{stamp} SUBMISSION CHECK {p.attacker.value}->{p.defender.value} threat={p.threat:.3f} resistance={p.resistance:.3f} position={p.position} stamina/context={p.stamina_context_term:.3f} pSUB={p.finish_probability:.3%} finished={p.finished}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, RoundScore):
            p=event.payload; print(f"{stamp} ROUND {p.round_number} JUDGING RED striking={p.red_effective_striking:.3f} grappling={p.red_effective_grappling:.3f} | BLUE striking={p.blue_effective_striking:.3f} grappling={p.blue_effective_grappling:.3f} | primary diff={p.primary_difference:.3f} criterion={p.criterion} winner={p.winner.value.upper()} score={p.red_score}-{p.blue_score}")
    attempts=Counter(); outcomes=Counter(); kds=Counter()
    for entry in result.sink_result:
        event=entry.payload
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt): attempts[(event.payload.side.value,event.payload.action_family)] += 1
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, ActionOutcome): outcomes[(event.payload.side.value,event.payload.action_family,event.payload.outcome)] += 1
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, PhysiologyOutcome) and event.payload.knockdown: kds[event.payload.attacker.value] += 1
    print(f"\nPATH SUMMARY winner={result.state.winner} method={result.state.finish_method or result.state.finish_reason} time={result.state.fight_time_seconds:.3f}s scheduled_horizon={result.state.finish_reason=='scheduled_horizon'}")
    print("attempts:", ", ".join(f"{side}.{family}={count}" for (side,family),count in sorted(attempts.items())))
    print("landed/outcomes:", ", ".join(f"{side}.{family}.{outcome}={count}" for (side,family,outcome),count in sorted(outcomes.items()) if outcome in {"landed","attempted"}))
    print(f"knockdowns red={kds['red']} blue={kds['blue']} | final stamina={result.state.red_stamina:.3f}/{result.state.blue_stamina:.3f} trauma={result.state.red_cumulative_trauma:.2f}/{result.state.blue_cumulative_trauma:.2f} acute={result.state.red_acute_vulnerability:.3f}/{result.state.blue_acute_vulnerability:.3f}")
    cards = [entry.payload.payload for entry in result.sink_result if entry.kind == "event" and isinstance(entry.payload, ConsequenceEvent) and isinstance(entry.payload.payload, RoundScore)]
    if result.state.finish_method == "DEC":
        print("FINAL DECISION")
        for card in cards: print(f"R{card.round_number} {card.winner.value.upper()} 10-9")
        totals=Counter(card.winner.value for card in cards); print(f"RED rounds={totals['red']} BLUE rounds={totals['blue']} winner={result.state.winner.upper()} method=DEC")


def format_time(seconds):
    round_no=int(seconds//300)+1; elapsed=seconds-(round_no-1)*300
    return f"t={seconds:7.3f} R{round_no} {int(elapsed//60)}:{elapsed%60:05.2f}"


def print_header(fight, calibration, key, seed, paths):
    print(f"EVENT MC V1 | supplied/resolved fight_id={fight.fight_id} | {fight.date}")
    print(f"RED {fight.red_name} vs BLUE {fight.blue_name} | division={fight.division} | rounds={fight.rounds} horizon={fight.rounds*300}s")
    print(f"paths={paths} base_seed={seed} per-path seed=base+index | config_key={key or 'defaults'} | fingerprint={calibration.fingerprint}")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--fight-id", "--bout-id", required=True); parser.add_argument("--paths", type=int, default=1); parser.add_argument("--trace", action="store_true"); parser.add_argument("--seed", type=int, default=DEFAULT_SEED); args=parser.parse_args()
    if args.paths < 1 or (args.trace and args.paths != 1): parser.error("--trace requires --paths 1")
    fight=resolve_fight(args.fight_id)
    run_trace(fight,args.seed) if args.trace else run_summary(fight,args.paths,args.seed)


if __name__ == "__main__": main()
