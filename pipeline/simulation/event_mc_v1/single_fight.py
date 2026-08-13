"""Codespaces-friendly historical single-fight EVENT MC sanity runner."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import time

import pandas as pd

from .calibration import DEFAULT_RESOLVER
from .components.action_rates import FightFlowRateProvider
from .components.actions import ActionAttempt, ActionOutcome
from .components.profiles import FighterProfile, MatchupProfiles
from .config import FightConfig
from .diagnostics.distance_parity import FSR_32_PATH
from .engine import SimulationEngine
from .events import ConsequenceEvent, FightFinished, PrimaryEvent, RoundEnded, RoundStarted
from .finishes import FinishOutcome, KOTKOFinishModel
from .flow_stats import FlowStatsSink
from .modifiers import DynamicModifierProvider
from .physiology import ImpactTraumaKnockdownModel, PhysiologyOutcome, PhysiologyTimeAdvanceModel
from .rng import RNGManager
from .sinks import FullTraceEventSink
from .stamina import StaminaModel

MASTER_PATH = "data/master/ufc_master.parquet"
DEFAULT_SEED = 20260813


@dataclass(frozen=True)
class HistoricalFight:
    fight_id: str
    date: str
    red_name: str
    blue_name: str
    division: str
    rounds: int
    profiles: MatchupProfiles


def resolve_fight(fight_id: str) -> HistoricalFight:
    master = pd.read_parquet(MASTER_PATH)
    match = master[master["fight_id"].astype(str) == str(fight_id)]
    if len(match) != 1:
        raise ValueError(f"fight_id must resolve exactly once: {fight_id!r} ({len(match)} rows)")
    row = match.iloc[0]
    fsr = pd.read_parquet(FSR_32_PATH)
    date = pd.Timestamp(row["date"])
    snapshots = fsr[(fsr["fight_id"].astype(str) == str(fight_id)) & (pd.to_datetime(fsr["date"]) == date)]
    by_name = snapshots.set_index("fighter_name")
    if row["r_name"] not in by_name.index or row["b_name"] not in by_name.index:
        raise ValueError("frozen FSR-32 lacks both prefight profiles for requested fight")
    def profile(name):
        values = by_name.loc[name].to_dict(); values["fighter_name"] = name
        return FighterProfile.from_mapping(values)
    return HistoricalFight(str(row["fight_id"]), date.date().isoformat(), str(row["r_name"]), str(row["b_name"]), str(row.get("division", "unknown")), int(row.get("total_rounds", 3)), MatchupProfiles(profile(row["r_name"]), profile(row["b_name"])))


def build_engine(fight: HistoricalFight, seed: int, sink):
    key = fight.division if fight.division in DEFAULT_RESOLVER.weight_classes else None
    calibration = DEFAULT_RESOLVER.for_weight_class(key)
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    return SimulationEngine(
        FightConfig(fight.rounds),
        FightFlowRateProvider(fight.profiles, stamina, DynamicModifierProvider(calibration), calibration),
        PhysiologyTimeAdvanceModel(stamina, calibration), RNGManager(seed), sink,
        round_recovery_model=stamina,
        physiology_model=ImpactTraumaKnockdownModel(fight.profiles, calibration),
        finish_model=KOTKOFinishModel(fight.profiles, calibration),
    ), calibration, key


def run_summary(fight: HistoricalFight, paths: int, seed: int):
    started = time.perf_counter(); rows = []
    for index in range(paths):
        engine, calibration, key = build_engine(fight, seed + index, FlowStatsSink())
        rows.append(engine.run())
    elapsed = time.perf_counter() - started
    wins = Counter(result.state.winner or "scheduled_horizon" for result in rows)
    finishes = [result.state.fight_time_seconds for result in rows if result.state.finish_method == "KO_TKO"]
    kds = [sum(item.knockdown for item in result.sink_result["physiology"]) for result in rows]
    print_header(fight, calibration, key, seed, paths)
    print(f"Completed: {len(rows)}/{paths} | runtime={elapsed:.2f}s | {paths/elapsed:.2f} paths/s")
    print("Outcomes:", ", ".join(f"{name}={count} ({count/paths:.1%})" for name, count in sorted(wins.items())))
    print(f"KO_TKO={len(finishes)} ({len(finishes)/paths:.1%}) | avg finish={sum(finishes)/max(len(finishes),1):.1f}s")
    print(f"KDs/path={sum(kds)/paths:.3f} | zero={sum(x==0 for x in kds)/paths:.1%} | >=1={sum(x>=1 for x in kds)/paths:.1%} | multi={sum(x>=2 for x in kds)/paths:.1%}")
    for side in ("red", "blue"):
        trauma = sum(getattr(result.state, f"{side}_cumulative_trauma") for result in rows)/paths
        stamina = sum(getattr(result.state, f"{side}_stamina") for result in rows)/paths
        control = sum(result.sink_result["clinch_control_seconds"][side] + result.sink_result["ground_control_seconds"][side] for result in rows)/paths
        print(f"{side.upper()}: final stamina={stamina:.3f} trauma={trauma:.2f} control={control:.1f}s")
        families = sorted(set().union(*(result.sink_result["attempts"][side] for result in rows)))
        print(f"  avg attempts: " + ", ".join(f"{family}={sum(result.sink_result['attempts'][side].get(family,0) for result in rows)/paths:.2f}" for family in families))
        landed = sum(sum(count for key, count in result.sink_result["outcomes"][side].items() if key.endswith("_landed")) for result in rows) / paths
        print(f"  avg landed strike/TD outcomes={landed:.2f}")
    print("Avg phase seconds:", " ".join(f"{phase}={sum(r.sink_result['phase_seconds'][phase] for r in rows)/paths:.1f}" for phase in ("distance","clinch","ground")))
    return {"outcomes": dict(wins), "knockdowns": tuple(kds), "finish_times": tuple(finishes)}


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
            p=event.payload; print(f"{stamp} {entry.before.phase:8} {p.side.value:4} {p.action_family:20} stamina {entry.before.red_stamina:.3f}/{entry.before.blue_stamina:.3f}->{entry.after.red_stamina:.3f}/{entry.after.blue_stamina:.3f} mods={p.dynamic_modifiers}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, ActionOutcome):
            print(f"{stamp} outcome {event.payload.side.value} {event.payload.action_family}={event.payload.outcome} phase={entry.after.phase}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, PhysiologyOutcome):
            p=event.payload; print(f"{stamp} IMPACT {p.attacker.value}->{p.defender.value} impact={p.impact:.3f} trauma={p.primary_trauma:.3f} KDres={p.current_resistance:.3f} pKD={p.knockdown_probability:.3%} KD={p.knockdown}")
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, FinishOutcome):
            p=event.payload; print(f"{stamp} FINISH CHECK ratio={p.impact_ratio:.3f} resistance={p.current_finish_resistance:.3f} p={p.finish_probability:.3%} finished={p.finished}")
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
