"""Single approved fight-agnostic locked Brain MC research harness.

ALL Brain MC research runs MUST execute through this module unless the user
explicitly approves an exception. Do not create or execute one-off runners,
ad-hoc fight scripts, alternate harnesses, wrapper runners, or workflow
monkeypatch runners.

Brain event generation uses one global 1-second availability clock. Rate controls
action availability once; simultaneous available actions are resolved by a neutral
uniform collision tie-break. The one-path diagnostic emits both selected-event
reports and every-second tick reports, then reconciles them to the frozen
independent KO/TKO survival clock so the reported termination is the actual final
MC outcome rather than the underlying pre-KO Brain path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.research import fsr_recency_cohort_shadow as recency
from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_time_ko_clock_2000 as time_ko
from pipeline.research import allen_shahbazyan_time_ko_validated_kd_2000 as validated_kd
from pipeline.research import allen_shahbazyan_decision_scored_outputs_2000 as scored
from pipeline.research import locked_brain_tick_clock as tick_clock
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

APPROVED_ENTRY_POINT = "pipeline.research.locked_brain_mc"
RUN_POLICY = (
    "ALL Brain MC research runs must execute through this harness; no one-off, "
    "alternate, wrapper, or fight-specific runners unless explicitly approved "
    "by the user."
)
LOCKED_BASE_COMMIT = "6ba1dd2d1e82aa7dec643bfd6f1d56bdd61b4e92"
LOCKED_PATHS = 500
LOCKED_EWM_DECAY = 0.50
LOCKED_EWM_CANONICAL_BLEND = 0.0
LOCKED_STANDING_ATTEMPT_SCALE = 0.25
EPS = 1e-12
CANONICAL_ARTIFACT_ID = 9494902022
CANONICAL_SOURCE_RUN_ID = 32645607979
CANONICAL_ARTIFACT_DIGEST = "sha256:a6abd062322eaf0c4a47997f215d7fa82c01c7db2755089fad1a30420da7d639"
OUTROOT = Path("data/research/locked_brain_mc")

LOCKED_BLOBS = {
    "pipeline/research/fsr_recency_cohort_shadow.py": "f126d734d346e674d06d4ac711fc2d776d242e73",
    "pipeline/research/allen_shahbazyan_new_timing_trace.py": "d9ac0a4da67ec0d83222b91337f6b4f396f262e7",
    "pipeline/research/allen_shahbazyan_time_ko_clock_2000.py": "edc24423d5549e0e706e17ee14459ec187a52542",
    "pipeline/research/allen_shahbazyan_time_ko_validated_kd_2000.py": "a333689887bb9a54cadfff2a9ac05a70ee844f64",
    "pipeline/research/allen_shahbazyan_decision_scored_outputs_2000.py": "6c23e1765b941c082c535588bf7a41b53fc6516d",
}

LOCKED_CORE_PATHS = (
    "pipeline/simulation/event_clock_mc_v2",
    "pipeline/fsr_v2",
    "pipeline/fsr_v3",
    "configs/event_clock_v2",
    "pipeline/research/ko_time_survival_oos.py",
    "pipeline/research/ko_v3_from_scratch_shadow.py",
    "pipeline/research/allen_shahbazyan_ground_opportunity_submission_trace.py",
    "pipeline/research/allen_shahbazyan_fighter_level_submission_trace.py",
    "pipeline/research/allen_shahbazyan_one_path_brain_trace_v1.py",
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the single approved fight-agnostic locked Brain MC harness"
    )
    parser.add_argument("--fight-id", required=True)
    parser.add_argument("--paths", type=int, default=LOCKED_PATHS)
    args = parser.parse_args(argv)
    args.fight_id = str(args.fight_id).strip()
    if not args.fight_id:
        parser.error("--fight-id must be non-empty")
    if args.paths < 1:
        parser.error("--paths must be >= 1")
    return args


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _enum(value):
    return None if value is None else getattr(value, "value", str(value))


def _set_fight_id(fight_id: str) -> None:
    validated_kd.base_trace.FIGHT_ID = fight_id
    time_ko.base_trace.FIGHT_ID = fight_id
    scored.base_trace.FIGHT_ID = fight_id
    timing.base_trace.FIGHT_ID = fight_id
    timing.target.FIGHT_ID = fight_id
    scored.pressure_mod.FIGHT_ID = fight_id


def assert_locked_sources() -> dict:
    failures: list[str] = []
    blobs: dict[str, str] = {}
    for path, expected in LOCKED_BLOBS.items():
        actual = _git("hash-object", path)
        blobs[path] = actual
        if actual != expected:
            failures.append(f"blob drift: {path}: expected {expected}, got {actual}")
    drift = subprocess.run(
        ["git", "diff", "--quiet", LOCKED_BASE_COMMIT, "--", *LOCKED_CORE_PATHS],
        check=False,
    ).returncode
    if drift != 0:
        changed = _git("diff", "--name-only", LOCKED_BASE_COMMIT, "--", *LOCKED_CORE_PATHS)
        failures.append("core source drift from locked base commit:\n" + changed)
    if failures:
        raise RuntimeError("LOCKED BRAIN HARNESS REFUSED TO RUN\n" + "\n".join(failures))
    return blobs


def locked_standing_rates(state, actor, capabilities, context, priors, config):
    """Single source of truth for all locked standing intent rates."""
    del state, capabilities, context, config
    return {
        ActionFamily.STAND_ATTACK: max(
            float(priors.standing_attempt_rate_15m) * LOCKED_STANDING_ATTEMPT_SCALE,
            EPS,
        ),
        ActionFamily.TAKEDOWN_ENTRY: max(
            float(priors.takedown_attempt_rate_15m) * float(timing.TD_SCALE),
            EPS,
        ),
        ActionFamily.CLINCH_ENTRY: max(float(timing.CLINCH_RATE_BY_SIDE[actor]), EPS),
    }, 0.0


def _primary_mechanic_probability(event, state, inputs, placeholders):
    fighter = inputs.fighter(event.actor)
    family = event.action_family
    if family in {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}:
        return {"mechanic": "standing_strike_landing", "probability": float(fighter.standing_strike_landing_probability)}
    if family is ActionFamily.CLINCH_ENTRY:
        return {"mechanic": "clinch_entry_success", "probability": float(placeholders.clinch_entry_success_probability)}
    if family in {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}:
        return {"mechanic": "takedown_completion", "probability": float(fighter.takedown_completion_probability)}
    if family is ActionFamily.CLINCH_STRIKE:
        return {"mechanic": "clinch_strike_landing", "probability": float(placeholders.clinch_strike_landing_probability)}
    if family is ActionFamily.BREAK_CLINCH:
        return {"mechanic": "break_clinch_success", "probability": float(placeholders.break_clinch_success_probability)}
    if family in {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}:
        return {"mechanic": "ground_strike_landing", "probability": float(fighter.ground_strike_landing_probability)}
    if family is ActionFamily.REVERSAL:
        return {"mechanic": "ground_reversal", "probability": float(fighter.ground_reversal_probability)}
    if family is ActionFamily.ESCAPE_STAND:
        return {"mechanic": "rate_driven_escape_event", "probability": 1.0}
    if family is ActionFamily.SUBMISSION_ATTACK:
        return {"mechanic": "submission_conversion", "probability": None}
    return {"mechanic": "deterministic_or_tactical", "probability": 1.0}


def _event_report_observer(original_run):
    captured = {}

    def observed_run(inputs, *, seed, horizon_seconds, initial_state=None, config=None, functions=None):
        pre_mechanics = []
        original_functions = functions or EngineFunctions()
        original_resolver = original_functions.mechanics_resolver

        def observed_resolver(event, state, mechanics_inputs, rng, placeholders, ko_kd_rng=None, submission_rng=None):
            pre_mechanics.append({
                "timestamp": float(event.timestamp_seconds),
                "actor": event.actor.value,
                "action": event.action_family.value,
                **_primary_mechanic_probability(event, state, mechanics_inputs, placeholders),
            })
            return original_resolver(
                event, state, mechanics_inputs, rng, placeholders, ko_kd_rng, submission_rng
            )

        # Preserve escape-rate introspection through the observer wrapper.
        if hasattr(original_resolver, "escape_mean_seconds"):
            observed_resolver.escape_mean_seconds = original_resolver.escape_mean_seconds
        if hasattr(original_resolver, "escape_checks"):
            observed_resolver.escape_checks = original_resolver.escape_checks

        observed_functions = EngineFunctions(
            timing_sampler=original_functions.timing_sampler,
            action_chooser=original_functions.action_chooser,
            mechanics_resolver=observed_resolver,
        )
        kwargs = {"seed": seed, "horizon_seconds": horizon_seconds, "functions": observed_functions}
        if initial_state is not None:
            kwargs["initial_state"] = initial_state
        if config is not None:
            kwargs["config"] = config
        out = original_run(inputs, **kwargs)

        brain = getattr(original_functions.action_chooser, "__self__", None)
        decisions = list(getattr(brain, "decisions", []))
        ticks = list(getattr(brain, "tick_trace", []))
        if len(decisions) != len(out.events):
            raise RuntimeError(f"event-report decision/event mismatch: {len(decisions)} != {len(out.events)}")
        if len(pre_mechanics) != len(out.events):
            raise RuntimeError(f"event-report mechanics/event mismatch: {len(pre_mechanics)} != {len(out.events)}")

        escape_checks = list(getattr(original_resolver, "escape_checks", []))
        escape_by_key = {
            (round(float(x["timestamp"]), 9), str(x["actor"])): x for x in escape_checks
        }
        rows = []
        previous_timestamp = 0.0
        for index, (decision, event, mechanic) in enumerate(
            zip(decisions, out.events, pre_mechanics, strict=True)
        ):
            key = (round(float(event.timestamp_seconds), 9), event.actor.value)
            primary_probability = (
                float(event.submission_probability)
                if event.selected_action is ActionFamily.SUBMISSION_ATTACK
                else mechanic["probability"]
            )
            rows.append({
                "event_index": index,
                "event_timestamp": float(event.timestamp_seconds),
                "seconds_since_prior_event": float(event.timestamp_seconds) - previous_timestamp,
                "round": int(decision["round"]),
                "phase": decision["phase"],
                "actor": event.actor.value,
                "selected_action": event.selected_action.value,
                "brain_options": decision["brain_options"],
                "brain_selected_probability": next(
                    (
                        float(x["probability"])
                        for x in decision["brain_options"]
                        if x["action"] == event.selected_action.value
                        and x.get("actor", event.actor.value) == event.actor.value
                    ),
                    None,
                ),
                "collision_rule": decision.get("collision_rule"),
                "dynamic_pressure": decision.get("dynamic_pressure"),
                "mechanic": mechanic["mechanic"],
                "mechanic_probability": primary_probability,
                "outcome": event.outcome.value,
                "transition_kind": _enum(event.transition_kind),
                "resulting_phase": event.resulting_phase.value,
                "resulting_controller": _enum(event.resulting_controller),
                "escape_model": escape_by_key.get(key),
                "impact": float(event.impact),
                "kd_probability": float(event.kd_probability),
                "knockdown": bool(event.knockdown),
                "strike_level_ko_probability": float(event.ko_probability),
                "strike_level_ko_tko": bool(event.ko_tko),
                "submission_attempt": bool(event.submission_attempt),
                "submission_conversion_probability": float(event.submission_probability),
                "submission_success": bool(event.submission_success),
            })
            previous_timestamp = float(event.timestamp_seconds)

        captured["seed"] = int(seed)
        captured["brain_reported_through_seconds"] = float(out.reported_through_seconds)
        captured["brain_termination"] = (
            None
            if out.termination is None
            else {"winner_side": out.termination.winner.value, "method": out.termination.finish_method.value}
        )
        captured["reported_through_seconds"] = float(out.reported_through_seconds)
        captured["termination"] = captured["brain_termination"]
        captured["events"] = rows
        captured["ticks"] = ticks
        return out

    return observed_run, captured


def _ko_piece_index(timestamp: float, pieces: int) -> int:
    # Tick t represents the interval (t-1, t], so t=300 uses the first-round hazard.
    return min(max(int(math.ceil(max(float(timestamp), EPS) / 300.0)) - 1, 0), pieces - 1)


def _reconcile_one_path_finish(captured, outdir: Path, fight) -> None:
    """Attach the exact frozen KO clock draw and actual final MC termination."""
    path_file = outdir / "run" / "sim" / "paths.csv"
    if not path_file.is_file():
        raise RuntimeError(f"missing one-path KO result: {path_file}")
    paths = pd.read_csv(path_file)
    if len(paths) != 1:
        raise RuntimeError(f"expected one KO path row, found {len(paths)}")
    row = paths.iloc[0]

    names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
    winner_name = str(row["winner"])
    winner_side = next((side for side, name in names.items() if name == winner_name), None)
    if winner_side is None:
        raise RuntimeError(f"could not map final winner {winner_name!r} to fight sides")

    _, _, _, clock = time_ko._time_clock_inputs()
    hazards = {
        side: np.asarray(clock[names[side]]["hazards_per_second"], dtype=float)
        for side in Side
    }
    sampled_times = {
        Side.RED: None if pd.isna(row.get("allen_clock_time")) else float(row["allen_clock_time"]),
        Side.BLUE: None if pd.isna(row.get("shahbazyan_clock_time")) else float(row["shahbazyan_clock_time"]),
    }

    final_time = float(row["end_seconds"])
    final_method = str(row["method"])
    clock_triggered = bool(row["clock_triggered"])

    reconciled_ticks = []
    for tick in captured.get("ticks", []):
        timestamp = float(tick["timestamp"])
        if timestamp >= final_time - 1e-12:
            break
        idx = _ko_piece_index(timestamp, len(hazards[Side.RED]))
        tick = dict(tick)
        tick["ko"] = {
            side.value: {
                "fighter": names[side],
                "hazard_per_second": float(hazards[side][idx]),
                "probability_next_1s": float(1.0 - math.exp(-float(hazards[side][idx]))),
                "sampled_clock_time": sampled_times[side],
                "fires_in_this_tick_interval": (
                    sampled_times[side] is not None
                    and timestamp - 1.0 < sampled_times[side] <= timestamp
                ),
            }
            for side in Side
        }
        reconciled_ticks.append(tick)

    captured["events"] = [
        event for event in captured.get("events", [])
        if float(event["event_timestamp"]) < final_time - 1e-12
    ]

    if final_method == "ko_tko" and clock_triggered:
        idx = _ko_piece_index(final_time, len(hazards[Side.RED]))
        prior = reconciled_ticks[-1] if reconciled_ticks else {}
        reconciled_ticks.append({
            "tick": None,
            "timestamp": final_time,
            "round": int(math.ceil(final_time / 300.0)),
            "phase": prior.get("phase"),
            "ground_controller": prior.get("ground_controller"),
            "clinch_controller": prior.get("clinch_controller"),
            "options": [],
            "available_count": 0,
            "collision": False,
            "collision_rule": "independent_continuous_ko_clock",
            "selected_actor": winner_side.value,
            "selected_action": "ko_clock",
            "selected_probability_given_available": 1.0,
            "ko_clock_event": True,
            "ko": {
                side.value: {
                    "fighter": names[side],
                    "hazard_per_second": float(hazards[side][idx]),
                    "probability_next_1s": float(1.0 - math.exp(-float(hazards[side][idx]))),
                    "sampled_clock_time": sampled_times[side],
                    "fires_in_this_tick_interval": sampled_times[side] == final_time,
                }
                for side in Side
            },
        })

    captured["ticks"] = reconciled_ticks
    captured["reported_through_seconds"] = final_time
    captured["termination"] = {
        "winner_side": winner_side.value,
        "winner_name": winner_name,
        "method": final_method,
        "end_seconds": final_time,
    }
    captured["ko_clock"] = {
        "architecture": "frozen piecewise continuous-time competing survival clock",
        "clock_triggered": clock_triggered,
        "sampled_times": {side.value: sampled_times[side] for side in Side},
        "fighters": {side.value: names[side] for side in Side},
        "final_override": final_method == "ko_tko" and clock_triggered,
    }


def _flatten_tick(row: dict) -> dict:
    flat = {k: v for k, v in row.items() if k not in {"options", "ko"}}
    flat["options"] = json.dumps(row.get("options", []), separators=(",", ":"))
    for option in row.get("options", []):
        actor = str(option.get("actor"))
        action = option.get("action") or "clinch_opportunity"
        prefix = f"{actor}_{action}"
        flat[f"{prefix}_rate_15m"] = option.get("rate_15m")
        flat[f"{prefix}_availability_probability_1s"] = option.get("availability_probability_1s")
        flat[f"{prefix}_availability_draw"] = option.get("availability_draw")
        flat[f"{prefix}_available"] = option.get("available")
    for side, ko in row.get("ko", {}).items():
        flat[f"{side}_ko_fighter"] = ko.get("fighter")
        flat[f"{side}_ko_hazard_per_second"] = ko.get("hazard_per_second")
        flat[f"{side}_ko_probability_next_1s"] = ko.get("probability_next_1s")
        flat[f"{side}_ko_sampled_clock_time"] = ko.get("sampled_clock_time")
        flat[f"{side}_ko_fires_in_interval"] = ko.get("fires_in_this_tick_interval")
    return flat


def _write_event_report(captured, fight_id, outdir):
    if not captured:
        raise RuntimeError("--paths 1 requested event report but no path was captured")

    event_payload = {
        "study": "locked Brain MC one-path selected-event report",
        "production_changed": False,
        "fight_id": fight_id,
        "paths": 1,
        "seed": captured["seed"],
        "brain_termination_before_ko_clock": captured.get("brain_termination"),
        "brain_reported_through_seconds": captured.get("brain_reported_through_seconds"),
        "reported_through_seconds": captured["reported_through_seconds"],
        "termination": captured["termination"],
        "ko_clock": captured.get("ko_clock"),
        "events": captured["events"],
    }
    (outdir / "event_report.json").write_text(
        json.dumps(event_payload, indent=2) + "\n", encoding="utf-8"
    )
    event_rows = []
    for row in captured["events"]:
        flat = dict(row)
        flat["brain_options"] = json.dumps(flat["brain_options"], separators=(",", ":"))
        flat["escape_model"] = (
            json.dumps(flat["escape_model"], separators=(",", ":"))
            if flat["escape_model"] is not None
            else None
        )
        event_rows.append(flat)
    pd.DataFrame(event_rows).to_csv(outdir / "event_report.csv", index=False)

    tick_payload = {
        "study": "locked Brain MC one-path every-second availability report",
        "production_changed": False,
        "fight_id": fight_id,
        "paths": 1,
        "seed": captured["seed"],
        "reported_through_seconds": captured["reported_through_seconds"],
        "termination": captured["termination"],
        "ko_clock": captured.get("ko_clock"),
        "ticks": captured.get("ticks", []),
    }
    (outdir / "tick_report.json").write_text(
        json.dumps(tick_payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([_flatten_tick(row) for row in captured.get("ticks", [])]).to_csv(
        outdir / "tick_report.csv", index=False
    )

    print("LOCKED_ONE_PATH_EVENT_REPORT")
    print(json.dumps({
        "fight_id": fight_id,
        "seed": captured["seed"],
        "termination": captured["termination"],
        "brain_termination_before_ko_clock": captured.get("brain_termination"),
        "ko_clock": captured.get("ko_clock"),
        "event_count": len(captured.get("events", [])),
        "tick_rows": len(captured.get("ticks", [])),
    }, indent=2))


def main(*, fight_id: str, paths: int = LOCKED_PATHS) -> None:
    fight_id = str(fight_id).strip()
    if not fight_id:
        raise ValueError("fight_id must be non-empty")
    if isinstance(paths, bool) or not isinstance(paths, int) or paths < 1:
        raise ValueError("paths must be an integer >= 1")
    _set_fight_id(fight_id)
    outdir = OUTROOT / fight_id
    outdir.mkdir(parents=True, exist_ok=True)
    verified_blobs = assert_locked_sources()
    snapshot_path = Path(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"canonical FSR snapshot missing: {snapshot_path}")
    original_snapshot_sha256 = _sha256(snapshot_path)

    with tempfile.TemporaryDirectory(prefix="locked-brain-canonical-") as td:
        backup = Path(td) / snapshot_path.name
        shutil.copy2(snapshot_path, backup)
        original_decay = recency.EWM_DECAY
        original_blend = recency.EWM_CANONICAL_BLEND
        original_timing = timing._new_timing_rates
        original_intent_rates = intent_mod._standing_rates
        original_base_trace_rates = timing.base_trace._standing_rates_no_reset
        original_target_rates = timing.target._standing_rates_no_reset
        original_escape_resolver = timing.target.ExpectedControlEscapeResolver
        original_time_paths = time_ko.PATHS
        original_scored_paths = scored.PATHS
        original_validated_out = validated_kd.OUTDIR
        original_time_run = time_ko.run_causal_path
        original_scored_main = scored.main
        event_capture = None
        try:
            canonical = pd.read_parquet(snapshot_path).copy()
            canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
            canonical["fight_id"] = canonical["fight_id"].astype(str)
            canonical["fighter_id"] = canonical["fighter_id"].astype(str)
            recency.EWM_DECAY = LOCKED_EWM_DECAY
            recency.EWM_CANONICAL_BLEND = LOCKED_EWM_CANONICAL_BLEND
            ewm = recency.build_variant(canonical, "ewm")
            ewm.to_parquet(snapshot_path, index=False)
            target = ewm[ewm["fight_id"].eq(fight_id)].copy()
            if len(target) != 2:
                raise RuntimeError(f"expected 2 PURE EWM target rows for {fight_id}, found {len(target)}")
            target.to_csv(outdir / "pure_ewm05_target_fsr_rows.csv", index=False)

            timing._new_timing_rates = locked_standing_rates
            intent_mod._standing_rates = locked_standing_rates
            timing.base_trace._standing_rates_no_reset = locked_standing_rates
            timing.target._standing_rates_no_reset = locked_standing_rates

            fight, _, _, _, _ = scored.pressure_mod.build_setup()
            by_id = target.set_index(target["fighter_id"].astype(str), drop=False)
            red = by_id.loc[str(fight.r_id)]
            blue = by_id.loc[str(fight.b_id)]
            ground_rates = {
                Side.RED: max(float(red["ground_striking_tendency"]) * float(blue["ground_striking_suppression"]), 0.0),
                Side.BLUE: max(float(blue["ground_striking_tendency"]) * float(red["ground_striking_suppression"]), 0.0),
            }
            ground_bursts = {
                Side.RED: max(float(red["ground_striking_burst_baseline"]), 0.0),
                Side.BLUE: max(float(blue["ground_striking_burst_baseline"]), 0.0),
            }
            tick_clock.configure(
                standing_rate_fn=locked_standing_rates,
                ground_rate_by_side=ground_rates,
                ground_burst_by_side=ground_bursts,
            )

            timing.target.ExpectedControlEscapeResolver = tick_clock.AlwaysEscapeResolver
            time_ko.run_causal_path = tick_clock.run_causal_path
            time_ko.PATHS = paths
            scored.PATHS = paths
            validated_kd.OUTDIR = outdir / "run"
            if paths == 1:
                observed_run, event_capture = _event_report_observer(tick_clock.run_causal_path)
                time_ko.run_causal_path = observed_run
                scored.main = lambda: print("LOCKED_SINGLE_PATH_AGGREGATE_SCORER_SKIPPED")

            manifest = {
                "entry_point": APPROVED_ENTRY_POINT,
                "single_approved_harness": True,
                "run_policy": RUN_POLICY,
                "locked_base_commit": LOCKED_BASE_COMMIT,
                "verified_blobs": verified_blobs,
                "fight_id": fight_id,
                "paths": paths,
                "event_report": paths == 1,
                "event_report_files": [
                    "event_report.json", "event_report.csv", "tick_report.json", "tick_report.csv"
                ] if paths == 1 else [],
                "ewm_decay": LOCKED_EWM_DECAY,
                "ewm_canonical_blend": LOCKED_EWM_CANONICAL_BLEND,
                "clock_architecture": "one global 1-second rate-driven action availability clock",
                "tick_interval_semantics": "300 one-second action intervals per 5-minute round including interval ending at horn",
                "collision_semantics": "rate determines availability once; simultaneous available actions use uniform 1/N tie-break",
                "standing_attempt_scale": LOCKED_STANDING_ATTEMPT_SCALE,
                "standing_rate_source": "pipeline.research.locked_brain_mc.locked_standing_rates",
                "standing_timing_and_chooser_share_exact_callable": True,
                "ground_rate_source": "FSR V3 ground_striking_tendency x opponent ground_striking_suppression plus validated burst baseline",
                "ground_rate_15m": {str(fight.r_name): ground_rates[Side.RED], str(fight.b_name): ground_rates[Side.BLUE]},
                "ground_burst_attempts": {str(fight.r_name): ground_bursts[Side.RED], str(fight.b_name): ground_bursts[Side.BLUE]},
                "ground_action_set_top": ["ground_strike", "submission_attack"],
                "ground_action_set_bottom": ["submission_attack", "escape_stand"],
                "ground_actions_removed": ["control", "bottom_strike", "reversal", "disengage", "improve_position", "advance_position"],
                "escape_semantics": "rate-driven escape event; matchup expected control seconds become mean escape time; selected escape succeeds",
                "ko": "piecewise continuous-time competing survival clock; one-path report reconciled to actual KO override",
                "kd": "OOS-selected static prefight KD hazard; no within-fight KD escalation",
                "submission": "OOS-selected fighter-level submission attempt rate mapped to relevant ground opportunity; conversion unchanged",
                "canonical_artifact_id": CANONICAL_ARTIFACT_ID,
                "canonical_source_run_id": CANONICAL_SOURCE_RUN_ID,
                "canonical_artifact_digest": CANONICAL_ARTIFACT_DIGEST,
                "canonical_snapshot_sha256_before": original_snapshot_sha256,
                "production_changed": False,
            }
            (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print("LOCKED_BRAIN_MC_MANIFEST")
            print(json.dumps(manifest, indent=2))

            validated_kd.main()
            if paths == 1:
                _reconcile_one_path_finish(event_capture, outdir, fight)
                _write_event_report(event_capture, fight_id, outdir)
        finally:
            recency.EWM_DECAY = original_decay
            recency.EWM_CANONICAL_BLEND = original_blend
            timing._new_timing_rates = original_timing
            intent_mod._standing_rates = original_intent_rates
            timing.base_trace._standing_rates_no_reset = original_base_trace_rates
            timing.target._standing_rates_no_reset = original_target_rates
            timing.target.ExpectedControlEscapeResolver = original_escape_resolver
            time_ko.PATHS = original_time_paths
            scored.PATHS = original_scored_paths
            validated_kd.OUTDIR = original_validated_out
            time_ko.run_causal_path = original_time_run
            scored.main = original_scored_main
            shutil.copy2(backup, snapshot_path)

    restored_sha256 = _sha256(snapshot_path)
    if restored_sha256 != original_snapshot_sha256:
        raise RuntimeError(
            f"canonical FSR restore verification failed: before={original_snapshot_sha256} after={restored_sha256}"
        )
    restore = {
        "canonical_snapshot_sha256_before": original_snapshot_sha256,
        "canonical_snapshot_sha256_after": restored_sha256,
        "byte_identical_restore": True,
    }
    (outdir / "restore_verification.json").write_text(json.dumps(restore, indent=2) + "\n", encoding="utf-8")
    print("CANONICAL_RESTORE_VERIFIED")
    print(json.dumps(restore, indent=2))


if __name__ == "__main__":
    args = _parse_args()
    main(fight_id=args.fight_id, paths=args.paths)
