"""Single approved fight-agnostic locked Brain MC research harness.

ALL Brain MC research runs MUST execute through this module unless the user
explicitly approves an exception. Do not create or execute one-off runners,
ad-hoc fight scripts, alternate harnesses, wrapper runners, or workflow
monkeypatch runners.

This file contains the locked implementation itself. There is no second harness
underneath it.

CLI examples:

    python -m pipeline.research.locked_brain_mc --fight-id 419fff06f338f5c6 --paths 500
    python -m pipeline.research.locked_brain_mc --fight-id 419fff06f338f5c6 --paths 1

--paths 1 automatically writes event_report.json and event_report.csv with the
full Brain action distribution plus mechanic probabilities/outcomes for each
event. Single-path mode skips only the legacy multi-path aggregate scorer after
the causal path has been generated; mechanics, seeds, KO/KD, submissions, and
event capture remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    ActionResolution,
    TransitionKind,
    TransitionRequest,
)

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
    parser.add_argument("--fight-id", required=True, help="target fight_id from repository master/prefight data")
    parser.add_argument(
        "--paths", type=int, default=LOCKED_PATHS,
        help=f"number of matched-seed paths (default: {LOCKED_PATHS}); --paths 1 automatically emits event_report.json/csv",
    )
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
    drift = subprocess.run(["git", "diff", "--quiet", LOCKED_BASE_COMMIT, "--", *LOCKED_CORE_PATHS], check=False).returncode
    if drift != 0:
        changed = _git("diff", "--name-only", LOCKED_BASE_COMMIT, "--", *LOCKED_CORE_PATHS)
        failures.append("core source drift from locked base commit:\n" + changed)
    if failures:
        raise RuntimeError("LOCKED BRAIN HARNESS REFUSED TO RUN\n" + "\n".join(failures))
    return blobs


def _scaled_standing_rates(original):
    def locked_scaled_rates(state, actor, capabilities, context, priors, config):
        rates, pressure = original(state, actor, capabilities, context, priors, config)
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] = float(rates[ActionFamily.STAND_ATTACK]) * LOCKED_STANDING_ATTEMPT_SCALE
        return rates, pressure
    return locked_scaled_rates


def _round_budget_resolver_class(original_cls):
    class RoundBudgetEscapeResolver(original_cls):
        def __init__(self, model, seed):
            super().__init__(model, seed)
            self.round_budgets = {}
            self.round_consumed = {}
        def _budget_key(self, state):
            return (int(state.round_number), state.ground_controller.value)
        def _ensure_budget(self, state):
            key = self._budget_key(state)
            if key not in self.round_budgets:
                ratio = float(self.rng.choice(self.model["ratios"]))
                expected = self._expected(state.ground_controller)
                budget = float(np.clip(expected * ratio, timing.target.MIN_DURATION, timing.target.MAX_DURATION))
                self.round_budgets[key] = budget
                self.round_consumed.setdefault(key, 0.0)
            return self.round_budgets[key]
        def _spell(self, state):
            key = (int(state.round_number), state.ground_controller.value, round(float(state.phase_started_at), 9))
            if key not in self.spells:
                budget_key = self._budget_key(state)
                budget = self._ensure_budget(state)
                consumed = float(self.round_consumed.get(budget_key, 0.0))
                remaining = max(0.0, budget - consumed)
                self.spells[key] = {
                    "round": int(state.round_number), "controller": state.ground_controller.value,
                    "phase_started_at": float(state.phase_started_at),
                    "expected_control_seconds": self._expected(state.ground_controller),
                    "round_control_budget_seconds": budget,
                    "round_control_consumed_before_spell_seconds": consumed,
                    "sampled_escape_threshold_seconds": remaining, "round_budget_semantics": True,
                }
            return self.spells[key]
        def _consume_spell(self, state, elapsed):
            key = self._budget_key(state)
            budget = self._ensure_budget(state)
            prior = float(self.round_consumed.get(key, 0.0))
            self.round_consumed[key] = min(budget, prior + max(float(elapsed), 0.0))
        def __call__(self, event, state, inputs, rng, placeholders, ko_kd_rng=None, submission_rng=None):
            if event.action_family is ActionFamily.ESCAPE_STAND:
                spell = self._spell(state)
                elapsed = float(state.fight_time_seconds - state.phase_started_at)
                threshold = float(spell["sampled_escape_threshold_seconds"])
                succeeded = elapsed >= threshold
                self.escape_checks.append({"timestamp": float(event.timestamp_seconds), "actor": event.actor.value, "controller": state.ground_controller.value, "elapsed_control_seconds": elapsed, **spell, "success": bool(succeeded)})
                if succeeded:
                    self._consume_spell(state, elapsed)
                return ActionResolution(event, ActionOutcome.ESCAPED if succeeded else ActionOutcome.FAILURE, TransitionRequest(TransitionKind.ESCAPE_GROUND, Phase.GROUND, Phase.STANDING) if succeeded else None)
            resolution = resolve_action(event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng)
            if state.phase is Phase.GROUND and resolution.transition is not None and resolution.transition.kind in {TransitionKind.REVERSE_GROUND, TransitionKind.DISENGAGE_GROUND, TransitionKind.ESCAPE_GROUND}:
                self._consume_spell(state, float(state.fight_time_seconds - state.phase_started_at))
            return resolution
    RoundBudgetEscapeResolver.__name__ = "LockedRoundBudgetEscapeResolver"
    return RoundBudgetEscapeResolver


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
        return {"mechanic": "round_control_budget_escape_threshold", "probability": None}
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
            pre_mechanics.append({"timestamp": float(event.timestamp_seconds), "actor": event.actor.value, "action": event.action_family.value, **_primary_mechanic_probability(event, state, mechanics_inputs, placeholders)})
            return original_resolver(event, state, mechanics_inputs, rng, placeholders, ko_kd_rng, submission_rng)
        observed_functions = EngineFunctions(timing_sampler=original_functions.timing_sampler, action_chooser=original_functions.action_chooser, mechanics_resolver=observed_resolver)
        kwargs = {"seed": seed, "horizon_seconds": horizon_seconds, "functions": observed_functions}
        if initial_state is not None:
            kwargs["initial_state"] = initial_state
        if config is not None:
            kwargs["config"] = config
        out = original_run(inputs, **kwargs)
        brain = getattr(original_functions.action_chooser, "__self__", None)
        decisions = list(getattr(brain, "decisions", []))
        if len(decisions) != len(out.events):
            raise RuntimeError(f"event-report decision/event mismatch: {len(decisions)} != {len(out.events)}")
        if len(pre_mechanics) != len(out.events):
            raise RuntimeError(f"event-report mechanics/event mismatch: {len(pre_mechanics)} != {len(out.events)}")
        escape_checks = list(getattr(original_resolver, "escape_checks", []))
        escape_by_key = {(round(float(x["timestamp"]), 9), str(x["actor"])): x for x in escape_checks}
        rows = []
        previous_timestamp = 0.0
        for index, (decision, event, mechanic) in enumerate(zip(decisions, out.events, pre_mechanics, strict=True)):
            key = (round(float(event.timestamp_seconds), 9), event.actor.value)
            primary_probability = float(event.submission_probability) if event.selected_action is ActionFamily.SUBMISSION_ATTACK else mechanic["probability"]
            rows.append({
                "event_index": index, "event_timestamp": float(event.timestamp_seconds),
                "seconds_since_prior_event": float(event.timestamp_seconds) - previous_timestamp,
                "round": int(decision["round"]), "phase": decision["phase"], "actor": event.actor.value,
                "selected_action": event.selected_action.value, "brain_options": decision["brain_options"],
                "brain_selected_probability": next((float(x["probability"]) for x in decision["brain_options"] if x["action"] == event.selected_action.value), None),
                "dynamic_pressure": decision.get("dynamic_pressure"), "mechanic": mechanic["mechanic"],
                "mechanic_probability": primary_probability, "outcome": event.outcome.value,
                "transition_kind": _enum(event.transition_kind), "resulting_phase": event.resulting_phase.value,
                "resulting_controller": _enum(event.resulting_controller), "escape_model": escape_by_key.get(key),
                "impact": float(event.impact), "kd_probability": float(event.kd_probability), "knockdown": bool(event.knockdown),
                "ko_probability": float(event.ko_probability), "ko_tko": bool(event.ko_tko),
                "submission_attempt": bool(event.submission_attempt), "submission_probability": float(event.submission_probability),
                "submission_success": bool(event.submission_success),
            })
            previous_timestamp = float(event.timestamp_seconds)
        captured["seed"] = int(seed)
        captured["reported_through_seconds"] = float(out.reported_through_seconds)
        captured["termination"] = None if out.termination is None else {"winner_side": out.termination.winner.value, "method": out.termination.finish_method.value}
        captured["events"] = rows
        return out
    return observed_run, captured


def _write_event_report(captured, fight_id, outdir):
    if not captured:
        raise RuntimeError("--paths 1 requested event report but no path was captured")
    payload = {"study": "locked Brain MC one-path event report", "production_changed": False, "fight_id": fight_id, "paths": 1, "seed": captured["seed"], "reported_through_seconds": captured["reported_through_seconds"], "termination": captured["termination"], "events": captured["events"]}
    (outdir / "event_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    flat_rows = []
    for row in captured["events"]:
        flat = dict(row)
        flat["brain_options"] = json.dumps(flat["brain_options"], separators=(",", ":"))
        flat["escape_model"] = json.dumps(flat["escape_model"], separators=(",", ":")) if flat["escape_model"] is not None else None
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(outdir / "event_report.csv", index=False)
    print("LOCKED_ONE_PATH_EVENT_REPORT")
    print(json.dumps(payload, indent=2))


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
            timing._new_timing_rates = _scaled_standing_rates(original_timing)
            timing.target.ExpectedControlEscapeResolver = _round_budget_resolver_class(original_escape_resolver)
            time_ko.PATHS = paths
            scored.PATHS = paths
            validated_kd.OUTDIR = outdir / "run"
            if paths == 1:
                observed_run, event_capture = _event_report_observer(original_time_run)
                time_ko.run_causal_path = observed_run
                scored.main = lambda: print("LOCKED_SINGLE_PATH_AGGREGATE_SCORER_SKIPPED")
            manifest = {
                "entry_point": APPROVED_ENTRY_POINT, "single_approved_harness": True, "run_policy": RUN_POLICY,
                "locked_base_commit": LOCKED_BASE_COMMIT, "verified_blobs": verified_blobs, "fight_id": fight_id,
                "fight_id_source": "CLI --fight-id", "paths": paths, "paths_source": "CLI --paths or locked default",
                "event_report": paths == 1, "event_report_files": ["event_report.json", "event_report.csv"] if paths == 1 else [],
                "single_path_aggregate_scorer_skipped": paths == 1,
                "ewm_decay": LOCKED_EWM_DECAY, "ewm_canonical_blend": LOCKED_EWM_CANONICAL_BLEND,
                "standing_attempt_scale": LOCKED_STANDING_ATTEMPT_SCALE,
                "control_duration_semantics": "one sampled round-total control budget per controller per round; re-entries consume remaining budget",
                "ko": "piecewise time-based competing clock from allen_shahbazyan_time_ko_clock_2000",
                "kd": "OOS-selected static prefight KD hazard; no within-fight KD escalation",
                "submission": "current locked ground-opportunity/fighter-level submission research stack",
                "canonical_artifact_id": CANONICAL_ARTIFACT_ID, "canonical_source_run_id": CANONICAL_SOURCE_RUN_ID,
                "canonical_artifact_digest": CANONICAL_ARTIFACT_DIGEST,
                "canonical_snapshot_sha256_before": original_snapshot_sha256, "production_changed": False,
            }
            (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print("LOCKED_BRAIN_MC_MANIFEST")
            print(json.dumps(manifest, indent=2))
            validated_kd.main()
            if paths == 1:
                _write_event_report(event_capture, fight_id, outdir)
        finally:
            recency.EWM_DECAY = original_decay
            recency.EWM_CANONICAL_BLEND = original_blend
            timing._new_timing_rates = original_timing
            timing.target.ExpectedControlEscapeResolver = original_escape_resolver
            time_ko.PATHS = original_time_paths
            scored.PATHS = original_scored_paths
            validated_kd.OUTDIR = original_validated_out
            time_ko.run_causal_path = original_time_run
            scored.main = original_scored_main
            shutil.copy2(backup, snapshot_path)
    restored_sha256 = _sha256(snapshot_path)
    if restored_sha256 != original_snapshot_sha256:
        raise RuntimeError(f"canonical FSR restore verification failed: before={original_snapshot_sha256} after={restored_sha256}")
    restore = {"canonical_snapshot_sha256_before": original_snapshot_sha256, "canonical_snapshot_sha256_after": restored_sha256, "byte_identical_restore": True}
    (outdir / "restore_verification.json").write_text(json.dumps(restore, indent=2) + "\n", encoding="utf-8")
    print("CANONICAL_RESTORE_VERIFIED")
    print(json.dumps(restore, indent=2))


if __name__ == "__main__":
    args = _parse_args()
    main(fight_id=args.fight_id, paths=args.paths)
