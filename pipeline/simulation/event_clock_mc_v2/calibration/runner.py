"""Deterministic historical Event Clock V2 calibration runner."""

from __future__ import annotations
import argparse, json, math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd
from pipeline.common.paths import (
    EVENT_CLOCK_V2_COHORT_MANIFEST_PATH,
    EVENT_CLOCK_V2_HISTORICAL_TARGETS_PATH,
    MASTER_PATH,
)
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import (
    EngineConfig,
    EngineFunctions,
    EngineInputs,
    FighterEngineInputs,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    fighter_mechanics_from_prefight,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import (
    _capabilities,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_intent_prior_shadow import (
    IntentPriorChooser,
)
from . import COHORT_VERSION, SEED_SET_VERSION
from .cohort import select_split, validate_manifest, validate_manifest_prefight_contract
from .config import config_hash, load_override_file, resolved_payload
from .invariants import inspect_path, status
from .ledger import artifact_digest, build_record, metrics_fingerprint, write_record
from .seeds import derive_path_seed
from .targets import evaluate, verify_frozen_targets

STAND = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
GROUND = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}


def scheduled_horizon_seconds(fight: pd.Series, bout_id: str) -> float:
    """Return the scheduled limit, independent of realized historical duration."""
    scheduled_rounds = int(fight["total_rounds"])
    if scheduled_rounds < 1:
        raise ValueError(f"invalid scheduled rounds for bout {bout_id}")
    return float(scheduled_rounds * 300)


def run(
    *,
    split: str,
    paths_per_fight: int,
    config_path: Path | None,
    output: Path,
    limit: int | None = None,
) -> dict:
    manifest = pd.read_csv(
        EVENT_CLOCK_V2_COHORT_MANIFEST_PATH,
        dtype={"bout_id": str, "red_fighter_id": str, "blue_fighter_id": str},
    )
    validate_manifest(manifest)
    cohort = select_split(manifest, split)
    if limit:
        cohort = cohort.head(limit)
    snapshots = load_prefight_snapshots()
    validate_manifest_prefight_contract(manifest, snapshots)
    cutoff = pd.to_datetime(manifest.date).min()
    reference = CapabilityReference.from_prefight_before(snapshots, cutoff)
    mechanics_config, explicit = load_override_file(config_path)
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    master["fight_id"] = master.fight_id.astype(str)
    lookup = master.set_index("fight_id")
    counts = Counter()
    phase = Counter()
    inv = Counter()
    stamina = []
    trauma = []
    acute = []
    finish_times = []
    seconds = 0.0
    finishes = Counter()
    path_fingerprints = []
    replay_mismatch = 0
    for _, item in cohort.iterrows():
        fight = lookup.loc[str(item.bout_id)]
        # Never censor simulated paths at the realized historical finish time.
        horizon = scheduled_horizon_seconds(fight, str(item.bout_id))
        date = pd.Timestamp(item.date)
        red, blue = historical_fighter_rows(
            snapshots,
            event_date=date,
            fight_id=str(item.bout_id),
            fighter_ids=(str(item.red_fighter_id), str(item.blue_fighter_id)),
        )
        rc, rr = _capabilities(red, blue, reference)
        bc, br = _capabilities(blue, red, reference)
        inputs = EngineInputs(
            FighterEngineInputs(
                rc,
                BrainTimingContext(),
                BrainDecisionContext(),
                fighter_mechanics_from_prefight(red, rr),
            ),
            FighterEngineInputs(
                bc,
                BrainTimingContext(),
                BrainDecisionContext(),
                fighter_mechanics_from_prefight(blue, br),
            ),
            mechanics_calibration=mechanics_config,
        )
        chooser = IntentPriorChooser(
            {
                Side.RED: BrainIntentPriors(
                    rr.standing_rate_15m, rr.takedown_rate_15m, 0.06, 3.0, 0.3
                ),
                Side.BLUE: BrainIntentPriors(
                    br.standing_rate_15m, br.takedown_rate_15m, 0.06, 3.0, 0.3
                ),
            }
        )
        funcs = EngineFunctions(action_chooser=chooser)
        cfg = EngineConfig(number_of_rounds=max(1, math.ceil(horizon / 300)))
        for path_id in range(paths_per_fight):
            seed = derive_path_seed(SEED_SET_VERSION, str(item.bout_id), path_id)
            out = run_causal_path(
                inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs
            )
            if not path_fingerprints:
                replay = run_causal_path(
                    inputs,
                    seed=seed,
                    horizon_seconds=horizon,
                    config=cfg,
                    functions=funcs,
                )
                signature = lambda result: (
                    len(result.events),
                    result.reported_through_seconds,
                    (
                        result.termination.finish_method.value
                        if result.termination
                        else None
                    ),
                    asdict(result.final_state),
                )
                replay_mismatch += int(signature(out) != signature(replay))
            seconds += out.reported_through_seconds
            for k, v in inspect_path(out).items():
                inv[k] += v
            for segment in out.timeline_segments:
                phase[segment.phase.value] += segment.duration
            p = out.final_state.physiology
            for row in (p.red, p.blue):
                stamina.append(row.stamina)
                trauma.append(row.cumulative_trauma)
                acute.append(row.acute_vulnerability)
            for event in out.events:
                a = event.selected_action
                counts[
                    (
                        "standing"
                        if a in STAND
                        else (
                            "ground"
                            if a in GROUND
                            else (
                                "clinch"
                                if a is ActionFamily.CLINCH_STRIKE
                                else (
                                    "td"
                                    if a in TD
                                    else (
                                        "sub"
                                        if a is ActionFamily.SUBMISSION_ATTACK
                                        else "other"
                                    )
                                )
                            )
                        )
                    )
                ] += 1
                counts["td_landed"] += int(
                    a in TD and event.resulting_phase.value == "ground"
                )
                counts["knockdowns"] += int(event.knockdown)
            if out.termination:
                finishes[out.termination.finish_method.value] += 1
                finish_times.append(out.reported_through_seconds)
            path_fingerprints.append(
                (
                    str(item.bout_id),
                    path_id,
                    len(out.events),
                    out.reported_through_seconds,
                    out.termination.finish_method.value if out.termination else None,
                    p.red.stamina,
                    p.blue.stamina,
                )
            )
    total_paths = len(cohort) * paths_per_fight
    total_phase = sum(phase.values())
    q = lambda x: (
        {
            "p10": float(np.quantile(x, 0.1)),
            "p25": float(np.quantile(x, 0.25)),
            "p50": float(np.quantile(x, 0.5)),
            "p75": float(np.quantile(x, 0.75)),
            "p90": float(np.quantile(x, 0.9)),
            "p95": float(np.quantile(x, 0.95)),
            "max": float(np.max(x)),
        }
        if x
        else {}
    )
    rate = lambda n: float(n * 900 / seconds / 2) if seconds else 0.0
    metrics = {
        "standing_attempts_per_fighter_15": rate(counts["standing"]),
        "clinch_strikes_per_fighter_15": rate(counts["clinch"]),
        "ground_strikes_per_fighter_15": rate(counts["ground"]),
        "td_attempts_per_fighter_15": rate(counts["td"]),
        "td_landed_per_fighter_15": rate(counts["td_landed"]),
        "submissions_per_fighter_15": rate(counts["sub"]),
        "standing_phase_share": phase["standing"] / total_phase,
        "clinch_phase_share": phase["clinch"] / total_phase,
        "ground_phase_share": phase["ground"] / total_phase,
        "mean_fight_duration_seconds": seconds / total_paths,
        "knockdowns_per_fight": counts["knockdowns"] / total_paths,
        "ko_tko_fight_share": finishes["ko_tko"] / total_paths,
        "submission_fight_share": finishes["submission"] / total_paths,
        "decision_fight_share": (total_paths - sum(finishes.values())) / total_paths,
        "mean_trauma_per_fighter_15": float(
            np.mean(trauma) * 900 / (seconds / total_paths)
        ),
        "trauma_quantiles": q(trauma),
        "acute_vulnerability_quantiles": q(acute),
        "mean_final_stamina": float(np.mean(stamina)),
        "final_stamina_quantiles": q(stamina),
        "finish_time_distribution": q(finish_times),
        "ko_tko_allocation_by_fighter": None,
        "actual_ko_winner_power_edge": None,
        "kd_allocation_relative_to_attacker_power": None,
        "td_allocation_relative_to_td_traits": None,
        "late_fight_performance_relative_to_stamina_traits": None,
        "winner_accuracy": None,
        "brier_score": None,
        "log_loss": None,
        "mean_favorite_probability": None,
        "predicted_probability_distribution": None,
        "calibration_bins": None,
        "diagnostic_note": "V2 causal runner has no approved historical judging/probability layer; predictive and allocation diagnostics without an approved outcome mapper are unavailable, not fabricated.",
        "simulator_phase_share_note": "Simulator exposure only; UFCStats has no authoritative historical phase-time denominator.",
    }
    invariants = status(dict(inv), replay_mismatch)
    targets = json.loads(EVENT_CLOCK_V2_HISTORICAL_TARGETS_PATH.read_text())
    verify_frozen_targets(targets)
    comparisons = evaluate({**metrics, **invariants["counts"]}, targets)
    identity = {
        "cohort_version": COHORT_VERSION,
        "cohort_manifest_digest": artifact_digest(EVENT_CLOCK_V2_COHORT_MANIFEST_PATH),
        "cohort_split": split,
        "fight_count": len(cohort),
        "paths_per_fight": paths_per_fight,
        "seed_set_version": SEED_SET_VERSION,
        "target_digest": targets["target_digest"],
        "parameter_config_hash": config_hash(
            resolved_payload(mechanics_config, explicit)
        ),
    }
    record = build_record(
        identity=identity,
        config=resolved_payload(mechanics_config, explicit),
        metrics={**metrics, "acceptance_results": comparisons},
        invariants=invariants,
    )
    record["path_replay_digest"] = config_hash(path_fingerprints)
    record["metrics_fingerprint"] = metrics_fingerprint(metrics, invariants)
    write_record(record, output)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohort",
        choices=("development", "calibration", "validation", "final_holdout"),
        default="calibration",
    )
    p.add_argument("--paths-per-fight", type=int, default=50)
    p.add_argument("--calibration-config", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.paths_per_fight < 1:
        p.error("paths-per-fight must be positive")
    print(
        json.dumps(
            run(
                split=a.cohort,
                paths_per_fight=a.paths_per_fight,
                config_path=a.calibration_config,
                output=a.output,
                limit=a.limit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
