"""Deterministic historical Event Clock V2 calibration runner."""

from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import joblib
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
    age_years_on_date,
    fighter_mechanics_from_prefight,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture
from pipeline.simulation.event_clock_mc_v2.mechanics.ko_kd_empirical import (
    kd_probability as empirical_kd_probability,
    ko_probability as empirical_ko_probability,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.judging import Event2JudgeModel
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
    ko_kd_architecture: KOKDArchitecture = KOKDArchitecture.EMPIRICAL_EVENT2,
    event_name: str | None = None,
) -> dict:
    manifest = pd.read_csv(
        EVENT_CLOCK_V2_COHORT_MANIFEST_PATH,
        dtype={"bout_id": str, "red_fighter_id": str, "blue_fighter_id": str},
    )
    validate_manifest(manifest)
    cohort = select_split(manifest, split)
    snapshots = load_prefight_snapshots()
    validate_manifest_prefight_contract(manifest, snapshots)
    cutoff = pd.to_datetime(manifest.date).min()
    reference = CapabilityReference.from_prefight_before(snapshots, cutoff)
    mechanics_config, explicit = load_override_file(config_path)
    bundle = joblib.load(DEFAULT_BUNDLE_PATH)
    context = bundle["context"]
    source_judge = context["judge_model"]
    training_decisions = int(context.get("judge_training_decisions", 0))
    judge_model = Event2JudgeModel.from_sklearn(source_judge, training_decisions=training_decisions)
    submission_offset = float(context["conversion_offset"])
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    master["fight_id"] = master.fight_id.astype(str)
    if event_name:
        selected = master[master.event_name.astype(str).eq(event_name)].copy()
        if selected.empty:
            raise ValueError(f"event not found: {event_name}")
        cohort = pd.DataFrame({
            "bout_id": selected.fight_id.astype(str), "date": selected.date,
            "red_fighter_id": selected.r_id.astype(str), "blue_fighter_id": selected.b_id.astype(str),
            "red_fighter": selected.r_name.astype(str), "blue_fighter": selected.b_name.astype(str),
        })
    if limit:
        cohort = cohort.head(limit)
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
    phase_ko_kd = defaultdict(Counter)
    allocation = defaultdict(list)
    ko_prior = Counter()
    ko_path_kds = []
    age_audit_candidates = []
    winners = Counter()
    decision_classes = Counter()
    decision_round_counts = Counter()
    round_probabilities = []
    judge_disagreements = 0
    submission_attempts = submission_successes = 0
    submission_probabilities = []
    submission_finish_times = []
    submission_winner_offense_edges = []
    submission_loser_defense_edges = []
    submission_winners = Counter()
    fight_path_outcomes = defaultdict(Counter)
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
        red_age = age_years_on_date(fight.get("r_dob"), date)
        blue_age = age_years_on_date(fight.get("b_dob"), date)
        red_mechanics = fighter_mechanics_from_prefight(red, rr, age_years=red_age, submission_conversion_offset=submission_offset)
        blue_mechanics = fighter_mechanics_from_prefight(blue, br, age_years=blue_age, submission_conversion_offset=submission_offset)
        for side, mechanics, opponent, dob in (
            ("red", red_mechanics, blue_mechanics, fight.get("r_dob")),
            ("blue", blue_mechanics, red_mechanics, fight.get("b_dob")),
        ):
            age_audit_candidates.append(
                {
                    "bout_id": str(item.bout_id),
                    "fight_date": date.date().isoformat(),
                    "side": side,
                    "fighter": str(
                        item.red_fighter if side == "red" else item.blue_fighter
                    ),
                    "dob": (
                        None if pd.isna(dob) else pd.Timestamp(dob).date().isoformat()
                    ),
                    "age_source": "master_dob_at_exact_fight_date",
                    "age_years": mechanics.age_years,
                    "striking_power": mechanics.striking_power,
                    "knockdown_resistance": mechanics.knockdown_resistance,
                    "p_ko_prior0": empirical_ko_probability(
                        mechanics,
                        opponent,
                        prior_defender_kds=0,
                        elapsed_seconds=0.0,
                        attacker_stamina=1.0,
                    ),
                    "p_kd_prior0": empirical_kd_probability(
                        mechanics,
                        opponent,
                        prior_defender_kds=0,
                        elapsed_seconds=0.0,
                        attacker_stamina=1.0,
                    ),
                }
            )
        inputs = EngineInputs(
            FighterEngineInputs(
                rc,
                BrainTimingContext(),
                BrainDecisionContext(),
                red_mechanics,
            ),
            FighterEngineInputs(
                bc,
                BrainTimingContext(),
                BrainDecisionContext(),
                blue_mechanics,
            ),
            mechanics_calibration=mechanics_config,
            ko_kd_architecture=ko_kd_architecture,
            judge_model=judge_model,
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
            path_knockdowns = 0
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
                    asdict(result.decision) if result.decision else None,
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
                if event.submission_attempt:
                    submission_attempts += 1
                    submission_successes += int(event.submission_success)
                    submission_probabilities.append(event.submission_probability)
                if event.knockdown:
                    path_knockdowns += 1
                if (
                    event.outcome.value == "landed"
                    and event.selected_action
                    in STAND | GROUND | {ActionFamily.CLINCH_STRIKE}
                ):
                    family = (
                        "standing"
                        if event.selected_action in STAND
                        else (
                            "clinch"
                            if event.selected_action is ActionFamily.CLINCH_STRIKE
                            else (
                                "bottom_ground"
                                if event.selected_action is ActionFamily.BOTTOM_STRIKE
                                else "ground"
                            )
                        )
                    )
                    diag = phase_ko_kd[family]
                    diag["landed_eligible_strikes"] += 1
                    diag["ko_probability_sum"] += event.ko_probability
                    if not event.ko_tko:
                        diag["kd_opportunities"] += 1
                        diag["kd_probability_sum"] += event.kd_probability
                    diag["realized_kds"] += int(event.knockdown)
                    diag["realized_kos"] += int(event.ko_tko)
                    attacker = (
                        red_mechanics if event.actor is Side.RED else blue_mechanics
                    )
                    defender = (
                        blue_mechanics if event.actor is Side.RED else red_mechanics
                    )
                    power_edge = attacker.striking_power - defender.striking_power
                    resistance_edge = (
                        defender.knockdown_resistance - attacker.knockdown_resistance
                    )
                    if event.ko_tko:
                        allocation["ko_power_edges"].append(power_edge)
                        ko_prior[">=1" if event.prior_defender_kds >= 1 else "0"] += 1
                    if event.knockdown:
                        allocation["kd_power_edges"].append(power_edge)
                        allocation["kd_defender_resistance_edges"].append(
                            resistance_edge
                        )
            if out.termination:
                finishes[out.termination.finish_method.value] += 1
                winners[out.termination.winner.value] += 1
                path_key = str(item.bout_id)
                fight_path_outcomes[path_key][out.termination.winner.value] += 1
                fight_path_outcomes[path_key][f"{out.termination.winner.value}_{out.termination.finish_method.value}"] += 1
                fight_path_outcomes[path_key][out.termination.finish_method.value] += 1
                fight_path_outcomes[path_key]["paths"] += 1
                for rounds_line in (1.5, 2.5, 3.5, 4.5):
                    fight_path_outcomes[path_key][f"under_{rounds_line}"] += int(out.reported_through_seconds < rounds_line * 300)
                finish_times.append(out.reported_through_seconds)
                if out.termination.finish_method.value == "ko_tko":
                    ko_path_kds.append(path_knockdowns)
                if out.termination.finish_method.value == "submission":
                    submission_finish_times.append(out.reported_through_seconds)
                    winner_name = str(item.red_fighter if out.termination.winner is Side.RED else item.blue_fighter)
                    submission_winners[winner_name] += 1
                    attacker = red_mechanics if out.termination.winner is Side.RED else blue_mechanics
                    defender = blue_mechanics if out.termination.winner is Side.RED else red_mechanics
                    submission_winner_offense_edges.append(attacker.submission_offense - defender.submission_offense)
                    submission_loser_defense_edges.append(defender.submission_defense - attacker.submission_defense)
            if out.decision:
                decision_classes[out.decision.classification] += 1
                decision_round_counts[len(out.decision.round_probabilities)] += 1
                round_probabilities.extend(out.decision.round_probabilities)
                judge_disagreements += int(out.decision.classification == "split_decision")
                fight_path_outcomes[str(item.bout_id)][out.decision.classification] += 1
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
    phase_breakdown = {
        family: {
            "landed_eligible_strikes": int(values["landed_eligible_strikes"]),
            "mean_p_ko": values["ko_probability_sum"]
            / max(1, values["landed_eligible_strikes"]),
            "mean_p_kd_survived_strike": values["kd_probability_sum"]
            / max(1, values["kd_opportunities"]),
            "realized_kds": int(values["realized_kds"]),
            "realized_kos": int(values["realized_kos"]),
        }
        for family, values in sorted(phase_ko_kd.items())
    }
    allocation_metrics = {
        key: {
            "count": len(values),
            "mean_edge": float(np.mean(values)) if values else None,
            "positive_edge_share": (
                float(np.mean(np.asarray(values) > 0)) if values else None
            ),
        }
        for key, values in allocation.items()
    }
    age_sorted = sorted(age_audit_candidates, key=lambda row: row["age_years"])
    age_audit = [
        age_sorted[index] for index in np.linspace(0, len(age_sorted) - 1, 6, dtype=int)
    ]
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
        "decision_fight_share": finishes["decision"] / total_paths,
        "red_path_win_probability": winners["red"] / total_paths,
        "blue_path_win_probability": winners["blue"] / total_paths,
        "unanimous_decision_fight_share": decision_classes["unanimous_decision"] / total_paths,
        "split_decision_fight_share": decision_classes["split_decision"] / total_paths,
        "decision_winner_allocation": {side: winners[side] / total_paths for side in ("red", "blue")},
        "round_red_win_probability_distribution": q(round_probabilities),
        "judge_disagreement_rate_per_decision": judge_disagreements / max(1, finishes["decision"]),
        "decision_round_count_distribution": {str(k): v / max(1, finishes["decision"]) for k, v in sorted(decision_round_counts.items())},
        "submission_conversion_rate_per_attempt": submission_successes / max(1, submission_attempts),
        "mean_submission_conversion_probability": float(np.mean(submission_probabilities)) if submission_probabilities else None,
        "submission_finish_time_distribution": q(submission_finish_times),
        "submission_winner_allocation": dict(submission_winners),
        "submission_offense_edge_among_winners": float(np.mean(submission_winner_offense_edges)) if submission_winner_offense_edges else None,
        "loser_submission_defense_edge": float(np.mean(submission_loser_defense_edges)) if submission_loser_defense_edges else None,
        "judge_model": {
            "source": judge_model.source,
            "features": list(("sig_diff", "kd_diff", "td_diff", "sub_diff", "ctrl_diff")),
            "training_decisions": judge_model.training_decisions,
            "scaler_mean": list(judge_model.scaler_mean), "scaler_scale": list(judge_model.scaler_scale),
            "intercept": judge_model.intercept, "coefficients": list(judge_model.coefficients),
        },
        "submission_model": {"source": "integrated_event2_replay", "conversion_offset": submission_offset},
        "fight_probabilities": {
            str(row.bout_id): {
                "red_fighter": str(row.red_fighter), "blue_fighter": str(row.blue_fighter),
                "red_moneyline": fight_path_outcomes[str(row.bout_id)]["red"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"]),
                "blue_moneyline": fight_path_outcomes[str(row.bout_id)]["blue"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"]),
                **{f"{side}_{method}": fight_path_outcomes[str(row.bout_id)][f"{side}_{method}"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"])
                   for side in ("red", "blue") for method in ("ko_tko", "submission", "decision")},
                **{f"fight_{method}": fight_path_outcomes[str(row.bout_id)][method] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"])
                   for method in ("ko_tko", "submission", "decision")},
                "unanimous_decision": fight_path_outcomes[str(row.bout_id)]["unanimous_decision"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"]),
                "split_decision": fight_path_outcomes[str(row.bout_id)]["split_decision"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"]),
                **{f"under_{line}": fight_path_outcomes[str(row.bout_id)][f"under_{line}"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"])
                   for line in (1.5, 2.5, 3.5, 4.5)},
                **{f"over_{line}": 1.0 - fight_path_outcomes[str(row.bout_id)][f"under_{line}"] / max(1, fight_path_outcomes[str(row.bout_id)]["paths"])
                   for line in (1.5, 2.5, 3.5, 4.5)},
            } for row in cohort.itertuples()
        },
        "mean_trauma_per_fighter_15": float(
            np.mean(trauma) * 900 / (seconds / total_paths)
        ),
        "trauma_quantiles": q(trauma),
        "acute_vulnerability_quantiles": q(acute),
        "mean_final_stamina": float(np.mean(stamina)),
        "final_stamina_quantiles": q(stamina),
        "finish_time_distribution": q(finish_times),
        "ko_kd_architecture": ko_kd_architecture.value,
        "phase_ko_kd_breakdown": phase_breakdown,
        "kds_per_ko_fight": float(np.mean(ko_path_kds)) if ko_path_kds else None,
        "ko_allocation_by_prior_defender_kds": {
            "prior_0_count": int(ko_prior["0"]),
            "prior_ge_1_count": int(ko_prior[">=1"]),
            "prior_0_share": ko_prior["0"] / max(1, sum(ko_prior.values())),
            "prior_ge_1_share": ko_prior[">=1"] / max(1, sum(ko_prior.values())),
        },
        "ko_kd_allocation_diagnostics": allocation_metrics,
        "historical_age_provenance_audit": age_audit,
        "ko_tko_allocation_by_fighter": allocation_metrics.get("ko_power_edges"),
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
        "diagnostic_note": "Decision judging is EVENT2_TOTAL_JUDGE_ROUND_TRANSFER and is not round-calibrated.",
        "simulator_phase_share_note": "Simulator exposure only; UFCStats has no authoritative historical phase-time denominator.",
    }
    invariants = status(dict(inv), replay_mismatch)
    targets = json.loads(EVENT_CLOCK_V2_HISTORICAL_TARGETS_PATH.read_text())
    verify_frozen_targets(targets)
    comparisons = evaluate({**metrics, **invariants["counts"]}, targets)
    identity = {
        "cohort_version": event_name or COHORT_VERSION,
        "cohort_manifest_digest": artifact_digest(EVENT_CLOCK_V2_COHORT_MANIFEST_PATH),
        "cohort_split": split,
        "fight_count": len(cohort),
        "paths_per_fight": paths_per_fight,
        "seed_set_version": SEED_SET_VERSION,
        "ko_kd_architecture": ko_kd_architecture.value,
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
    p.add_argument("--event-name")
    p.add_argument(
        "--ko-kd-architecture",
        type=KOKDArchitecture,
        choices=tuple(KOKDArchitecture),
        default=KOKDArchitecture.EMPIRICAL_EVENT2,
    )
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
                ko_kd_architecture=a.ko_kd_architecture,
                event_name=a.event_name,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
