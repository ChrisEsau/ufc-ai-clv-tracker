"""Measurement-only post-Phase-7C submission exposure decomposition."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..components.actions import ActionAttempt
from ..events import ConsequenceEvent, PrimaryEvent
from ..physiology import PhysiologyOutcome
from ..single_fight import build_engine
from ..submission_finishes import SubmissionFinishOutcome
from .population_validation import METHODS, _fight, build_cohort, normalize_method, observed_duration_seconds


@dataclass
class SubmissionDecompositionSink:
    """Collect sufficient statistics without changing engine state or RNG."""

    attempts_by_round: Counter = field(default_factory=Counter)
    attempts_by_position: Counter = field(default_factory=Counter)
    submission_finishes: int = 0
    submission_finishes_by_position: Counter = field(default_factory=Counter)
    landed_strikes: int = 0
    knockdowns: int = 0
    exposure_seconds: float = 0.0
    ground_seconds: float = 0.0
    ground_control_seconds: Counter = field(default_factory=Counter)

    def on_time_advance(self, dt, before, after):
        self.exposure_seconds += dt
        if before.phase == "ground":
            self.ground_seconds += dt
            if before.ground_controller:
                self.ground_control_seconds[before.ground_controller] += dt

    def on_event(self, event, before, after):
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt):
            if event.payload.action_family == "submission_attempt":
                round_number = int(max(event.timestamp_seconds - 1e-12, 0) // 300) + 1
                position = "top" if before.ground_controller == event.payload.side.value else "bottom"
                self.attempts_by_round[round_number] += 1
                self.attempts_by_position[position] += 1
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, PhysiologyOutcome):
            self.landed_strikes += 1
            self.knockdowns += int(event.payload.knockdown)
        if isinstance(event, ConsequenceEvent) and isinstance(event.payload, SubmissionFinishOutcome):
            self.submission_finishes += int(event.payload.finished)
            self.submission_finishes_by_position[event.payload.position] += int(event.payload.finished)

    def finalize(self):
        return {
            "attempts_by_round": dict(self.attempts_by_round),
            "attempts_by_position": dict(self.attempts_by_position),
            "submission_finishes": self.submission_finishes,
            "submission_finishes_by_position": dict(self.submission_finishes_by_position),
            "landed_strikes": self.landed_strikes,
            "knockdowns": self.knockdowns,
            "exposure_seconds": self.exposure_seconds,
            "ground_seconds": self.ground_seconds,
            "ground_control_seconds": dict(self.ground_control_seconds),
        }


def run(paths=10, start_year=2020, limit=100, seed=20260813,
        output_dir=Path("data/diagnostics/event_mc_v1_phase7d")):
    cohort, fsr = build_cohort(start_year, limit)
    started = time.perf_counter(); rows = []; round_attempts = Counter(); position_attempts = Counter(); position_finishes = Counter()
    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = _fight(historical, fsr)
        for path_index in range(paths):
            sink = SubmissionDecompositionSink()
            result = build_engine(fight, seed + fight_index * 100000 + path_index, sink)[0].run()
            stats = result.sink_result
            attempts = sum(stats["attempts_by_round"].values())
            round_attempts.update(stats["attempts_by_round"]); position_attempts.update(stats["attempts_by_position"])
            position_finishes.update(stats["submission_finishes_by_position"])
            rows.append({
                "fight_index": fight_index, "actual_red": int(historical.winner == historical.r_name),
                "red_win": int(result.state.winner == "red"), "method": result.state.finish_method,
                "seconds": stats["exposure_seconds"], "ground_seconds": stats["ground_seconds"],
                "control_seconds": sum(stats["ground_control_seconds"].values()), "sub_attempts": attempts,
                "sub_finishes": stats["submission_finishes"], "landed": stats["landed_strikes"],
                "kds": stats["knockdowns"],
            })
    frame = pd.DataFrame(rows); exposure = frame.seconds.sum(); ground = frame.ground_seconds.sum()
    methods = frame.method.value_counts(normalize=True); historical_methods = cohort.method.map(normalize_method).value_counts(normalize=True)
    nondec = frame[frame.method != "DEC"]; finish_rounds = (np.maximum(nondec.seconds.to_numpy() - 1e-12, 0) // 300 + 1).astype(int)
    probabilities = frame.groupby("fight_index").red_win.mean().to_numpy(); actual = cohort.apply(lambda row: int(row.winner == row.r_name), axis=1).to_numpy(); safe = np.clip(probabilities, 1e-12, 1 - 1e-12)
    historical_nondec = cohort[cohort.method.map(normalize_method) != "DEC"]
    historical_exposure = float(cohort.apply(observed_duration_seconds, axis=1).sum())
    historical_attempt_total = int((cohort.r_sub_att + cohort.b_sub_att).sum())
    total_attempts = int(frame.sub_attempts.sum()); total_subs = int(frame.sub_finishes.sum())
    summary = {
        "fights": len(cohort), "paths": len(frame), "paths_per_fight": paths,
        "cohort_dates": [str(cohort.event_date.min().date()), str(cohort.event_date.max().date())],
        "historical_method_shares": {method: float(historical_methods.get(method, 0)) for method in METHODS},
        "simulated_method_shares": {method: float(methods.get(method, 0)) for method in METHODS},
        "submission_exposure": {
            "attempts": total_attempts, "attempts_per_path": float(frame.sub_attempts.mean()),
            "attempts_per_15min": float(total_attempts / exposure * 900),
            "paths_with_attempt_share": float((frame.sub_attempts > 0).mean()),
            "attempts_by_round": {str(key): int(value) for key, value in sorted(round_attempts.items())},
            "attempts_by_position": dict(position_attempts),
            "attempts_per_15_ground_minutes": float(total_attempts / ground * 900) if ground else None,
        },
        "historical_submission_exposure": {
            "attempts": historical_attempt_total,
            "attempts_per_fight": float(historical_attempt_total / len(cohort)),
            "attempts_per_15_observed_minutes": float(historical_attempt_total / historical_exposure * 900),
            "fights_with_attempt_share": float(((cohort.r_sub_att + cohort.b_sub_att) > 0).mean()),
            "round_position_and_ground_breakdowns": "Unavailable from the authoritative fight-level master totals; no non-comparable inference was made.",
        },
        "submission_conversion": {"attempts": total_attempts, "finishes": total_subs, "p_sub_given_attempt": float(total_subs / total_attempts) if total_attempts else None},
        "ground_exposure": {"seconds_per_path": float(frame.ground_seconds.mean()), "seconds_per_15min": float(ground / exposure * 900), "control_seconds_per_path": float(frame.control_seconds.mean())},
        "outcomes": {
            "historical_mean_nondecision_finish_time": float(historical_nondec.apply(observed_duration_seconds, axis=1).mean()),
            "simulated_mean_nondecision_finish_time": float(nondec.seconds.mean()),
            "historical_finish_round_shares": {str(round_number): float((historical_nondec.finish_round == round_number).mean()) for round_number in range(1, 6)},
            "simulated_finish_round_shares": {str(round_number): float(np.mean(finish_rounds == round_number)) for round_number in range(1, 6)},
            "winner_accuracy": float(np.mean((probabilities >= .5) == actual)),
            "brier": float(np.mean((probabilities - actual) ** 2)),
            "log_loss": float(-np.mean(actual * np.log(safe) + (1 - actual) * np.log(1 - safe))),
        },
        "guardrails": {"kd_per_path": float(frame.kds.mean()), "kd_per_100_landed": float(frame.kds.sum() / frame.landed.sum() * 100), "kd_per_15min": float(frame.kds.sum() / exposure * 900), "ko_tko_share": float(methods.get("KO_TKO", 0))},
        "mean_fight_duration": float(frame.seconds.mean()),
        "runtime_seconds": time.perf_counter() - started,
    }
    summary["submission_conversion"]["by_position"] = {
        position: {
            "attempts": int(position_attempts[position]),
            "finishes": int(position_finishes[position]),
            "p_sub_given_attempt": float(position_finishes[position] / position_attempts[position]) if position_attempts[position] else None,
        }
        for position in ("top", "bottom")
    }
    summary["submission_exposure"]["attempts_per_15_position_minutes"] = {
        position: float(position_attempts[position] / ground * 900) if ground else None
        for position in ("top", "bottom")
    }
    historical_attempts = float((cohort.r_sub_att + cohort.b_sub_att).mean())
    simulated_attempts = summary["submission_exposure"]["attempts_per_path"]
    historical_sub = summary["historical_method_shares"]["SUB"]; simulated_sub = summary["simulated_method_shares"]["SUB"]
    historical_conversion_proxy = historical_sub / historical_attempts if historical_attempts else None
    simulated_conversion = summary["submission_conversion"]["p_sub_given_attempt"]
    exposure_limited = historical_attempts > 0 and simulated_attempts < historical_attempts * .75
    conversion_limited = historical_conversion_proxy is not None and simulated_conversion is not None and simulated_conversion < historical_conversion_proxy * .75
    classification = "both" if exposure_limited and conversion_limited else ("attempt-exposure limited" if exposure_limited else ("conversion limited" if conversion_limited else "neither clearly limited"))
    summary["interpretation"] = {
        "historical_attempts_per_fight": historical_attempts,
        "historical_conversion_proxy_sub_share_per_attempt": historical_conversion_proxy,
        "simulated_conversion_per_attempt": simulated_conversion,
        "attempt_exposure_ratio": float(simulated_attempts / historical_attempts) if historical_attempts else None,
        "sub_share_ratio": float(simulated_sub / historical_sub) if historical_sub else None,
        "classification": classification,
        "qualification": "The historical conversion value is a descriptive population proxy (SUB fight share divided by recorded attempts/fight), not attempt-level linkage.",
    }
    output_dir.mkdir(parents=True, exist_ok=True); frame.to_csv(output_dir / "path_sufficient_statistics.csv", index=False); (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)); print(json.dumps(summary, indent=2, sort_keys=True)); return frame, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", type=int, default=10); parser.add_argument("--start-year", type=int, default=2020); parser.add_argument("--limit", type=int, default=100); parser.add_argument("--seed", type=int, default=20260813); parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7d")); run(**vars(parser.parse_args()))
