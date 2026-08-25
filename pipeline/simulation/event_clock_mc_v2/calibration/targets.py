"""Freeze empirical targets once; simulation candidates only read this file."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from . import EXPECTED_TARGET_DIGEST, TARGET_VERSION


def build_targets(manifest: pd.DataFrame) -> dict:
    cohort = manifest.loc[manifest.cohort_split.eq("calibration")]
    ids = set(cohort.bout_id.astype(str))
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    rounds = rounds[rounds.fight_id.astype(str).isin(ids)]
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    master = master[master.fight_id.astype(str).isin(ids)]
    durations = pd.to_numeric(master.match_time_sec, errors="coerce").sum()
    fighter_seconds = 2 * durations

    def per15(column):
        return float(
            pd.to_numeric(rounds[column], errors="coerce").fillna(0).sum()
            * 900
            / fighter_seconds
        )

    method = master.method.astype(str).str.lower()
    metrics = {
        "standing_attempts_per_fighter_15": per15("sig_str_attempted")
        - per15("clinch_attempted")
        - per15("ground_attempted"),
        "clinch_strikes_per_fighter_15": per15("clinch_attempted"),
        "ground_strikes_per_fighter_15": per15("ground_attempted"),
        "td_attempts_per_fighter_15": per15("td_attempted"),
        "td_landed_per_fighter_15": per15("td_landed"),
        "submissions_per_fighter_15": per15("sub_att"),
        "knockdowns_per_fight": float(rounds.kd.sum() / len(master)),
        "ko_tko_fight_share": float(method.str.contains("ko").mean()),
        "submission_fight_share": float(method.str.contains("sub").mean()),
        "decision_fight_share": float(method.str.contains("dec").mean()),
        "mean_fight_duration_seconds": float(
            pd.to_numeric(master.match_time_sec).mean()
        ),
    }
    bands = {}
    for key, target in metrics.items():
        tolerance = max(
            abs(target) * (0.20 if "share" not in key else 0.15),
            0.02 if "share" in key else 0.05,
        )
        bands[key] = {
            "target": target,
            "warn_low": target - tolerance,
            "warn_high": target + tolerance,
            "fail_low": target - 2 * tolerance,
            "fail_high": target + 2 * tolerance,
        }
    bands.update(
        {
            k: {"target": 0, "required": 0}
            for k in (
                "illegal_cross_phase_actions",
                "timeline_exposure_mismatch",
                "post_finish_events",
                "invalid_state_transitions",
                "nan_or_non_finite_state",
                "impossible_physiology_state",
                "deterministic_replay_mismatch",
            )
        }
    )
    # Canonical V2 currently fixes submission conversion at zero. Preserve the
    # empirical value as evidence, but do not present it as a reachable gate.
    bands["submission_fight_share"]["diagnostic_only"] = True
    payload = {
        "target_schema_version": TARGET_VERSION,
        "historical_comparator_split": "calibration",
        "cohort_bout_ids": sorted(ids),
        "metrics": metrics,
        "metric_groups": {
            "structural_targets": [
                "standing_attempts_per_fighter_15",
                "clinch_strikes_per_fighter_15",
                "ground_strikes_per_fighter_15",
                "td_attempts_per_fighter_15",
                "td_landed_per_fighter_15",
                "submissions_per_fighter_15",
            ],
            "physiology_targets": [
                "knockdowns_per_fight",
                "ko_tko_fight_share",
                "decision_fight_share",
                "mean_fight_duration_seconds",
            ],
            "diagnostic_only": ["submission_fight_share"],
            "discrimination_diagnostics": [],
            "predictive_diagnostics": [],
            "invariants": [key for key, band in bands.items() if "required" in band],
        },
        "acceptance_bands": bands,
        "historical_phase_exposure": None,
        "phase_exposure_note": "UFCStats has action-location counts but no authoritative phase-time denominators.",
    }
    payload["target_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return payload


def evaluate(metrics: dict, targets: dict) -> dict:
    result = {}
    for key, band in targets["acceptance_bands"].items():
        if band.get("diagnostic_only"):
            result[key] = {
                "status": "WARN",
                "reason": "diagnostic only: canonical V2 has no approved submission conversion model",
                "value": metrics.get(key),
                **band,
            }
            continue
        if key not in metrics:
            result[key] = {"status": "WARN", "reason": "not available"}
            continue
        value = metrics[key]
        if "required" in band:
            state = "PASS" if value == band["required"] else "FAIL"
        elif band["warn_low"] <= value <= band["warn_high"]:
            state = "PASS"
        elif band["fail_low"] <= value <= band["fail_high"]:
            state = "WARN"
        else:
            state = "FAIL"
        result[key] = {"status": state, "value": value, **band}
    return result


def verify_frozen_targets(targets: dict) -> None:
    """Verify the payload against both self-declared and independently pinned digests."""
    declared = targets.get("target_digest")
    unsigned = dict(targets)
    unsigned.pop("target_digest", None)
    actual = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    if declared != actual or actual != EXPECTED_TARGET_DIGEST:
        raise ValueError(
            f"frozen historical target digest mismatch: declared={declared} "
            f"actual={actual} expected={EXPECTED_TARGET_DIGEST}"
        )
