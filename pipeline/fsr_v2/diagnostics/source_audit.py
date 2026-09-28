"""Measure required FSR V2 schema and edge cases without mutating sources."""

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import FSR_V2_DIAGNOSTICS_DIR, MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.sources.master import MASTER_COLUMNS, load_master
from pipeline.fsr_v2.sources.round_stats import ROUND_COLUMNS, build_paired_rounds, load_round_stats


def audit_sources(round_path: Path = ROUND_STATS_PATH, master_path: Path = MASTER_PATH) -> dict:
    config = FSRV2Config()
    raw = load_round_stats(round_path)
    master = load_master(master_path)
    paired = build_paired_rounds(raw, master, config)
    group_sizes = raw.groupby(["fight_id", "round"]).size()
    target_attempts = paired["head_attempted"] + paired["body_attempted"] + paired["leg_attempted"]
    phase_attempts = paired["distance_attempted"] + paired["clinch_attempted"] + paired["ground_attempted"]
    zero_td_control = (paired["ctrl_sec"] > 0) & (paired["td_landed"] == 0)
    meaningful_zero_td = (paired["ctrl_sec"] >= config.zero_td_control_threshold_seconds) & (paired["td_landed"] == 0)
    zero_control_ground = (paired["ground_side_attempted"] > 0) & (paired["ground_exposure_seconds"] == 0)
    submission_fights = paired.groupby(["fight_id", "fighter_id"], as_index=False).agg(
        submission_finish=("submission_finish", "max"), sub_att=("sub_att", "sum")
    )
    submission_winner_rows = submission_fights["submission_finish"]
    audit = {
        "round_source": str(round_path), "master_source": str(master_path),
        "round_shape": list(raw.shape), "master_shape": list(master.shape),
        "round_columns_used": ROUND_COLUMNS, "master_columns_used": MASTER_COLUMNS,
        "round_null_rates": {c: float(raw[c].isna().mean()) for c in ROUND_COLUMNS},
        "master_null_rates": {c: float(master[c].isna().mean()) for c in MASTER_COLUMNS},
        "round_date_min": str(raw.event_date.min().date()), "round_date_max": str(raw.event_date.max().date()),
        "fighter_round_rows": len(raw), "fights": int(raw.fight_id.nunique()),
        "fight_round_groups": len(group_sizes), "non_reciprocal_groups": int((group_sizes != 2).sum()),
        "duplicate_fighter_round_rows": int(raw.duplicated(["fight_id", "round", "fighter_id"]).sum()),
        "round_elapsed_min": float(paired.round_elapsed_seconds.min()),
        "round_elapsed_max": float(paired.round_elapsed_seconds.max()),
        "final_round_time_semantics": paired.match_time_interpretation.value_counts().to_dict(),
        "combined_control_min": int(paired.combined_control_seconds_raw.min()),
        "combined_control_max": int(paired.combined_control_seconds_raw.max()),
        "combined_control_exceeds_elapsed_fighter_rows": int((paired.combined_control_seconds_raw > paired.round_elapsed_seconds).sum()),
        "combined_control_exceeds_elapsed_rounds": int((paired.combined_control_seconds_raw > paired.round_elapsed_seconds).sum() // 2),
        "control_positive_zero_td_fighter_rounds": int(zero_td_control.sum()),
        "control_positive_zero_td_seconds": int(paired.loc[zero_td_control, "ctrl_sec"].sum()),
        "meaningful_control_zero_td_fighter_rounds": int(meaningful_zero_td.sum()),
        "zero_td_fallback_threshold_seconds": config.zero_td_control_threshold_seconds,
        "ground_clinch_attempts_positive_zero_control_fighter_rounds": int(zero_control_ground.sum()),
        "ground_clinch_attempts_in_zero_control_rows": int(paired.loc[zero_control_ground, "ground_side_attempted"].sum()),
        "zero_control_fallback_seconds": config.zero_control_ground_fallback_seconds,
        "submission_attempt_distribution": {str(k): int(v) for k, v in paired.sub_att.value_counts().sort_index().items()},
        "submission_attempt_positive_zero_control_rows": int(((paired.sub_att > 0) & (paired.ground_exposure_seconds == 0)).sum()),
        "submission_finish_fighter_rows": int(submission_winner_rows.sum()),
        "submission_finish_zero_attempt_fighter_fights": int((submission_winner_rows & (submission_fights.sub_att == 0)).sum()),
        "target_attempt_sum_mismatch_rows": int((target_attempts != paired.sig_str_attempted).sum()),
        "phase_attempt_sum_mismatch_rows": int((phase_attempts != paired.sig_str_attempted).sum()),
        "leg_attempts_exceed_distance_attempts_rows": int((paired.leg_attempted > paired.distance_attempted).sum()),
        "leg_attempts_positive_zero_distance_rows": int(((paired.leg_attempted > 0) & (paired.distance_attempted == 0)).sum()),
        "authoritative_round_source_sha256_before": _sha256(round_path),
    }
    audit["authoritative_round_source_sha256_after"] = _sha256(round_path)
    audit["authoritative_round_source_unchanged"] = audit["authoritative_round_source_sha256_before"] == audit["authoritative_round_source_sha256_after"]
    return audit


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FSR_V2_DIAGNOSTICS_DIR / "source_audit.json")
    args = parser.parse_args()
    audit = audit_sources()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
