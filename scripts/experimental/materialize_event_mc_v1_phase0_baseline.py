"""Materialize the frozen current-simulator EVENT MC V1 Phase 0 baseline.

This is observational orchestration only: it calls the existing aligned FSR-32
cohort and ``StaticFSRMCFullFightV1`` without changing simulator behavior.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_2026_baseline_age_power_same_decay as age_power


FSR_PATH = Path("data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet")
OUT_DIR = Path("data/experimental/event_mc_v1_baseline")
TRACE_SEEDS = (7, 17, 20260811)
MATCHUP_ROOT_SEED = 20260811
COHORT_ROOT_SEED = 20260810
FIXTURES = (
    ("Rob Font", "Raul Rosas Jr.", "2026-03-07"),
    ("Derrick Lewis", "Chris Daukaus", "2021-12-18"),
    ("Max Holloway", "Calvin Kattar", "2021-01-16"),
    ("Charles Oliveira", "Dustin Poirier", "2021-12-11"),
    ("Merab Dvalishvili", "Petr Yan", "2023-03-11"),
)
STAT_FIELDS = (
    "sig_att", "sig_landed", "td_att", "td_landed", "control_seconds",
    "ground_control_seconds", "clinch_control_seconds", "sub_att", "reversals",
    "knockdowns_scored",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def scheduled_rounds(row: pd.Series) -> int:
    value = pd.to_numeric(pd.Series([row.get("total_rounds")]), errors="coerce").iloc[0]
    return 5 if pd.notna(value) and int(value) == 5 else 3


def resolve_fixtures(cohort: pd.DataFrame, pairs: dict[str, tuple[pd.Series, pd.Series]]):
    resolved = []
    for left, right, date in FIXTURES:
        matches = []
        for _, row in cohort.iterrows():
            pair = pairs.get(str(row["bout_id"]))
            if pair is None or pd.Timestamp(row["event_date"]).date().isoformat() != date:
                continue
            names = {norm(base._display_name(pair[0])), norm(base._display_name(pair[1]))}
            if names == {norm(left), norm(right)}:
                matches.append((row, pair[0], pair[1]))
        if len(matches) != 1:
            raise RuntimeError(f"fixture resolution expected one match for {left} vs {right} ({date}); got {len(matches)}")
        resolved.append(matches[0])
    return resolved


def run_path(row: pd.Series, red: pd.Series, blue: pd.Series, seed: int):
    return full.StaticFSRMCFullFightV1(
        red, blue, rounds=scheduled_rounds(row), seed=int(seed),
        red_age=float(row["r_age"]), blue_age=float(row["b_age"]),
    ).run()


def path_record(path: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "winner_corner": "red" if int(path.winner) == 0 else "blue",
        "method": str(path.method),
        "finish_round": int(path.finish.round) if path.finish is not None else None,
        "finish_segment": int(path.finish.segment) if path.finish is not None else None,
        "events": path.events,
    }
    for index, side in ((0, "red"), (1, "blue")):
        stats = path.stats[index]
        for field in STAT_FIELDS:
            record[f"{side}_{field}"] = int(getattr(stats, field, 0))
        for phase in ("DISTANCE", "CLINCH", "GROUND"):
            record[f"{side}_{phase.lower()}_segments"] = int(stats.phase_segments.get(phase, 0))
    return record


def summarize_paths(paths: list[Any]) -> dict[str, Any]:
    n = len(paths)
    out: dict[str, Any] = {
        "paths": n,
        "p_red_win": sum(int(p.winner) == 0 for p in paths) / n,
        "p_blue_win": sum(int(p.winner) == 1 for p in paths) / n,
    }
    for method in ("KO/TKO", "SUB", "DEC"):
        out[f"p_{method.lower().replace('/', '_')}"] = sum(p.method == method for p in paths) / n
    finishes = [p.finish for p in paths if p.finish is not None]
    out["mean_finish_round"] = float(np.mean([f.round for f in finishes])) if finishes else np.nan
    for index, side in ((0, "red"), (1, "blue")):
        for field in STAT_FIELDS:
            out[f"mean_{side}_{field}"] = float(np.mean([getattr(p.stats[index], field, 0) for p in paths]))
        phase_means = {}
        for phase in ("DISTANCE", "CLINCH", "GROUND"):
            value = float(np.mean([p.stats[index].phase_segments.get(phase, 0) for p in paths]))
            out[f"mean_{side}_{phase.lower()}_segments"] = value
            phase_means[phase] = value
        total = sum(phase_means.values()) or 1.0
        for phase, value in phase_means.items():
            out[f"mean_{side}_{phase.lower()}_occupancy"] = value / total
    td_att = out["mean_red_td_att"] + out["mean_blue_td_att"]
    out["td_success_rate"] = (out["mean_red_td_landed"] + out["mean_blue_td_landed"]) / td_att if td_att else np.nan
    return out


def fixture_identity(row: pd.Series, red: pd.Series, blue: pd.Series) -> dict[str, Any]:
    return {
        "bout_id": str(row["bout_id"]), "event_date": pd.Timestamp(row["event_date"]).date().isoformat(),
        "red_name": base._display_name(red), "blue_name": base._display_name(blue),
        "red_fighter_id": str(red["fighter_id"]), "blue_fighter_id": str(blue["fighter_id"]),
        "red_age": float(row["r_age"]), "blue_age": float(row["b_age"]),
        "scheduled_rounds": scheduled_rounds(row),
    }


def main() -> None:
    if not FSR_PATH.is_file():
        raise FileNotFoundError(FSR_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    age_power._install_uniform_physical_age_layer()
    cohort, pairs = cohort32.build_aligned_cohort()
    # The aligned helper intentionally keeps a compact metadata projection. Add
    # scheduled rounds from the same authoritative master loader without
    # changing cohort membership or ordering.
    master = modern._load_master(modern.MASTER_PATH)
    rounds = master[["fight_id", "total_rounds"]].rename(columns={"fight_id": "bout_id"})
    rounds["bout_id"] = rounds["bout_id"].astype(str)
    cohort = cohort.merge(rounds, on="bout_id", how="left", validate="one_to_one", sort=False)
    fixtures = resolve_fixtures(cohort, pairs)

    trace_path = OUT_DIR / "single_path_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for row, red, blue in fixtures:
            identity = fixture_identity(row, red, blue)
            for seed in TRACE_SEEDS:
                record = {**identity, "seed": seed, **path_record(run_path(row, red, blue, seed))}
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    matchup_rows = []
    fixture_manifest = []
    matchup_seeds = np.random.default_rng(MATCHUP_ROOT_SEED).integers(1, np.iinfo(np.int32).max, 1000, dtype=np.int64)
    for row, red, blue in fixtures:
        identity = fixture_identity(row, red, blue)
        paths = [run_path(row, red, blue, int(seed)) for seed in matchup_seeds]
        result = {**identity, **summarize_paths(paths)}
        if str(row["bout_id"]) == "bed89a91da9d04c1":
            traits = ("wrestling_entry", "wrestling_conversion", "td_defense", "control_imposition",
                      "control_resistance", "distance_striking_pressure", "clinch_striking_pressure")
            for side, fighter in (("red", red), ("blue", blue)):
                for trait in traits:
                    result[f"{side}_{trait}"] = float(fighter[trait])
                result[f"{side}_legacy_wrestling_pref"] = (
                    0.75 * float(fighter["wrestling_entry"])
                    + 0.25 * float(fighter["control_imposition"])
                    - 0.50 * float(fighter["distance_striking_pressure"])
                    - 0.50 * float(fighter["clinch_striking_pressure"])
                )
        matchup_rows.append(result)
        fixture_manifest.append(identity)
    matchup_path = OUT_DIR / "matchup_summary.csv"
    pd.DataFrame(matchup_rows).to_csv(matchup_path, index=False)

    cohort_seeds = np.random.default_rng(COHORT_ROOT_SEED).integers(
        1, np.iinfo(np.int32).max, 200 * 10, dtype=np.int64
    ).reshape(200, 10)
    cohort_rows = []
    for bout_index, (_, row) in enumerate(cohort.iloc[:200].iterrows()):
        red, blue = pairs[str(row["bout_id"])]
        paths = [run_path(row, red, blue, int(seed)) for seed in cohort_seeds[bout_index]]
        summary = summarize_paths(paths)
        actual_red_win = int(str(row["winner_id"]) == str(row["r_id"]))
        summary.update(fixture_identity(row, red, blue))
        summary.update({"actual_red_win": actual_red_win,
                        "winner_correct": int((summary["p_red_win"] >= 0.5) == bool(actual_red_win)),
                        "winner_brier": (summary["p_red_win"] - actual_red_win) ** 2})
        cohort_rows.append(summary)
    cohort_path = OUT_DIR / "cohort_200_summary.csv"
    pd.DataFrame(cohort_rows).to_csv(cohort_path, index=False)

    full_method_path = OUT_DIR / "full_method_baseline.csv"
    pd.DataFrame([{
        "cohort_fights": len(cohort), "paths_per_fight": 10,
        "historical_sub_rate": 0.1623, "simulated_sub_rate": 0.1649,
        "neutral_submission_finish_probability": full.CALIBRATED_SUBMISSION_NEUTRAL_RATE,
        "historical_submission_attempts_per_fight": 0.5655,
        "simulated_submission_attempts_per_path": 0.4994,
        "historical_any_submission_attempt_rate": 0.3502,
        "simulated_any_submission_attempt_rate": 0.3508,
        "status": "materialized frozen existing diagnostic observations; no physics changes",
    }]).to_csv(full_method_path, index=False)

    fsr = pd.read_parquet(FSR_PATH)
    outputs = [trace_path, matchup_path, cohort_path, full_method_path]
    manifest = {
        "repository": "ChrisEsau/ufc-ai-clv-tracker",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python_version": platform.python_version(), "architecture_revision": "v0.3",
        "fsr_32": {"path": str(FSR_PATH), "sha256": sha256(FSR_PATH), "bytes": FSR_PATH.stat().st_size,
                   "rows": len(fsr), "columns": len(fsr.columns), "latest_date": str(pd.to_datetime(fsr["date"]).max().date())},
        "simulator": "scripts.experimental.fsr_static_mc_ko_sub_decision_v1.StaticFSRMCFullFightV1",
        "cohort_builder": "scripts.experimental.fsr_32_historical_cohort.build_aligned_cohort",
        "fixtures": fixture_manifest, "fixture_substitutions": [], "fixture_omissions": [],
        "single_path_seeds": list(TRACE_SEEDS),
        "matchup": {"root_seed": MATCHUP_ROOT_SEED, "paths": 1000,
                    "seed_method": "numpy.random.default_rng(root).integers(1, int32_max, paths)"},
        "cohort": {"root_seed": COHORT_ROOT_SEED, "bouts": 200, "paths_per_bout": 10,
                   "ordering": "first 200 rows from build_aligned_cohort() stable order"},
        "age_rule": "physical traits unchanged through age 30; -2 points/year after 30",
        "finish_recovery_candidate": "current KO locks + 34% neutral submission conversion + current global recovery",
        "metric_definitions": "counts are current simulator FighterStats per path; phase occupancy is segment share; control is seconds",
        "outputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
