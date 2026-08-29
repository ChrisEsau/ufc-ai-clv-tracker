#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from pipeline.research.elo_fsr_brain_one_fight_shadow import (
    _apply_fsr,
    _scale_clock_row,
    _target_indices,
    run_locked,
)
from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES
from pipeline.research.prefight_strength_elo import build_bouts, run_elo

FIGHT_IDS = [
    "419fff06f338f5c6",
    "58ffa2dac4f2e7d0",
    "5c69b019e6deee41",
    "5d2eedd05081ed23",
    "20d74ed23d3e9b3a",
    "44cfbb8c3c356c65",
    "b0474597b2c60482",
    "b23a1a5d35eb438a",
    "33afdd7ad43a2756",
    "7208e40818401e88",
]
PATHS = 500
DOMAIN_SHARE = 0.20
BASE_RATING = 1000.0
K_FACTOR = 170.0
ROOT = Path("data/diagnostics/elo_fsr_brain_10_fight_shadow")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_one(fight_id: str, by_id: pd.DataFrame) -> dict:
    if fight_id not in by_id.index:
        raise RuntimeError(f"fight {fight_id} missing from Elo history")
    row = by_id.loc[fight_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    fighter_a = str(row.red_fighter)
    fighter_b = str(row.blue_fighter)
    elo_a = float(row.red_pre_rating)
    elo_b = float(row.blue_pre_rating)
    elo_delta = elo_a - elo_b
    matchup_log_odds = math.log(10.0) * elo_delta / 400.0
    shift_a = 0.5 * matchup_log_odds * DOMAIN_SHARE
    shift_b = -0.5 * matchup_log_odds * DOMAIN_SHARE

    fight_out = ROOT / fight_id
    fight_out.mkdir(parents=True, exist_ok=True)
    adjusted_bundle = fight_out / "adjusted_bundle"
    if adjusted_bundle.exists():
        shutil.rmtree(adjusted_bundle)
    shutil.copytree(Path(DEFAULT_BUNDLE_DIR), adjusted_bundle)

    audit: list[dict] = []
    ewm_path = adjusted_bundle / FILES["ewm_fsr"]
    ewm = pd.read_parquet(ewm_path)
    ia, ib = _target_indices(ewm, fight_id, fighter_a, fighter_b)
    _apply_fsr(ewm, ia, fighter_a, shift_a, audit)
    _apply_fsr(ewm, ib, fighter_b, shift_b, audit)
    ewm.to_parquet(ewm_path, index=False)

    ko_path = adjusted_bundle / FILES["ko_prefight"]
    ko = pd.read_parquet(ko_path)
    kia, kib = _target_indices(ko, fight_id, fighter_a, fighter_b)
    _scale_clock_row(ko, kia, fighter_a, shift_a, 2.0, "ko_clock", audit)
    _scale_clock_row(ko, kib, fighter_b, shift_b, 2.0, "ko_clock", audit)
    ko.to_parquet(ko_path, index=False)

    sub_path = adjusted_bundle / FILES["sub_prefight"]
    sub = pd.read_parquet(sub_path)
    sia, sib = _target_indices(sub, fight_id, fighter_a, fighter_b)
    _scale_clock_row(sub, sia, fighter_a, shift_a, 1.0, "sub_clock", audit)
    _scale_clock_row(sub, sib, fighter_b, shift_b, 1.0, "sub_clock", audit)
    sub.to_parquet(sub_path, index=False)

    manifest_path = adjusted_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, path in (("ewm_fsr", ewm_path), ("ko_prefight", ko_path), ("sub_prefight", sub_path)):
        manifest["files"][key]["sha256"] = sha256(path)
    manifest["research_shadow"] = {
        "type": "elo_full_active_brain_state_10fight_pilot",
        "fight_id": fight_id,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_delta": elo_delta,
        "matchup_log_odds": matchup_log_odds,
        "domain_share": DOMAIN_SHARE,
        "fighter_a_domain_shift": shift_a,
        "fighter_b_domain_shift": shift_b,
        "touches_ko_finish_hazard": True,
        "touches_submission_finish_hazard": True,
        "production_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{fighter_a} vs {fighter_b} | Elo {elo_a:.1f}-{elo_b:.1f}", flush=True)
    adjusted = run_locked(fight_id, adjusted_bundle, PATHS, fight_out / "adjusted_results.json")
    pd.DataFrame(audit).assign(fight_id=fight_id).to_csv(fight_out / "elo_active_input_adjustments.csv", index=False)
    summary = {
        "fight_id": fight_id,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "actual_winner": row.winner,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_delta": elo_delta,
        "domain_share": DOMAIN_SHARE,
        "shift_a": shift_a,
        "shift_b": shift_b,
        "paths": PATHS,
        "adjusted": adjusted,
    }
    (fight_out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    shutil.rmtree(adjusted_bundle)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight-id", choices=FIGHT_IDS)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet("data/master/ufc_master.parquet")
    bouts = build_bouts(source)
    elo_fights, _ = run_elo(bouts, base_rating=BASE_RATING, k_factor=K_FACTOR)
    by_id = elo_fights.set_index("bout_id", drop=False)
    ids = [args.fight_id] if args.fight_id else FIGHT_IDS
    summaries = [run_one(fid, by_id) for fid in ids]
    print(json.dumps({"fight_count": len(summaries), "fight_ids": ids}, indent=2))


if __name__ == "__main__":
    main()
