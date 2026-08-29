#!/usr/bin/env python3
"""Research-only 10-fight shadow: apply the full Glicko matchup signal to every active Brain input.

This is the undiluted counterpart to glicko_fsr_brain_10_fight_shadow.py.
The standard prefight Glicko central-rating gap is converted to natural-log odds,
split symmetrically between the two fighters, and applied at 100% strength to
all simulator-active FSR fields plus both KO and submission time clocks.

Production and locked Brain mechanics are unchanged. Rating deviation is audited
but is not used in the transform in this experiment.
"""
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
from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import run as run_glicko

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
DOMAIN_SHARE = 1.0
ROOT = Path("data/diagnostics/glicko_full_strength_all_inputs_10fight_shadow")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight-id", required=True, choices=FIGHT_IDS)
    args = ap.parse_args()
    fid = args.fight_id

    bouts = build_bouts(pd.read_parquet("data/master/ufc_master.parquet"))
    gf = run_glicko(bouts)
    target = gf[gf.bout_id.astype(str).eq(fid)]
    if len(target) != 1:
        raise RuntimeError(f"expected one Glicko row for {fid}, found {len(target)}")

    row = target.iloc[0]
    a = str(row.red_fighter)
    b = str(row.blue_fighter)
    ra = float(row.red_pre_rating)
    rb = float(row.blue_pre_rating)
    rda = float(row.red_pre_rd)
    rdb = float(row.blue_pre_rd)

    delta = ra - rb
    matchup_log_odds = math.log(10.0) * delta / 400.0
    shift_a = +0.5 * matchup_log_odds * DOMAIN_SHARE
    shift_b = -0.5 * matchup_log_odds * DOMAIN_SHARE

    out = ROOT / fid
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "adjusted_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(Path(DEFAULT_BUNDLE_DIR), bundle)
    audit: list[dict] = []

    # Every active EWM/FSR input consumed by the existing full-state transform.
    ewm_path = bundle / FILES["ewm_fsr"]
    ewm = pd.read_parquet(ewm_path)
    ia, ib = _target_indices(ewm, fid, a, b)
    _apply_fsr(ewm, ia, a, shift_a, audit)
    _apply_fsr(ewm, ib, b, shift_b, audit)
    ewm.to_parquet(ewm_path, index=False)

    # Both finish clocks use the same full-strength matchup signal.
    ko_path = bundle / FILES["ko_prefight"]
    ko = pd.read_parquet(ko_path)
    ia, ib = _target_indices(ko, fid, a, b)
    _scale_clock_row(ko, ia, a, shift_a, 2.0, "ko_clock", audit)
    _scale_clock_row(ko, ib, b, shift_b, 2.0, "ko_clock", audit)
    ko.to_parquet(ko_path, index=False)

    sub_path = bundle / FILES["sub_prefight"]
    sub = pd.read_parquet(sub_path)
    ia, ib = _target_indices(sub, fid, a, b)
    _scale_clock_row(sub, ia, a, shift_a, 1.0, "sub_clock", audit)
    _scale_clock_row(sub, ib, b, shift_b, 1.0, "sub_clock", audit)
    sub.to_parquet(sub_path, index=False)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, path in (
        ("ewm_fsr", ewm_path),
        ("ko_prefight", ko_path),
        ("sub_prefight", sub_path),
    ):
        manifest["files"][key]["sha256"] = sha256(path)
    manifest["research_shadow"] = {
        "type": "glicko_central_rating_full_strength_all_active_brain_inputs",
        "fight_id": fid,
        "fighter_a": a,
        "fighter_b": b,
        "glicko_a": ra,
        "glicko_b": rb,
        "rd_a": rda,
        "rd_b": rdb,
        "rating_delta": delta,
        "matchup_log_odds": matchup_log_odds,
        "domain_share": DOMAIN_SHARE,
        "fighter_a_shift": shift_a,
        "fighter_b_shift": shift_b,
        "rd_used_in_transform": False,
        "touches_all_active_fsr_inputs": True,
        "touches_ko_finish_clock": True,
        "touches_submission_finish_clock": True,
        "production_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    adjusted = run_locked(fid, bundle, PATHS, out / "adjusted_results.json")
    pd.DataFrame(audit).assign(fight_id=fid).to_csv(
        out / "glicko_full_strength_active_input_adjustments.csv", index=False
    )
    summary = {
        "fight_id": fid,
        "fighter_a": a,
        "fighter_b": b,
        "actual_winner": row.winner,
        "glicko_a": ra,
        "glicko_b": rb,
        "rd_a": rda,
        "rd_b": rdb,
        "rating_delta": delta,
        "matchup_log_odds": matchup_log_odds,
        "domain_share": DOMAIN_SHARE,
        "shift_a": shift_a,
        "shift_b": shift_b,
        "paths": PATHS,
        "adjusted": adjusted,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    shutil.rmtree(bundle)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
