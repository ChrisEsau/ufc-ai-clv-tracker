#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES
from pipeline.research.prefight_strength_elo import build_bouts, run_elo
from pipeline.research.elo_fsr_brain_one_fight_shadow import _apply_fsr, _scale_clock_row, _target_indices

FIGHT_IDS = [
    "419fff06f338f5c6", "58ffa2dac4f2e7d0", "5c69b019e6deee41",
    "5d2eedd05081ed23", "20d74ed23d3e9b3a", "44cfbb8c3c356c65",
    "b0474597b2c60482", "b23a1a5d35eb438a", "33afdd7ad43a2756",
    "7208e40818401e88",
]
PATHS = 500
DOMAIN_SHARE = 0.20
OUT = Path("data/diagnostics/elo_fsr_brain_10fight_adjusted_only")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_locked(fight_id: str, bundle_dir: Path) -> dict:
    subprocess.run([
        "python", "-m", "pipeline.research.locked_brain_mc",
        "--fight-id", fight_id, "--paths", str(PATHS), "--bundle-dir", str(bundle_dir),
    ], check=True)
    src = Path("data/research/locked_brain_mc") / fight_id / "run" / "sim" / "results.json"
    if not src.is_file():
        raise FileNotFoundError(src)
    dest = OUT / "raw" / f"{fight_id}_adjusted_results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return json.loads(src.read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet("data/master/ufc_master.parquet")
    bouts = build_bouts(source)
    elo_fights, _ = run_elo(bouts, base_rating=1000.0, k_factor=170.0)
    by_id = elo_fights.set_index("bout_id", drop=False)

    all_rows = []
    audit_rows = []
    baseline_bundle = Path(DEFAULT_BUNDLE_DIR)

    for n, fight_id in enumerate(FIGHT_IDS, start=1):
        if fight_id not in by_id.index:
            raise RuntimeError(f"fight missing from Elo history: {fight_id}")
        erow = by_id.loc[fight_id]
        if isinstance(erow, pd.DataFrame):
            erow = erow.iloc[0]
        fighter_a = str(erow.red_fighter)
        fighter_b = str(erow.blue_fighter)
        elo_a = float(erow.red_pre_rating)
        elo_b = float(erow.blue_pre_rating)

        elo_delta = elo_a - elo_b
        matchup_log_odds = math.log(10.0) * elo_delta / 400.0
        shift_a = 0.5 * matchup_log_odds * DOMAIN_SHARE
        shift_b = -0.5 * matchup_log_odds * DOMAIN_SHARE

        adjusted_bundle = OUT / "work" / fight_id / "adjusted_bundle"
        if adjusted_bundle.exists():
            shutil.rmtree(adjusted_bundle)
        adjusted_bundle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(baseline_bundle, adjusted_bundle)

        local_audit = []
        ewm_path = adjusted_bundle / FILES["ewm_fsr"]
        ewm = pd.read_parquet(ewm_path)
        ia, ib = _target_indices(ewm, fight_id, fighter_a, fighter_b)
        _apply_fsr(ewm, ia, fighter_a, shift_a, local_audit)
        _apply_fsr(ewm, ib, fighter_b, shift_b, local_audit)
        ewm.to_parquet(ewm_path, index=False)

        ko_path = adjusted_bundle / FILES["ko_prefight"]
        ko = pd.read_parquet(ko_path)
        kia, kib = _target_indices(ko, fight_id, fighter_a, fighter_b)
        _scale_clock_row(ko, kia, fighter_a, shift_a, 2.0, "ko_clock", local_audit)
        _scale_clock_row(ko, kib, fighter_b, shift_b, 2.0, "ko_clock", local_audit)
        ko.to_parquet(ko_path, index=False)

        sub_path = adjusted_bundle / FILES["sub_prefight"]
        sub = pd.read_parquet(sub_path)
        sia, sib = _target_indices(sub, fight_id, fighter_a, fighter_b)
        _scale_clock_row(sub, sia, fighter_a, shift_a, 1.0, "sub_clock", local_audit)
        _scale_clock_row(sub, sib, fighter_b, shift_b, 1.0, "sub_clock", local_audit)
        sub.to_parquet(sub_path, index=False)

        manifest_path = adjusted_bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key, path in (("ewm_fsr", ewm_path), ("ko_prefight", ko_path), ("sub_prefight", sub_path)):
            manifest["files"][key]["sha256"] = sha256(path)
        manifest["research_shadow"] = {
            "type": "elo_full_active_brain_state_10fight_adjusted_only",
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
            "baseline_rerun": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        print(f"[{n}/{len(FIGHT_IDS)}] {fighter_a} vs {fighter_b} | Elo {elo_a:.1f}-{elo_b:.1f}")
        result = run_locked(fight_id, adjusted_bundle)
        summary = result.get("summary", [])
        if len(summary) != 2:
            raise RuntimeError(f"expected two summary rows for {fight_id}, got {len(summary)}")
        total = sum(int(x.get("wins", 0)) for x in summary)
        if total != PATHS:
            raise RuntimeError(f"path total mismatch for {fight_id}: {total}")
        for x in summary:
            all_rows.append({
                "fight_id": fight_id,
                "fighter": x["fighter"],
                "opponent": fighter_b if x["fighter"] == fighter_a else fighter_a,
                "elo": elo_a if x["fighter"] == fighter_a else elo_b,
                "opponent_elo": elo_b if x["fighter"] == fighter_a else elo_a,
                "elo_delta_fighter_minus_opp": (elo_a-elo_b) if x["fighter"] == fighter_a else (elo_b-elo_a),
                "domain_share": DOMAIN_SHARE,
                "domain_shift": shift_a if x["fighter"] == fighter_a else shift_b,
                "adjusted_ml": x["ml_probability"],
                "adjusted_ko_tko": x["ko_tko_probability"],
                "adjusted_sub": x["submission_probability"],
                "adjusted_dec": x["decision_probability"],
                "wins": x["wins"],
                "paths": PATHS,
            })
        for a in local_audit:
            audit_rows.append({"fight_id": fight_id, **a})

        shutil.rmtree(OUT / "work" / fight_id, ignore_errors=True)

    pd.DataFrame(all_rows).to_csv(OUT / "elo_adjusted_10fight_results.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "elo_adjusted_10fight_input_audit.csv", index=False)
    (OUT / "run_manifest.json").write_text(json.dumps({
        "fight_ids": FIGHT_IDS,
        "paths_per_fight": PATHS,
        "domain_share": DOMAIN_SHARE,
        "base_rating": 1000.0,
        "k_factor": 170.0,
        "baseline_rerun": False,
        "production_changed": False,
    }, indent=2) + "\n")
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
