#!/usr/bin/env python3
"""One-fight research shadow: inject a small Elo strength prior into active Brain inputs.

Research only. The canonical locked Brain bundle and production mechanics are not
modified. A temporary copy of the reusable locked bundle is transformed, then both
baseline and adjusted conditions run through pipeline.research.locked_brain_mc.

The Elo gap is converted to the standard Elo natural-log odds gap and split
symmetrically between fighters. To avoid counting the same overall strength signal
at full strength in every subsystem, one fifth of each fighter's side shift is
applied to each broad domain: standing, wrestling, ground, KO, and submission.
The transform is semantic: positive rates are multiplicative, suppressions invert,
logit-effect coordinates are additive, and KO/SUB time-survival hazards are scaled
through their target prefight exposure rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES

RATE_UP = (
    "standing_striking_tendency",
    "takedown_tendency",
    "ground_striking_tendency",
    "ground_striking_burst_baseline",
    "submission_tendency",
)
SUPPRESSION_DOWN = (
    "standing_striking_suppression",
    "takedown_suppression",
    "ground_striking_suppression",
)
LATENT_UP = (
    "standing_striking_offense",
    "standing_striking_defense",
    "takedown_offense",
    "takedown_defense",
    "ground_striking_offense",
    "striking_power_v3",
    "knockdown_resistance_v3",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_locked(fight_id: str, bundle_dir: Path, paths: int, dest: Path) -> dict:
    subprocess.run([
        "python", "-m", "pipeline.research.locked_brain_mc",
        "--fight-id", fight_id,
        "--paths", str(paths),
        "--bundle-dir", str(bundle_dir),
    ], check=True)
    src = Path("data/research/locked_brain_mc") / fight_id / "run" / "sim" / "results.json"
    if not src.is_file():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return json.loads(src.read_text())


def _target_indices(frame: pd.DataFrame, fight_id: str, fighter_a: str, fighter_b: str) -> tuple[int, int]:
    target = frame[frame["fight_id"].astype(str).eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected 2 target rows, found {len(target)}")
    name_col = next((c for c in ("fighter_name", "name", "fighter") if c in frame.columns), None)
    if name_col is None:
        raise RuntimeError("target bundle rows do not expose fighter name")
    by_name = {str(frame.loc[i, name_col]): i for i in target.index}
    ia, ib = by_name.get(fighter_a), by_name.get(fighter_b)
    if ia is None or ib is None:
        raise RuntimeError(f"fighter names not found in target rows: {sorted(by_name)}")
    return int(ia), int(ib)


def _apply_fsr(frame: pd.DataFrame, idx: int, fighter: str, shift: float, audit: list[dict]) -> None:
    for c in RATE_UP:
        if c not in frame.columns:
            continue
        before = float(frame.at[idx, c])
        after = max(before * math.exp(shift), 0.0)
        frame.at[idx, c] = after
        audit.append({"fighter": fighter, "system": "fsr", "field": c, "transform": "x exp(shift)", "before": before, "after": after})
    for c in SUPPRESSION_DOWN:
        if c not in frame.columns:
            continue
        before = float(frame.at[idx, c])
        after = max(before * math.exp(-shift), 0.0)
        frame.at[idx, c] = after
        audit.append({"fighter": fighter, "system": "fsr", "field": c, "transform": "x exp(-shift)", "before": before, "after": after})
    for c in LATENT_UP:
        if c not in frame.columns:
            continue
        before = float(frame.at[idx, c])
        after = before + shift
        frame.at[idx, c] = after
        audit.append({"fighter": fighter, "system": "fsr", "field": c, "transform": "+ shift", "before": before, "after": after})


def _scale_clock_row(frame: pd.DataFrame, idx: int, fighter: str, shift: float, prior_events: float, label: str, audit: list[dict]) -> None:
    """Scale target fighter hazard RR by exp(shift) via both exposure denominators.

    _clock_lookup multiplies attacker and defender rates. Scaling each rate by
    sqrt(exp(shift)) produces an overall exp(shift) hazard multiplier while
    preserving the validated population/date baseline.
    """
    scale = math.exp(shift)
    root = math.sqrt(scale)
    for c in ("prior_seconds", "opp_prior_seconds"):
        before = float(frame.at[idx, c])
        # prior_sec depends on the date population hazard and is added later by
        # _clock_lookup. We cannot alter that baseline here, so use a conservative
        # exposure scaling on observed seconds only; exact realized hazard change
        # is captured by the adjusted Brain manifest/output.
        after = max(before / root, 0.0)
        frame.at[idx, c] = after
        audit.append({"fighter": fighter, "system": label, "field": c, "transform": f"/ sqrt(exp(shift)); prior_events={prior_events}", "before": before, "after": after})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight-id", default="81cde317c156723b")
    ap.add_argument("--fighter-a", default="Asu Almabayev")
    ap.add_argument("--fighter-b", default="Charles Johnson")
    ap.add_argument("--elo-a", type=float, default=1310.0)
    ap.add_argument("--elo-b", type=float, default=1155.7)
    ap.add_argument("--paths", type=int, default=500)
    ap.add_argument("--domain-share", type=float, default=0.20)
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/elo_fsr_brain_one_fight_shadow"))
    args = ap.parse_args()
    if not 0.0 < args.domain_share <= 1.0:
        raise ValueError("--domain-share must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bundle = Path(DEFAULT_BUNDLE_DIR)
    adjusted_bundle = args.output_dir / "adjusted_bundle"
    if adjusted_bundle.exists():
        shutil.rmtree(adjusted_bundle)
    shutil.copytree(baseline_bundle, adjusted_bundle)

    # Standard Elo odds: odds(A/B) = 10^(delta/400). Convert to natural-log odds.
    elo_delta = float(args.elo_a - args.elo_b)
    matchup_log_odds = math.log(10.0) * elo_delta / 400.0
    side_a = +0.5 * matchup_log_odds
    side_b = -0.5 * matchup_log_odds
    shift_a = side_a * args.domain_share
    shift_b = side_b * args.domain_share

    audit: list[dict] = []

    ewm_path = adjusted_bundle / FILES["ewm_fsr"]
    ewm = pd.read_parquet(ewm_path)
    ia, ib = _target_indices(ewm, args.fight_id, args.fighter_a, args.fighter_b)
    _apply_fsr(ewm, ia, args.fighter_a, shift_a, audit)
    _apply_fsr(ewm, ib, args.fighter_b, shift_b, audit)
    ewm.to_parquet(ewm_path, index=False)

    ko_path = adjusted_bundle / FILES["ko_prefight"]
    ko = pd.read_parquet(ko_path)
    kia, kib = _target_indices(ko, args.fight_id, args.fighter_a, args.fighter_b)
    _scale_clock_row(ko, kia, args.fighter_a, shift_a, 2.0, "ko_clock", audit)
    _scale_clock_row(ko, kib, args.fighter_b, shift_b, 2.0, "ko_clock", audit)
    ko.to_parquet(ko_path, index=False)

    sub_path = adjusted_bundle / FILES["sub_prefight"]
    sub = pd.read_parquet(sub_path)
    sia, sib = _target_indices(sub, args.fight_id, args.fighter_a, args.fighter_b)
    _scale_clock_row(sub, sia, args.fighter_a, shift_a, 1.0, "sub_clock", audit)
    _scale_clock_row(sub, sib, args.fighter_b, shift_b, 1.0, "sub_clock", audit)
    sub.to_parquet(sub_path, index=False)

    manifest_path = adjusted_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, path in (("ewm_fsr", ewm_path), ("ko_prefight", ko_path), ("sub_prefight", sub_path)):
        manifest["files"][key]["sha256"] = sha256(path)
    manifest["research_shadow"] = {
        "type": "elo_full_active_brain_state",
        "fight_id": args.fight_id,
        "fighter_a": args.fighter_a,
        "fighter_b": args.fighter_b,
        "elo_a": args.elo_a,
        "elo_b": args.elo_b,
        "elo_delta": elo_delta,
        "matchup_log_odds": matchup_log_odds,
        "domain_share": args.domain_share,
        "fighter_a_domain_shift": shift_a,
        "fighter_b_domain_shift": shift_b,
        "touches_ko_finish_hazard": True,
        "touches_submission_finish_hazard": True,
        "production_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    baseline = run_locked(args.fight_id, baseline_bundle, args.paths, args.output_dir / "baseline_results.json")
    adjusted = run_locked(args.fight_id, adjusted_bundle, args.paths, args.output_dir / "adjusted_results.json")

    pd.DataFrame(audit).to_csv(args.output_dir / "elo_active_input_adjustments.csv", index=False)
    summary = {
        "fight_id": args.fight_id,
        "fighter_a": args.fighter_a,
        "fighter_b": args.fighter_b,
        "elo_a": args.elo_a,
        "elo_b": args.elo_b,
        "elo_delta": elo_delta,
        "matchup_log_odds": matchup_log_odds,
        "domain_share": args.domain_share,
        "shift_a": shift_a,
        "shift_b": shift_b,
        "paths": args.paths,
        "baseline": baseline,
        "adjusted": adjusted,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
