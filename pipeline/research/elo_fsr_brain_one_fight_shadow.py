#!/usr/bin/env python3
"""One-fight research shadow: apply a small symmetric Elo-derived offset to FSR inputs.

This does NOT modify production mechanics or the canonical locked Brain bundle.
It copies the locked bundle to a temporary research bundle, adjusts only the two
target fight rows in the PURE EWM 0.50 FSR parquet, updates the copied manifest,
and runs the approved locked Brain entry point on baseline and adjusted inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight-id", default="81cde317c156723b")
    ap.add_argument("--fighter-a", default="Asu Almabayev")
    ap.add_argument("--fighter-b", default="Charles Johnson")
    ap.add_argument("--elo-a", type=float, default=1310.0)
    ap.add_argument("--elo-b", type=float, default=1155.7)
    ap.add_argument("--paths", type=int, default=500)
    ap.add_argument("--points-per-100-elo", type=float, default=1.0)
    ap.add_argument("--max-fsr-shift", type=float, default=2.0)
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/elo_fsr_brain_one_fight_shadow"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bundle = Path(DEFAULT_BUNDLE_DIR)
    adjusted_bundle = args.output_dir / "adjusted_bundle"
    if adjusted_bundle.exists():
        shutil.rmtree(adjusted_bundle)
    shutil.copytree(baseline_bundle, adjusted_bundle)

    # Symmetric matchup adjustment: each side moves half the Elo-implied FSR spread.
    elo_delta = args.elo_a - args.elo_b
    total_spread = (elo_delta / 100.0) * args.points_per_100_elo
    shift_a = max(-args.max_fsr_shift, min(args.max_fsr_shift, total_spread / 2.0))
    shift_b = -shift_a

    ewm_path = adjusted_bundle / FILES["ewm_fsr"]
    ewm = pd.read_parquet(ewm_path)
    target = ewm[ewm["fight_id"].astype(str).eq(str(args.fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected 2 target FSR rows, found {len(target)}")

    # FSR trait columns are numeric 0-100 rating-like fields. Exclude identifiers,
    # raw physical/context/exposure fields, dates, and counts. This is intentionally
    # broad for the first sensitivity pilot; exact columns are exported for audit.
    excluded_tokens = (
        "id", "date", "age", "height", "reach", "weight", "seconds", "count",
        "fight", "bout", "prior", "experience", "sample", "year", "round",
    )
    trait_cols = []
    for c in ewm.columns:
        if not pd.api.types.is_numeric_dtype(ewm[c]):
            continue
        lc = c.lower()
        if any(tok in lc for tok in excluded_tokens):
            continue
        vals = pd.to_numeric(target[c], errors="coerce")
        if vals.notna().all() and ((vals >= 0) & (vals <= 100)).all():
            trait_cols.append(c)
    if not trait_cols:
        raise RuntimeError("no eligible 0-100 FSR trait columns found")

    # Match names if present, otherwise preserve row order deterministically.
    name_col = next((c for c in ("fighter_name", "name", "fighter") if c in ewm.columns), None)
    idxs = list(target.index)
    if name_col:
        by_name = {str(ewm.loc[i, name_col]): i for i in idxs}
        ia = by_name.get(args.fighter_a)
        ib = by_name.get(args.fighter_b)
        if ia is None or ib is None:
            raise RuntimeError(f"fighter names not found in target rows: {sorted(by_name)}")
    else:
        ia, ib = idxs[0], idxs[1]

    before_rows = []
    for fighter, idx, shift in ((args.fighter_a, ia, shift_a), (args.fighter_b, ib, shift_b)):
        for c in trait_cols:
            before = float(ewm.at[idx, c])
            after = max(0.0, min(100.0, before + shift))
            ewm.at[idx, c] = after
            before_rows.append({"fighter": fighter, "trait": c, "before": before, "shift": shift, "after": after})

    ewm.to_parquet(ewm_path, index=False)
    manifest_path = adjusted_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["ewm_fsr"]["sha256"] = sha256(ewm_path)
    manifest["research_shadow"] = {
        "type": "symmetric_elo_fsr_offset",
        "fight_id": args.fight_id,
        "fighter_a": args.fighter_a,
        "fighter_b": args.fighter_b,
        "elo_a": args.elo_a,
        "elo_b": args.elo_b,
        "points_per_100_elo": args.points_per_100_elo,
        "shift_a": shift_a,
        "shift_b": shift_b,
        "adjusted_traits": trait_cols,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    baseline = run_locked(args.fight_id, baseline_bundle, args.paths, args.output_dir / "baseline_results.json")
    adjusted = run_locked(args.fight_id, adjusted_bundle, args.paths, args.output_dir / "adjusted_results.json")

    pd.DataFrame(before_rows).to_csv(args.output_dir / "fsr_adjustments.csv", index=False)
    summary = {
        "fight_id": args.fight_id,
        "fighter_a": args.fighter_a,
        "fighter_b": args.fighter_b,
        "elo_a": args.elo_a,
        "elo_b": args.elo_b,
        "elo_delta": elo_delta,
        "shift_a": shift_a,
        "shift_b": shift_b,
        "paths": args.paths,
        "adjusted_trait_count": len(trait_cols),
        "adjusted_traits": trait_cols,
        "baseline": baseline,
        "adjusted": adjusted,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
