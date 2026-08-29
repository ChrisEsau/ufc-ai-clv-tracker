#!/usr/bin/env python3
"""Standalone Step 4: result-quality-weighted Elo research.

Research-only. No Brain, FSR, market, or production dependencies.

The experiment asks whether Elo improves when the rating update magnitude depends
on result type. Candidate multipliers are selected using fights strictly before a
training cutoff, then frozen and evaluated on untouched fights on/after the cutoff.

Supported result classes from the master data:
- KO/TKO
- Submission
- Standard decision
- Close decision (split/majority)

Because the current master bout layer does not expose trustworthy scorecard margin
for every historical fight, we do not invent a separate "dominant decision" class.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts, _expected


@dataclass
class State:
    rating: float = 1000.0


def _classify(method: object) -> str:
    s = str(method or "").lower()
    if "ko" in s or "tko" in s:
        return "ko"
    if "sub" in s:
        return "sub"
    if "split" in s or "majority" in s:
        return "close_dec"
    if "decision" in s or "dec" in s:
        return "decision"
    return "other"


def _metrics(df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    d = df[df["winner"].notna()].copy()
    p = np.clip(d[prob_col].astype(float).to_numpy(), 1e-9, 1 - 1e-9)
    y = (d["winner"] == d["red_fighter"]).astype(float).to_numpy()
    non_tie = np.abs(p - 0.5) > 1e-12
    return {
        "n": int(len(d)),
        "accuracy_non_ties": float(np.mean((p[non_tie] > 0.5) == (y[non_tie] > 0.5))) if non_tie.any() else None,
        "non_ties": int(non_tie.sum()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
    }


def replay(bouts: pd.DataFrame, *, k: float, base: float, multipliers: dict[str, float]) -> pd.DataFrame:
    states: dict[str, State] = {}
    rows = []
    for bout in bouts.itertuples(index=False):
        r, b = bout.red_fighter, bout.blue_fighter
        sr = states.setdefault(r, State(base)); sb = states.setdefault(b, State(base))
        rr, br = sr.rating, sb.rating
        rp = _expected(rr, br)
        winner = bout.winner
        rscore = 1.0 if winner == r else (0.0 if winner == b else None)
        cls = _classify(getattr(bout, "method", ""))
        rows.append({
            "date": pd.Timestamp(bout.date), "bout_id": bout.bout_id,
            "red_fighter": r, "blue_fighter": b, "winner": winner,
            "method": getattr(bout, "method", ""), "result_class": cls,
            "red_pre_rating": rr, "blue_pre_rating": br,
            "red_win_prob": rp, "blue_win_prob": 1-rp,
        })
        if rscore is None:
            continue
        mult = multipliers.get(cls, multipliers.get("other", 1.0))
        delta = k * mult * (rscore - rp)
        sr.rating = rr + delta
        sb.rating = br - delta
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_result_quality"))
    ap.add_argument("--train-before", default="2025-01-01")
    ap.add_argument("--k-factor", type=float, default=170.0)
    ap.add_argument("--base-rating", type=float, default=1000.0)
    args = ap.parse_args()

    src = pd.read_parquet(args.input)
    bouts = build_bouts(src)
    cutoff = pd.Timestamp(args.train_before)

    # Small interpretable grid. Standard decision remains anchor at 1.0.
    # We test whether decisive finishes should update more and close decisions less.
    ko_vals = [0.9, 1.0, 1.1, 1.2]
    sub_vals = [0.9, 1.0, 1.1, 1.2]
    close_vals = [0.5, 0.65, 0.8, 1.0]

    candidates = []
    for ko in ko_vals:
        for sub in sub_vals:
            for close in close_vals:
                mult = {"ko": ko, "sub": sub, "decision": 1.0, "close_dec": close, "other": 1.0}
                full = replay(bouts, k=args.k_factor, base=args.base_rating, multipliers=mult)
                train = full[full["date"] < cutoff]
                m = _metrics(train, "red_win_prob")
                candidates.append({**mult, "train_log_loss": m["log_loss"], "train_brier": m["brier"], "train_accuracy_non_ties": m["accuracy_non_ties"]})

    cand = pd.DataFrame(candidates).sort_values(["train_log_loss", "train_brier"], kind="stable").reset_index(drop=True)
    best = cand.iloc[0]
    selected = {k: float(best[k]) for k in ["ko", "sub", "decision", "close_dec", "other"]}

    baseline_mult = {"ko":1.0,"sub":1.0,"decision":1.0,"close_dec":1.0,"other":1.0}
    base_df = replay(bouts, k=args.k_factor, base=args.base_rating, multipliers=baseline_mult)
    rq_df = replay(bouts, k=args.k_factor, base=args.base_rating, multipliers=selected)

    merged = rq_df.rename(columns={"red_win_prob":"rq_red_win_prob","blue_win_prob":"rq_blue_win_prob","red_pre_rating":"rq_red_pre_rating","blue_pre_rating":"rq_blue_pre_rating"})
    merged["baseline_red_win_prob"] = base_df["red_win_prob"].to_numpy()
    merged["baseline_blue_win_prob"] = base_df["blue_win_prob"].to_numpy()
    merged["baseline_red_pre_rating"] = base_df["red_pre_rating"].to_numpy()
    merged["baseline_blue_pre_rating"] = base_df["blue_pre_rating"].to_numpy()

    train = merged[merged["date"] < cutoff]
    hold = merged[merged["date"] >= cutoff]
    summary = {
        "source": str(args.input),
        "train_before": str(cutoff.date()),
        "selection_metric": "pre-2025 train log loss, then Brier",
        "selected_multipliers": selected,
        "candidate_count": int(len(cand)),
        "train_baseline": _metrics(train, "baseline_red_win_prob"),
        "train_result_quality": _metrics(train, "rq_red_win_prob"),
        "holdout_baseline": _metrics(hold, "baseline_red_win_prob"),
        "holdout_result_quality": _metrics(hold, "rq_red_win_prob"),
        "scope": "research-only; QP excluded; no Brain/FSR/market dependency",
        "dominant_decision_note": "not separately modeled because universal trustworthy scorecard margin is not available in this bout layer",
        "leakage_rule": "multipliers selected only on fights before cutoff, then frozen for holdout",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cand.to_csv(args.output_dir / "candidate_grid.csv", index=False)
    merged.to_csv(args.output_dir / "fight_result_quality_elo.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
