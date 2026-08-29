#!/usr/bin/env python3
"""Standalone Step 5: inactivity-adjusted Elo research.

Research-only. No Brain, FSR, market, QP, or production dependencies.

Question: should an established fighter's stored Elo deviation from the base rating
shrink toward neutral after a long layoff?

For each fighter, before a bout we compute days since their previous UFC bout.
After an optional grace period, the fighter's rating deviation from base is
multiplied by an exponential half-life decay. The decayed rating is then used for
the prefight probability and becomes the state from which the current result
updates Elo.

Candidate grace/half-life settings are selected using fights strictly before the
training cutoff and then frozen for untouched holdout evaluation.
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
    last_date: pd.Timestamp | None = None


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


def _decay_rating(rating: float, base: float, days: float | None, grace_days: float, half_life_days: float | None) -> float:
    if days is None or half_life_days is None:
        return rating
    excess = max(0.0, float(days) - float(grace_days))
    if excess <= 0:
        return rating
    factor = 0.5 ** (excess / float(half_life_days))
    return base + (rating - base) * factor


def replay(bouts: pd.DataFrame, *, k: float, base: float, grace_days: float, half_life_days: float | None) -> pd.DataFrame:
    states: dict[str, State] = {}
    rows = []
    for bout in bouts.itertuples(index=False):
        date = pd.Timestamp(bout.date)
        r, b = bout.red_fighter, bout.blue_fighter
        sr = states.setdefault(r, State(base)); sb = states.setdefault(b, State(base))

        r_days = None if sr.last_date is None else max(0.0, (date - sr.last_date).days)
        b_days = None if sb.last_date is None else max(0.0, (date - sb.last_date).days)
        rr = _decay_rating(sr.rating, base, r_days, grace_days, half_life_days)
        br = _decay_rating(sb.rating, base, b_days, grace_days, half_life_days)
        rp = _expected(rr, br)

        winner = bout.winner
        rscore = 1.0 if winner == r else (0.0 if winner == b else None)
        rows.append({
            "date": date, "bout_id": bout.bout_id,
            "red_fighter": r, "blue_fighter": b, "winner": winner,
            "red_days_since_last": r_days, "blue_days_since_last": b_days,
            "red_pre_rating": rr, "blue_pre_rating": br,
            "red_win_prob": rp, "blue_win_prob": 1-rp,
        })

        # The inactivity-adjusted prefight state becomes the current rating state.
        sr.rating = rr; sb.rating = br
        if rscore is not None:
            delta = k * (rscore - rp)
            sr.rating = rr + delta
            sb.rating = br - delta
        sr.last_date = date; sb.last_date = date
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_inactivity"))
    ap.add_argument("--train-before", default="2025-01-01")
    ap.add_argument("--k-factor", type=float, default=170.0)
    ap.add_argument("--base-rating", type=float, default=1000.0)
    args = ap.parse_args()

    src = pd.read_parquet(args.input)
    bouts = build_bouts(src)
    cutoff = pd.Timestamp(args.train_before)

    baseline = replay(bouts, k=args.k_factor, base=args.base_rating, grace_days=0.0, half_life_days=None)

    grace_vals = [0.0, 180.0, 365.0, 540.0]
    half_life_vals = [365.0, 730.0, 1095.0, 1460.0, 2190.0, 3650.0]
    candidates = []
    for grace in grace_vals:
        for hl in half_life_vals:
            d = replay(bouts, k=args.k_factor, base=args.base_rating, grace_days=grace, half_life_days=hl)
            train = d[d["date"] < cutoff]
            m = _metrics(train, "red_win_prob")
            candidates.append({
                "grace_days": grace,
                "half_life_days": hl,
                "train_log_loss": m["log_loss"],
                "train_brier": m["brier"],
                "train_accuracy_non_ties": m["accuracy_non_ties"],
            })

    cand = pd.DataFrame(candidates).sort_values(["train_log_loss", "train_brier"], kind="stable").reset_index(drop=True)
    best = cand.iloc[0]
    selected = {"grace_days": float(best["grace_days"]), "half_life_days": float(best["half_life_days"])}
    adj = replay(bouts, k=args.k_factor, base=args.base_rating, **selected)

    merged = adj.rename(columns={
        "red_win_prob":"inactivity_red_win_prob", "blue_win_prob":"inactivity_blue_win_prob",
        "red_pre_rating":"inactivity_red_pre_rating", "blue_pre_rating":"inactivity_blue_pre_rating",
    })
    merged["baseline_red_win_prob"] = baseline["red_win_prob"].to_numpy()
    merged["baseline_blue_win_prob"] = baseline["blue_win_prob"].to_numpy()
    merged["baseline_red_pre_rating"] = baseline["red_pre_rating"].to_numpy()
    merged["baseline_blue_pre_rating"] = baseline["blue_pre_rating"].to_numpy()

    train = merged[merged["date"] < cutoff]
    hold = merged[merged["date"] >= cutoff]
    summary = {
        "source": str(args.input),
        "train_before": str(cutoff.date()),
        "mechanism": "after grace period, Elo deviation from base decays exponentially toward neutral before next UFC bout",
        "selection_metric": "pre-2025 train log loss, then Brier",
        "selected": selected,
        "candidate_count": int(len(cand)),
        "train_baseline": _metrics(train, "baseline_red_win_prob"),
        "train_inactivity_adjusted": _metrics(train, "inactivity_red_win_prob"),
        "holdout_baseline": _metrics(hold, "baseline_red_win_prob"),
        "holdout_inactivity_adjusted": _metrics(hold, "inactivity_red_win_prob"),
        "scope": "research-only; result-quality and QP excluded; no Brain/FSR/market dependency",
        "leakage_rule": "grace/half-life selected only on fights before cutoff, then frozen for holdout",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cand.to_csv(args.output_dir / "candidate_grid.csv", index=False)
    merged.to_csv(args.output_dir / "fight_inactivity_elo.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
