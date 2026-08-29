#!/usr/bin/env python3
"""Standalone Step 6: opponent-strength / schedule-strength Elo research.

Research-only. No Brain, FSR, market, QP, result-quality, or inactivity dependency.

For each fighter, track the prefight Elo ratings of their most recent UFC opponents.
Before the next bout, form a leakage-safe recent schedule-strength estimate from the
last N opponent ratings. Adjust the fighter's effective Elo only for prediction:

    effective_rating = raw_elo + alpha * (recent_opp_mean - base_rating)

The underlying Elo update remains plain chronological Elo. Alpha and lookback are
selected strictly on fights before the cutoff, then frozen for holdout evaluation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts, _expected


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


def replay(bouts: pd.DataFrame, *, k: float, base: float, alpha: float, lookback: int) -> pd.DataFrame:
    ratings: dict[str, float] = defaultdict(lambda: base)
    opp_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=lookback))
    rows = []

    for bout in bouts.itertuples(index=False):
        r, b = bout.red_fighter, bout.blue_fighter
        rr, br = ratings[r], ratings[b]

        r_hist = list(opp_hist[r]); b_hist = list(opp_hist[b])
        r_opp_mean = float(np.mean(r_hist)) if r_hist else np.nan
        b_opp_mean = float(np.mean(b_hist)) if b_hist else np.nan
        r_sched_edge = 0.0 if not r_hist else (r_opp_mean - base)
        b_sched_edge = 0.0 if not b_hist else (b_opp_mean - base)

        r_eff = rr + alpha * r_sched_edge
        b_eff = br + alpha * b_sched_edge
        sched_p = _expected(r_eff, b_eff)
        base_p = _expected(rr, br)

        rows.append({
            "date": pd.Timestamp(bout.date),
            "bout_id": bout.bout_id,
            "red_fighter": r,
            "blue_fighter": b,
            "winner": bout.winner,
            "red_raw_elo": rr,
            "blue_raw_elo": br,
            "red_recent_opp_mean": r_opp_mean,
            "blue_recent_opp_mean": b_opp_mean,
            "red_schedule_fights": len(r_hist),
            "blue_schedule_fights": len(b_hist),
            "baseline_red_win_prob": base_p,
            "baseline_blue_win_prob": 1-base_p,
            "schedule_red_win_prob": sched_p,
            "schedule_blue_win_prob": 1-sched_p,
            "red_effective_rating": r_eff,
            "blue_effective_rating": b_eff,
        })

        winner = bout.winner
        rscore = 1.0 if winner == r else (0.0 if winner == b else None)
        if rscore is not None:
            delta = k * (rscore - base_p)
            ratings[r] = rr + delta
            ratings[b] = br - delta

        # Store opponent strength as it was known before this bout; current-bout
        # outcome never enters the schedule feature for this same prediction.
        opp_hist[r].append(br)
        opp_hist[b].append(rr)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_schedule"))
    ap.add_argument("--train-before", default="2025-01-01")
    ap.add_argument("--k-factor", type=float, default=170.0)
    ap.add_argument("--base-rating", type=float, default=1000.0)
    args = ap.parse_args()

    bouts = build_bouts(pd.read_parquet(args.input))
    cutoff = pd.Timestamp(args.train_before)

    alphas = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    lookbacks = [3, 5, 8]
    candidates = []
    outputs = {}
    for lookback in lookbacks:
        for alpha in alphas:
            df = replay(bouts, k=args.k_factor, base=args.base_rating, alpha=alpha, lookback=lookback)
            train = df[df["date"] < cutoff]
            m = _metrics(train, "schedule_red_win_prob")
            candidates.append({
                "lookback": lookback,
                "alpha": alpha,
                "train_log_loss": m["log_loss"],
                "train_brier": m["brier"],
                "train_accuracy_non_ties": m["accuracy_non_ties"],
            })
            outputs[(lookback, alpha)] = df

    cand = pd.DataFrame(candidates).sort_values(["train_log_loss", "train_brier"], kind="stable").reset_index(drop=True)
    best = cand.iloc[0]
    selected_lookback = int(best["lookback"])
    selected_alpha = float(best["alpha"])
    selected = outputs[(selected_lookback, selected_alpha)].copy()

    train = selected[selected["date"] < cutoff]
    hold = selected[selected["date"] >= cutoff]
    summary = {
        "source": str(args.input),
        "train_before": str(cutoff.date()),
        "mechanism": "prediction-only Elo adjustment from mean prefight Elo of recent UFC opponents; Elo updates remain plain",
        "selected": {"lookback": selected_lookback, "alpha": selected_alpha},
        "candidate_count": int(len(cand)),
        "selection_metric": "pre-2025 train log loss, then Brier",
        "train_baseline": _metrics(train, "baseline_red_win_prob"),
        "train_schedule_adjusted": _metrics(train, "schedule_red_win_prob"),
        "holdout_baseline": _metrics(hold, "baseline_red_win_prob"),
        "holdout_schedule_adjusted": _metrics(hold, "schedule_red_win_prob"),
        "scope": "research-only; plain Elo benchmark; no Brain/FSR/market/QP/result-quality/inactivity dependency",
        "leakage_rule": "recent opponent ratings are captured before each prior bout update; alpha/lookback selected only before cutoff",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cand.to_csv(args.output_dir / "candidate_grid.csv", index=False)
    selected.to_csv(args.output_dir / "fight_schedule_strength_elo.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
