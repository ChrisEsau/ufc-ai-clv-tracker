#!/usr/bin/env python3
"""Evidence-gated direct six-way Glicko ablation.

Research-only. No Brain, FSR, or market inputs.

This starts from the pure direct joint Glicko-6 architecture, but changes only
how method-specific offense/defense ratings learn after each bout:

* Predictions remain one six-class softmax over R/B x KO/SUB/DEC.
* The online method-frequency intercept is unchanged.
* Only the ACTUAL method track is updated after a bout.
* Within that method, the winning side receives positive evidence and the
  losing side receives negative evidence.
* Non-observed methods receive no update from that bout.

Thus a decision is not automatically negative evidence for KO or submission,
and a KO is not automatically negative evidence for submission or decision.
All predictions are captured pre-bout; updates happen only after the bout.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_fightmatrix_glicko import State, update
from pipeline.research.standalone_glicko_six_way import (
    METHODS,
    SIX_COLS,
    class_signal,
    method_family,
    method_metrics,
    method_priors,
    share_metrics,
    side_metrics,
    six_metrics,
    softmax,
)


def run(bouts: pd.DataFrame) -> pd.DataFrame:
    offense = {m: defaultdict(State) for m in METHODS}
    defense = {m: defaultdict(State) for m in METHODS}
    method_counts = {m: 1.0 for m in METHODS}
    rows: list[dict] = []

    for b in bouts.itertuples(index=False):
        r, bl = b.red_fighter, b.blue_fighter
        priors = method_priors(method_counts)
        scores: list[float] = []
        q_values: dict[str, float] = {}

        # Same pure-joint prefight score construction as the baseline.
        for side, candidate_winner, candidate_loser in (("R", r, bl), ("B", bl, r)):
            for meth in METHODS:
                so = offense[meth][candidate_winner]
                sd = defense[meth][candidate_loser]
                score, q = class_signal(so, sd, b.date, priors[meth])
                scores.append(score)
                q_values[f"q_{side.lower()}_{meth.lower()}"] = q

        p = softmax(scores)
        probs = dict(zip(SIX_COLS, p.tolist()))

        actual_m = method_family(getattr(b, "method", ""))
        actual_six = None
        if b.winner is not None and actual_m is not None:
            actual_six = ("R_" if b.winner == r else "B_") + actual_m

        p_red_win = probs["p_red_ko"] + probs["p_red_sub"] + probs["p_red_dec"]
        row = {
            "date": b.date,
            "bout_id": b.bout_id,
            "red_fighter": r,
            "blue_fighter": bl,
            "winner": b.winner,
            "method": getattr(b, "method", ""),
            "p_red_win": p_red_win,
            "p_blue_win": 1.0 - p_red_win,
            **probs,
            "p_method_ko": probs["p_red_ko"] + probs["p_blue_ko"],
            "p_method_sub": probs["p_red_sub"] + probs["p_blue_sub"],
            "p_method_dec": probs["p_red_dec"] + probs["p_blue_dec"],
            "actual_method": actual_m,
            "actual_six": actual_six,
            "prior_ko": priors["KO"],
            "prior_sub": priors["SUB"],
            "prior_dec": priors["DEC"],
            **q_values,
        }
        rows.append(row)

        if actual_six is None:
            continue

        # Evidence gating: update ONLY the observed method. We still update both
        # orientations inside that method so the observed winner side is the
        # positive class and the loser side is the negative class.
        pending = []
        for side, candidate_winner, candidate_loser in (("R", r, bl), ("B", bl, r)):
            label = f"{side}_{actual_m}"
            y = 1.0 if label == actual_six else 0.0
            so = offense[actual_m][candidate_winner]
            sd = defense[actual_m][candidate_loser]
            no, nro = update(so.rating, so.rd, sd.rating, sd.rd, y)
            nd, nrd = update(sd.rating, sd.rd, so.rating, so.rd, 1.0 - y)
            pending.append((so, no, nro, sd, nd, nrd))

        for so, no, nro, sd, nd, nrd in pending:
            so.rating, so.rd, so.last_date = no, nro, b.date
            sd.rating, sd.rd, sd.last_date = nd, nrd, b.date

        method_counts[actual_m] += 1.0

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/standalone_glicko_six_way_gated"),
    )
    args = ap.parse_args()

    bouts = build_bouts(pd.read_parquet(args.input))
    pred = run(bouts)
    cutoff = pd.Timestamp(args.holdout_from)
    train = pred[pred.date < cutoff].copy()
    hold = pred[pred.date >= cutoff].copy()

    summary = {
        "architecture": "pure direct joint Glicko-6 with actual-method-only evidence-gated updates",
        "update_rule": "only observed method updated; winner side positive, loser side negative",
        "hierarchical_winner_x_method": False,
        "no_brain": True,
        "no_fsr": True,
        "no_market": True,
        "holdout_from": args.holdout_from,
        "train_side": side_metrics(train),
        "holdout_side": side_metrics(hold),
        "train_method": method_metrics(train),
        "holdout_method": method_metrics(hold),
        "train_six": six_metrics(train),
        "holdout_six": six_metrics(hold),
        "train_method_shares": share_metrics(train),
        "holdout_method_shares": share_metrics(hold),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir / "fight_predictions.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
