#!/usr/bin/env python3
"""Standalone FightMatrix-inspired Quality Performance diagnostic.

Research-only. Reads UFC master parquet and processes bouts chronologically.
No Brain, FSR, market, or production dependencies.

This does NOT reproduce proprietary CIRRS. It tests the disclosed concept that
recent results should validate a fighter's competitive level against sufficiently
strong opposition. Approximate FightMatrix public thresholds are represented as:
- win: opponent pre-rating >= 1/3 of fighter pre-rating
- draw: opponent pre-rating >= 1/2 of fighter pre-rating (when resolvable)
- close split/majority loss: opponent pre-rating >= 2/3 of fighter pre-rating

Because our Elo scale has an arbitrary additive origin, ratio thresholds are
applied to positive "strength above floor" values rather than raw Elo points.
The primary predictive diagnostics are therefore also exported continuously:
opponent-adjusted residual, recent quality rate, quality recency, and recent
opponent strength. All features are captured strictly before the current bout.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts, _expected


@dataclass
class State:
    rating: float = 1000.0
    fights: int = 0


def _quality_strength(rating: float, floor: float = 600.0) -> float:
    return max(1.0, rating - floor)


def _is_close_decision(method: Any) -> bool:
    s = str(method or "").lower()
    return "split" in s or "majority" in s


def _recent_mean(q: deque[float], n: int) -> float | None:
    vals = list(q)[-n:]
    return float(np.mean(vals)) if vals else None


def _recent_days_since(dates: deque[pd.Timestamp], now: pd.Timestamp) -> float | None:
    if not dates:
        return None
    return float((now - dates[-1]).days)


def run(bouts: pd.DataFrame, k: float = 170.0, base: float = 1000.0) -> pd.DataFrame:
    states = defaultdict(lambda: State(base, 0))
    quality = defaultdict(lambda: deque(maxlen=20))
    quality_dates = defaultdict(lambda: deque(maxlen=20))
    opp_ratings = defaultdict(lambda: deque(maxlen=20))
    residuals = defaultdict(lambda: deque(maxlen=20))
    rows = []

    for bout in bouts.itertuples(index=False):
        r, b = bout.red_fighter, bout.blue_fighter
        sr, sb = states[r], states[b]
        rr, br = float(sr.rating), float(sb.rating)
        rp = _expected(rr, br); bp = 1-rp
        winner = bout.winner
        rscore = 1.0 if winner == r else (0.0 if winner == b else None)
        method = getattr(bout, "method", "")
        date = pd.Timestamp(bout.date)

        r_q3 = _recent_mean(quality[r], 3); b_q3 = _recent_mean(quality[b], 3)
        r_q5 = _recent_mean(quality[r], 5); b_q5 = _recent_mean(quality[b], 5)
        r_days = _recent_days_since(quality_dates[r], date); b_days = _recent_days_since(quality_dates[b], date)
        r_opp5 = _recent_mean(opp_ratings[r], 5); b_opp5 = _recent_mean(opp_ratings[b], 5)
        r_res5 = _recent_mean(residuals[r], 5); b_res5 = _recent_mean(residuals[b], 5)

        # Validation metric: quality rate first, recency breaks ties. Requires >=3 prior UFC fights each.
        qp_pick = None
        if sr.fights >= 3 and sb.fights >= 3 and r_q5 is not None and b_q5 is not None:
            if abs(r_q5-b_q5) > 1e-12:
                qp_pick = r if r_q5 > b_q5 else b
            elif r_days is not None and b_days is not None and r_days != b_days:
                qp_pick = r if r_days < b_days else b

        row = {
            "date": date, "bout_id": bout.bout_id, "red_fighter": r, "blue_fighter": b, "winner": winner,
            "red_pre_rating": rr, "blue_pre_rating": br, "red_elo_win_prob": rp, "blue_elo_win_prob": bp,
            "red_prior_fights": sr.fights, "blue_prior_fights": sb.fights,
            "red_qp_rate_last3": r_q3, "blue_qp_rate_last3": b_q3,
            "red_qp_rate_last5": r_q5, "blue_qp_rate_last5": b_q5,
            "red_days_since_qp": r_days, "blue_days_since_qp": b_days,
            "red_last5_opp_pre_rating": r_opp5, "blue_last5_opp_pre_rating": b_opp5,
            "red_last5_residual": r_res5, "blue_last5_residual": b_res5,
            "qp_pick": qp_pick,
            "qp_pick_correct": bool(qp_pick == winner) if qp_pick and winner else np.nan,
        }

        if rscore is None:
            rows.append(row); continue

        bscore = 1-rscore
        rs = _quality_strength(rr); bs = _quality_strength(br)
        close = _is_close_decision(method)
        r_qp = (rscore == 1 and bs >= rs/3) or (rscore == 0 and close and bs >= 2*rs/3)
        b_qp = (bscore == 1 and rs >= bs/3) or (bscore == 0 and close and rs >= 2*bs/3)

        quality[r].append(float(r_qp)); quality[b].append(float(b_qp))
        if r_qp: quality_dates[r].append(date)
        if b_qp: quality_dates[b].append(date)
        opp_ratings[r].append(br); opp_ratings[b].append(rr)
        residuals[r].append(rscore-rp); residuals[b].append(bscore-bp)

        delta = k*(rscore-rp)
        sr.rating = rr+delta; sb.rating = br-delta
        sr.fights += 1; sb.fights += 1
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength_quality"))
    ap.add_argument("--k-factor", type=float, default=170.0)
    args=ap.parse_args()
    src=pd.read_parquet(args.input)
    bouts=build_bouts(src)
    out=run(bouts,k=args.k_factor)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output_dir/"fight_quality_performance.csv",index=False)
    eligible=out[out.qp_pick_correct.notna()]
    elo=out[(out.winner.notna()) & ((out.red_elo_win_prob-0.5).abs()>1e-12)].copy()
    elo["elo_correct"]=np.where(elo.red_elo_win_prob>0.5, elo.red_fighter==elo.winner, elo.blue_fighter==elo.winner)
    summary={
        "source":str(args.input),"bouts":int(len(out)),"qp_eligible":int(len(eligible)),
        "qp_pick_accuracy":float(eligible.qp_pick_correct.mean()) if len(eligible) else None,
        "elo_accuracy_non_ties":float(elo.elo_correct.mean()) if len(elo) else None,
        "qp_definition":"recent binary rating-validation rate; FightMatrix-inspired, not CIRRS",
        "leakage_rule":"all quality features captured before current bout update"
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
