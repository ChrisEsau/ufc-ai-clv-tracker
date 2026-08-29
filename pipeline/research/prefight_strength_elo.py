#!/usr/bin/env python3
"""Standalone leakage-safe prefight fighter-strength research tool.

Reads the master UFC round parquet, collapses it to one row per bout, processes
bouts strictly chronologically, and emits prefight Elo plus simple recent-form
and recent-schedule diagnostics. It has no dependency on Brain MC, FSR, market
odds, or production simulation code.

Version 1 is intentionally simple and auditable. The goal is to establish
whether opponent-adjusted competitive strength contains OOS signal before any
more complex CIRRS-like mechanics are considered.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATE_CANDIDATES = ("date", "event_date", "fight_date", "bout_date")
BOUT_CANDIDATES = ("bout_id", "fight_id", "bout", "fight")
RED_NAME_CANDIDATES = ("r_name", "red_fighter", "red_name", "fighter_red", "r_fighter")
BLUE_NAME_CANDIDATES = ("b_name", "blue_fighter", "blue_name", "fighter_blue", "b_fighter")
WINNER_CANDIDATES = ("winner", "winner_name", "result", "bout_winner")
METHOD_CANDIDATES = ("method", "win_method", "finish_method")
ROUND_CANDIDATES = ("round", "finish_round")
TIME_CANDIDATES = ("time", "match_time_sec", "finish_time", "time_sec")


def _resolve(columns: Iterable[str], candidates: Iterable[str], *, required: bool = True) -> str | None:
    lookup = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(
            "Could not resolve required column. Tried: "
            + ", ".join(candidates)
            + ". Available columns: "
            + ", ".join(map(str, columns))
        )
    return None


def _clean_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).strip().split())


def _parse_winner(raw: Any, red: str, blue: str) -> str | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    s = str(raw).strip()
    sl = s.lower()
    if not s or sl in {"draw", "d", "nc", "no contest", "no_contest", "nan", "none"}:
        return None
    if sl in {"r", "red", "red_fighter"}:
        return red
    if sl in {"b", "blue", "blue_fighter"}:
        return blue
    if sl in {red.lower(), _clean_name(red).lower()}:
        return red
    if sl in {blue.lower(), _clean_name(blue).lower()}:
        return blue
    if red.lower() in sl and blue.lower() not in sl:
        return red
    if blue.lower() in sl and red.lower() not in sl:
        return blue
    # Common encoded outcomes.
    if sl in {"w", "win"}:
        # Ambiguous in a bout-level row; do not guess.
        return None
    return None


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _prob_from_diff(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


@dataclass
class FighterState:
    rating: float
    prior_fights: int = 0
    prior_wins: int = 0
    prior_losses: int = 0
    prior_draws: int = 0


def _record_string(results: deque[float], n: int) -> str:
    vals = list(results)[-n:]
    w = sum(v == 1.0 for v in vals)
    l = sum(v == 0.0 for v in vals)
    d = sum(v == 0.5 for v in vals)
    return f"{w}-{l}" + (f"-{d}" if d else "")


def _recent_rate(results: deque[float], n: int) -> float | None:
    vals = list(results)[-n:]
    if not vals:
        return None
    return float(np.mean(vals))


def _recent_mean(values: deque[float], n: int) -> float | None:
    vals = list(values)[-n:]
    if not vals:
        return None
    return float(np.mean(vals))


def build_bouts(rounds: pd.DataFrame) -> pd.DataFrame:
    date_col = _resolve(rounds.columns, DATE_CANDIDATES)
    bout_col = _resolve(rounds.columns, BOUT_CANDIDATES)
    red_col = _resolve(rounds.columns, RED_NAME_CANDIDATES)
    blue_col = _resolve(rounds.columns, BLUE_NAME_CANDIDATES)
    winner_col = _resolve(rounds.columns, WINNER_CANDIDATES)
    method_col = _resolve(rounds.columns, METHOD_CANDIDATES, required=False)
    round_col = _resolve(rounds.columns, ROUND_CANDIDATES, required=False)
    time_col = _resolve(rounds.columns, TIME_CANDIDATES, required=False)

    keep = [date_col, bout_col, red_col, blue_col, winner_col]
    for optional in (method_col, round_col, time_col):
        if optional and optional not in keep:
            keep.append(optional)

    df = rounds[keep].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"Date column {date_col!r} could not be parsed")

    # A round parquet normally has several rows per bout. Bout-level metadata
    # should be constant, so keep the first non-null bout row.
    df = df.sort_values([date_col, bout_col], kind="stable")
    bouts = df.groupby(bout_col, as_index=False, sort=False).first()

    out = pd.DataFrame(
        {
            "date": bouts[date_col],
            "bout_id": bouts[bout_col].astype(str),
            "red_fighter": bouts[red_col].map(_clean_name),
            "blue_fighter": bouts[blue_col].map(_clean_name),
            "winner_raw": bouts[winner_col],
        }
    )
    out["winner"] = [
        _parse_winner(raw, red, blue)
        for raw, red, blue in zip(out["winner_raw"], out["red_fighter"], out["blue_fighter"])
    ]
    if method_col:
        out["method"] = bouts[method_col].astype(str)
    if round_col:
        out["finish_round"] = bouts[round_col]
    if time_col:
        out["finish_time"] = bouts[time_col]

    out = out[(out["red_fighter"] != "") & (out["blue_fighter"] != "")].copy()
    out = out.sort_values(["date", "bout_id"], kind="stable").reset_index(drop=True)
    return out


def run_elo(bouts: pd.DataFrame, *, base_rating: float, k_factor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    states: dict[str, FighterState] = defaultdict(lambda: FighterState(base_rating))
    recent_results: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
    recent_opp_ratings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
    rows: list[dict[str, Any]] = []

    scored = 0
    skipped = 0

    for bout in bouts.itertuples(index=False):
        red = bout.red_fighter
        blue = bout.blue_fighter
        sr = states[red]
        sb = states[blue]

        red_pre = float(sr.rating)
        blue_pre = float(sb.rating)
        red_expected = _expected(red_pre, blue_pre)

        winner = bout.winner
        if winner == red:
            red_score = 1.0
        elif winner == blue:
            red_score = 0.0
        else:
            red_score = None

        row = {
            "date": bout.date,
            "bout_id": bout.bout_id,
            "red_fighter": red,
            "blue_fighter": blue,
            "winner": winner,
            "red_pre_rating": red_pre,
            "blue_pre_rating": blue_pre,
            "red_rating_edge": red_pre - blue_pre,
            "red_elo_win_prob": red_expected,
            "blue_elo_win_prob": 1.0 - red_expected,
            "red_prior_fights": sr.prior_fights,
            "blue_prior_fights": sb.prior_fights,
            "red_last3_record": _record_string(recent_results[red], 3),
            "blue_last3_record": _record_string(recent_results[blue], 3),
            "red_last5_record": _record_string(recent_results[red], 5),
            "blue_last5_record": _record_string(recent_results[blue], 5),
            "red_last3_result_rate": _recent_rate(recent_results[red], 3),
            "blue_last3_result_rate": _recent_rate(recent_results[blue], 3),
            "red_last5_result_rate": _recent_rate(recent_results[red], 5),
            "blue_last5_result_rate": _recent_rate(recent_results[blue], 5),
            "red_last5_opp_pre_rating": _recent_mean(recent_opp_ratings[red], 5),
            "blue_last5_opp_pre_rating": _recent_mean(recent_opp_ratings[blue], 5),
            "cold_start_either_le2": bool(sr.prior_fights <= 2 or sb.prior_fights <= 2),
        }

        if red_score is None:
            row["red_post_rating"] = red_pre
            row["blue_post_rating"] = blue_pre
            row["elo_pick"] = red if red_expected > 0.5 else blue
            row["elo_pick_correct"] = np.nan
            skipped += 1
            rows.append(row)
            continue

        delta = k_factor * (red_score - red_expected)
        sr.rating = red_pre + delta
        sb.rating = blue_pre - delta

        red_result = red_score
        blue_result = 1.0 - red_score
        recent_results[red].append(red_result)
        recent_results[blue].append(blue_result)
        recent_opp_ratings[red].append(blue_pre)
        recent_opp_ratings[blue].append(red_pre)

        sr.prior_fights += 1
        sb.prior_fights += 1
        if red_score == 1.0:
            sr.prior_wins += 1
            sb.prior_losses += 1
        elif red_score == 0.0:
            sr.prior_losses += 1
            sb.prior_wins += 1
        else:
            sr.prior_draws += 1
            sb.prior_draws += 1

        pick = red if red_expected > 0.5 else blue
        row["red_post_rating"] = float(sr.rating)
        row["blue_post_rating"] = float(sb.rating)
        row["elo_pick"] = pick
        row["elo_pick_correct"] = bool(pick == winner)
        scored += 1
        rows.append(row)

    fights = pd.DataFrame(rows)
    fighter_rows = []
    for fighter, state in states.items():
        fighter_rows.append(
            {
                "fighter": fighter,
                "rating": float(state.rating),
                "elo_implied_vs_1000": _prob_from_diff(float(state.rating) - base_rating),
                "fights": state.prior_fights,
                "wins": state.prior_wins,
                "losses": state.prior_losses,
                "draws": state.prior_draws,
                "last3_record": _record_string(recent_results[fighter], 3),
                "last5_record": _record_string(recent_results[fighter], 5),
                "last5_opp_pre_rating": _recent_mean(recent_opp_ratings[fighter], 5),
            }
        )
    fighters = pd.DataFrame(fighter_rows).sort_values("rating", ascending=False).reset_index(drop=True)
    fights.attrs["scored"] = scored
    fights.attrs["skipped"] = skipped
    return fights, fighters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/fight_details/ufc_round_stats.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/prefight_strength"))
    parser.add_argument("--base-rating", type=float, default=1000.0)
    parser.add_argument("--k-factor", type=float, default=170.0)
    args = parser.parse_args()

    rounds = pd.read_parquet(args.input)
    bouts = build_bouts(rounds)
    fights, fighters = run_elo(bouts, base_rating=args.base_rating, k_factor=args.k_factor)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fights_path = args.output_dir / "fight_prefight_strength.csv"
    fighters_path = args.output_dir / "fighter_current_strength.csv"
    summary_path = args.output_dir / "summary.json"

    fights.to_csv(fights_path, index=False)
    fighters.to_csv(fighters_path, index=False)

    scored_mask = fights["elo_pick_correct"].notna()
    scored = fights.loc[scored_mask]
    cold = scored[scored["cold_start_either_le2"]]
    established = scored[~scored["cold_start_either_le2"]]

    summary = {
        "source": str(args.input),
        "round_rows": int(len(rounds)),
        "bouts": int(len(bouts)),
        "scored_bouts": int(len(scored)),
        "unscored_or_unresolved_bouts": int(len(fights) - len(scored)),
        "fighters": int(len(fighters)),
        "base_rating": args.base_rating,
        "k_factor": args.k_factor,
        "elo_pick_accuracy_all": float(scored["elo_pick_correct"].mean()) if len(scored) else None,
        "elo_pick_accuracy_cold_start": float(cold["elo_pick_correct"].mean()) if len(cold) else None,
        "elo_pick_accuracy_established": float(established["elo_pick_correct"].mean()) if len(established) else None,
        "cold_start_definition": "either fighter has <=2 prior UFC fights at cutoff",
        "leakage_rule": "all exported fight features are captured before the current bout updates either fighter",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {fights_path}")
    print(f"Wrote {fighters_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
