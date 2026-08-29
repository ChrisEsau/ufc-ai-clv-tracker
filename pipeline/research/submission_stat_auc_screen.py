#!/usr/bin/env python3
"""Leakage-safe univariate screen of recorded UFC round stats for future submission outcomes.

Research-only. No Brain, FSR, or market inputs.

For every discoverable red/blue stat pair in ufc_round_stats.parquet, derive a
parseable numeric bout-level value, then build chronological pre-fight career
summaries. Screen each stat against two targets:

1) side-specific: this fighter wins the upcoming fight by submission
2) fight-level: the upcoming fight ends by submission (either side)

For side-specific prediction we test:
- own: fighter's prior career mean for the stat
- opp_allowed: opponent's prior career mean allowed for the same stat
- matchup_mean: mean(own, opp_allowed)
- matchup_sum: own + opp_allowed
- matchup_diff: own - opp_allowed

For fight-level SUB we combine the two side matchup_mean values with mean/max/sum.
All features for a bout are captured before that bout's statistics are added.
AUC is reported both signed and direction-free discrimination=max(AUC,1-AUC).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts

SIDE_PREFIXES = [("r_", "b_"), ("red_", "blue_"), ("R_", "B_"), ("RED_", "BLUE_")]
ID_BASES = {"fighter", "fighter_name", "name", "bout_id", "round", "date", "event", "method", "winner", "weight_class"}
EPS = 1e-12


def method_family(method: str) -> str | None:
    m = str(method or "").lower()
    if "submission" in m or re.search(r"\bsub\b", m):
        return "SUB"
    if "decision" in m:
        return "DEC"
    if "ko" in m or "tko" in m:
        return "KO"
    return None


def auc_binary(y, score):
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    y = y[mask].astype(int); s = s[mask]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, len(y), n1, n0
    # Mann-Whitney / rank-sum AUC; average ranks handle ties.
    r = pd.Series(s).rank(method="average").to_numpy(float)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc), len(y), n1, n0


def parse_value(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "--", "n/a"}:
        return np.nan
    if s.endswith("%"):
        try: return float(s[:-1].strip()) / 100.0
        except Exception: return np.nan
    if re.fullmatch(r"\d{1,3}:\d{2}", s):
        mm, ss = s.split(":")
        return float(int(mm) * 60 + int(ss))
    try:
        return float(s.replace(",", ""))
    except Exception:
        return np.nan


def parse_of_value(x):
    if x is None:
        return (np.nan, np.nan)
    s = str(x).strip().lower()
    m = re.fullmatch(r"\s*([\d.]+)\s+of\s+([\d.]+)\s*", s)
    if not m:
        return (np.nan, np.nan)
    return float(m.group(1)), float(m.group(2))


def discover_pairs(df: pd.DataFrame):
    cols = list(df.columns); colset = set(cols); out = []
    seen = set()
    for rp, bp in SIDE_PREFIXES:
        for c in cols:
            if not c.startswith(rp):
                continue
            base = c[len(rp):]
            if base.lower() in ID_BASES:
                continue
            bc = bp + base
            if bc in colset and (c, bc) not in seen:
                out.append((base, c, bc)); seen.add((c, bc))
    return out


def derive_pair_features(df: pd.DataFrame, base: str, rc: str, bc: str):
    """Return list of (feature_name, red_series, blue_series) for one raw pair."""
    rr = df[rc]; bb = df[bc]
    # Detect 'X of Y' representation using a sample from either side.
    sample = pd.concat([rr, bb], ignore_index=True).dropna().astype(str).head(500)
    of_frac = float(sample.str.contains(r"^\s*[\d.]+\s+of\s+[\d.]+\s*$", regex=True).mean()) if len(sample) else 0.0
    feats = []
    if of_frac >= 0.5:
        rparsed = rr.map(parse_of_value); bparsed = bb.map(parse_of_value)
        feats.append((base + "__landed", rparsed.map(lambda z: z[0]), bparsed.map(lambda z: z[0])))
        feats.append((base + "__attempted", rparsed.map(lambda z: z[1]), bparsed.map(lambda z: z[1])))
        return feats
    rnum = rr.map(parse_value); bnum = bb.map(parse_value)
    valid = pd.concat([rnum, bnum], ignore_index=True).notna().mean()
    if valid >= 0.25:
        feats.append((base, rnum, bnum))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-stats", type=Path, default=Path("data/fight_details/ufc_round_stats.parquet"))
    ap.add_argument("--master", type=Path, default=Path("data/master/ufc_master.parquet"))
    ap.add_argument("--holdout-from", default="2025-01-01")
    ap.add_argument("--min-prior-fights", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/submission_stat_auc_screen"))
    args = ap.parse_args()

    raw = pd.read_parquet(args.round_stats)
    bouts = build_bouts(pd.read_parquet(args.master)).copy()
    if "bout_id" not in raw.columns:
        raise RuntimeError("round stats parquet has no bout_id; cannot leakage-safely align rounds to fights")

    pairs = discover_pairs(raw)
    if not pairs:
        raise RuntimeError(f"No red/blue stat pairs discovered. Columns: {list(raw.columns)}")

    # Derive all parseable numeric side-paired round stats, then sum to bout level.
    derived = {"bout_id": raw["bout_id"].astype(str)}
    feature_names = []
    source_map = {}
    for base, rc, bc in pairs:
        for fname, rs, bs in derive_pair_features(raw, base, rc, bc):
            rname = "R__" + fname; bname = "B__" + fname
            derived[rname] = rs.astype(float); derived[bname] = bs.astype(float)
            feature_names.append(fname); source_map[fname] = {"red_column": rc, "blue_column": bc}
    d = pd.DataFrame(derived)
    if not feature_names:
        raise RuntimeError("Side-paired columns were found but none were parseable as numeric stats")

    agg_cols = [c for c in d.columns if c != "bout_id"]
    bout_stats = d.groupby("bout_id", as_index=False)[agg_cols].sum(min_count=1)
    bouts["bout_id"] = bouts["bout_id"].astype(str)
    bouts = bouts.merge(bout_stats, on="bout_id", how="left")
    bouts = bouts.sort_values(["date", "bout_id"]).reset_index(drop=True)

    # Career stores: per fighter, per feature [sum_own, sum_allowed, n].
    career = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    side_rows = []
    fight_rows = []

    for b in bouts.itertuples(index=False):
        r, bl = b.red_fighter, b.blue_fighter
        am = method_family(getattr(b, "method", ""))
        r_subwin = int(am == "SUB" and b.winner == r)
        b_subwin = int(am == "SUB" and b.winner == bl)
        fight_sub = int(am == "SUB")

        side_feature_cache = {"R": {}, "B": {}}
        for fname in feature_names:
            for side, fighter, opp in (("R", r, bl), ("B", bl, r)):
                cf = career[fighter][fname]; co = career[opp][fname]
                own = cf[0] / cf[2] if cf[2] >= args.min_prior_fights else np.nan
                opp_allowed = co[1] / co[2] if co[2] >= args.min_prior_fights else np.nan
                side_feature_cache[side][fname] = (own, opp_allowed)

        for side, fighter, opp, y in (("R", r, bl, r_subwin), ("B", bl, r, b_subwin)):
            row = {"date": b.date, "bout_id": b.bout_id, "side": side, "fighter": fighter, "opponent": opp, "y_sub_win": y}
            for fname in feature_names:
                own, allowed = side_feature_cache[side][fname]
                row[f"{fname}__own"] = own
                row[f"{fname}__opp_allowed"] = allowed
                row[f"{fname}__matchup_mean"] = np.nanmean([own, allowed]) if np.isfinite(own) or np.isfinite(allowed) else np.nan
                row[f"{fname}__matchup_sum"] = own + allowed if np.isfinite(own) and np.isfinite(allowed) else np.nan
                row[f"{fname}__matchup_diff"] = own - allowed if np.isfinite(own) and np.isfinite(allowed) else np.nan
            side_rows.append(row)

        frow = {"date": b.date, "bout_id": b.bout_id, "y_fight_sub": fight_sub}
        for fname in feature_names:
            vals = []
            for side in ("R", "B"):
                own, allowed = side_feature_cache[side][fname]
                vals.append(np.nanmean([own, allowed]) if np.isfinite(own) or np.isfinite(allowed) else np.nan)
            finite = [x for x in vals if np.isfinite(x)]
            frow[f"{fname}__fight_mean"] = float(np.mean(finite)) if finite else np.nan
            frow[f"{fname}__fight_max"] = float(np.max(finite)) if finite else np.nan
            frow[f"{fname}__fight_sum"] = float(np.sum(finite)) if len(finite) == 2 else np.nan
        fight_rows.append(frow)

        # Update careers only after prediction rows have been captured.
        for fname in feature_names:
            rv = getattr(b, "R__" + fname, np.nan); bv = getattr(b, "B__" + fname, np.nan)
            if np.isfinite(rv) and np.isfinite(bv):
                cr = career[r][fname]; cb = career[bl][fname]
                cr[0] += float(rv); cr[1] += float(bv); cr[2] += 1
                cb[0] += float(bv); cb[1] += float(rv); cb[2] += 1

    sides = pd.DataFrame(side_rows); fights = pd.DataFrame(fight_rows)
    cutoff = pd.Timestamp(args.holdout_from)

    results = []
    def screen(frame, ycol, scope):
        for col in frame.columns:
            if col in {"date", "bout_id", "side", "fighter", "opponent", ycol} or not pd.api.types.is_numeric_dtype(frame[col]):
                continue
            fname = col.split("__")[0]
            transform = "__".join(col.split("__")[1:])
            for split_name, mask in (("train", frame.date < cutoff), ("holdout", frame.date >= cutoff)):
                auc, n, n1, n0 = auc_binary(frame.loc[mask, ycol].to_numpy(), frame.loc[mask, col].to_numpy())
                if not np.isfinite(auc):
                    continue
                direction = 1 if auc >= 0.5 else -1
                results.append({"scope": scope, "feature": fname, "transform": transform, "split": split_name, "auc": auc,
                                "discrimination_auc": max(auc, 1.0-auc), "direction": direction, "n": n, "positives": n1, "negatives": n0})

    screen(sides, "y_sub_win", "side_sub_win")
    screen(fights, "y_fight_sub", "fight_sub")
    res = pd.DataFrame(results)

    # Best holdout transform per raw stat and scope; include train AUC for the same transform.
    hold = res[res.split == "holdout"].copy()
    hold = hold[hold.n >= 100]
    idx = hold.groupby(["scope", "feature"])["discrimination_auc"].idxmax()
    best = hold.loc[idx].sort_values(["scope", "discrimination_auc"], ascending=[True, False]).reset_index(drop=True)
    train_lookup = res[res.split == "train"].set_index(["scope", "feature", "transform"])
    best["train_auc_same_transform"] = [
        train_lookup.loc[(r.scope, r.feature, r.transform), "auc"] if (r.scope, r.feature, r.transform) in train_lookup.index else np.nan
        for r in best.itertuples(index=False)
    ]
    best["train_discrimination_same_transform"] = best.train_auc_same_transform.map(lambda x: max(x,1-x) if np.isfinite(x) else np.nan)
    best["source_red_column"] = best.feature.map(lambda x: source_map.get(x, {}).get("red_column"))
    best["source_blue_column"] = best.feature.map(lambda x: source_map.get(x, {}).get("blue_column"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "all_auc_results.csv", index=False)
    best.to_csv(args.output_dir / "best_stat_auc_rankings.csv", index=False)
    sides.to_csv(args.output_dir / "side_prefight_features.csv", index=False)
    fights.to_csv(args.output_dir / "fight_prefight_features.csv", index=False)
    with open(args.output_dir / "schema.json", "w") as f:
        json.dump({"round_stats_columns": list(raw.columns), "discovered_pairs": pairs, "derived_features": feature_names, "source_map": source_map}, f, indent=2)

    print("Derived stat features:", len(feature_names))
    for scope in ("fight_sub", "side_sub_win"):
        print("\nTOP", scope.upper())
        print(best[best.scope == scope].head(30).to_string(index=False))

if __name__ == "__main__":
    main()
