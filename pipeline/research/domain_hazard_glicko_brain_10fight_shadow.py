#!/usr/bin/env python3
"""Research-only domain hazard Glicko shadow for locked Brain KO/SUB clocks.

Builds separate attacker/defender Gaussian ratings from censored finish exposure:
  KO_OFF / KO_DEF from KO/TKO finish-time survival
  SUB_OFF / SUB_DEF from submission finish-time survival

The update is Glicko-style (central rating + RD) but uses the exponential survival
likelihood already validated by the locked KO/SUB clocks instead of a binary fight-win
likelihood. Same-date fights are delayed as one batch. RD is estimated and recorded,
but only the central rating enters the Brain clock in this first shadow.

Only target-fight KO/SUB prefight clock rows in a temporary locked-bundle copy are
modified. Standing/wrestling/ground FSR, judging, seeds, and mechanics are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.elo_fsr_brain_one_fight_shadow import _target_indices, run_locked
from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES
from pipeline.research import ko_time_survival_oos as ko_surv
from pipeline.research import sub_time_survival_oos as sub_surv

FIGHT_IDS = [
    "419fff06f338f5c6", "58ffa2dac4f2e7d0", "5c69b019e6deee41", "5d2eedd05081ed23",
    "20d74ed23d3e9b3a", "44cfbb8c3c356c65", "b0474597b2c60482", "b23a1a5d35eb438a",
    "33afdd7ad43a2756", "7208e40818401e88",
]
PATHS = 500
ROOT = Path("data/diagnostics/domain_hazard_glicko_brain_10fight_shadow")
BASE = 1500.0
INIT_RD = 250.0
MIN_RD = 45.0
MAX_STEP = 175.0
Q = math.log(10.0) / 400.0
# Weak pre-UFC-scale hazard priors only stabilize the earliest history; by 2026 cumulative data dominate.
KO_PRIOR_EVENTS, KO_PRIOR_SECONDS = 2.0, 3600.0
SUB_PRIOR_EVENTS, SUB_PRIOR_SECONDS = 1.0, 3600.0


def sha256(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def _g(rd_a: float, rd_b: float) -> float:
    rd2 = rd_a * rd_a + rd_b * rd_b
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd2 / (math.pi * math.pi))


def build_hazard_ratings(ff: pd.DataFrame, event_col: str, *, prior_events: float, prior_seconds: float) -> pd.DataFrame:
    """Same-date-delayed online Gaussian hazard ratings using censored exposure likelihood."""
    ff = ff.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True).copy()
    off = defaultdict(lambda: [BASE, INIT_RD])
    deff = defaultdict(lambda: [BASE, INIT_RD])
    cum_events = float(prior_events)
    cum_seconds = float(prior_seconds)
    rows = []

    for date, batch in ff.groupby("event_date", sort=True):
        p0 = max(cum_events / max(cum_seconds, 1.0), 1e-9)
        staged = []
        # Snapshot all states before any same-date update.
        for r in batch.itertuples(index=False):
            fid, oid = str(r.fighter_id), str(r.opponent_id)
            ro, rdo = off[fid]
            rd, rdd = deff[oid]
            g = _g(rdo, rdd)
            log_rr = Q * g * (ro - rd)
            rr = float(np.clip(math.exp(log_rr), 0.05, 20.0))
            t = float(r.fight_seconds)
            y = float(getattr(r, event_col))
            H = max(p0 * t * rr, 1e-12)
            resid = y - H
            curv = (Q * g) ** 2 * H
            po = 1.0 / (rdo * rdo) + curv
            pd_ = 1.0 / (rdd * rdd) + curv
            step_o = float(np.clip((Q * g * resid) / po, -MAX_STEP, MAX_STEP))
            step_d = float(np.clip((-Q * g * resid) / pd_, -MAX_STEP, MAX_STEP))
            new_ro = ro + step_o
            new_rd = rd + step_d
            new_rdo = max(MIN_RD, math.sqrt(1.0 / po))
            new_rdd = max(MIN_RD, math.sqrt(1.0 / pd_))
            rows.append({
                "event_date": pd.Timestamp(date), "fight_id": str(r.fight_id),
                "fighter_id": fid, "fighter_name": str(r.fighter_name), "opponent_id": oid,
                "off_rating": ro, "off_rd": rdo, "opp_def_rating": rd, "opp_def_rd": rdd,
                "population_hazard": p0, "matchup_rr": rr,
            })
            staged.append((fid, oid, new_ro, new_rdo, new_rd, new_rdd))
        # Apply all updates after snapshots to enforce same-date delay.
        for fid, oid, ro2, rdo2, rd2, rdd2 in staged:
            off[fid] = [ro2, rdo2]
            deff[oid] = [rd2, rdd2]
        # Each fighter-side contributes exposure; each finish contributes exactly one event side.
        cum_seconds += float(batch.fight_seconds.sum())
        cum_events += float(batch[event_col].sum())
    return pd.DataFrame(rows)


def _set_clock_from_ratings(frame: pd.DataFrame, idx: int, *, off_rating: float, def_rating: float,
                            p0: float, prior_events: float, audit: list[dict], fighter: str, label: str) -> None:
    """Encode desired attacker/vulnerability rates into the existing locked lookup row.

    Desired attacker rate = p0 * exp(Q*(off-BASE)); defender vulnerability =
    p0 * exp(Q*(BASE-def)). We use a large pseudo exposure solely to make the locked
    lookup reproduce those rates; the population/date piecewise baseline is unchanged.
    """
    att = p0 * math.exp(Q * (off_rating - BASE))
    vuln = p0 * math.exp(Q * (BASE - def_rating))
    prior_sec = prior_events / p0
    pseudo = 100000.0
    win = max(att * (pseudo + prior_sec) - prior_events, 0.0)
    loss = max(vuln * (pseudo + prior_sec) - prior_events, 0.0)
    event = "ko" if label == "ko_clock" else "sub"
    changes = {
        "prior_seconds": pseudo,
        f"prior_{event}_win": win,
        "opp_prior_seconds": pseudo,
        f"opp_prior_{event}_loss": loss,
    }
    for col, after in changes.items():
        before = float(frame.at[idx, col])
        frame.at[idx, col] = after
        audit.append({"fighter": fighter, "system": label, "field": col,
                      "transform": "domain_hazard_glicko_clock_encoding", "before": before, "after": after})


def _target_rating(ratings: pd.DataFrame, fight_id: str, fighter_name: str) -> pd.Series:
    x = ratings[ratings.fight_id.astype(str).eq(str(fight_id)) & ratings.fighter_name.astype(str).eq(str(fighter_name))]
    if len(x) != 1:
        raise RuntimeError(f"expected one rating row for {fight_id}/{fighter_name}, found {len(x)}")
    return x.iloc[0]


def run_one(fid: str, ko_r: pd.DataFrame, sub_r: pd.DataFrame) -> dict:
    ko_raw = ko_surv.load_fighter_fights()
    target = ko_raw[ko_raw.fight_id.astype(str).eq(fid)]
    if len(target) != 2: raise RuntimeError(f"expected two target sides for {fid}, got {len(target)}")
    a, b = [str(x) for x in target.fighter_name.tolist()]
    actual = str(target.loc[target.ko_event.eq(1), "fighter_name"].iloc[0]) if target.ko_event.sum() else None
    # Actual winner irrespective of method.
    winner_rows = target[target.won]
    if len(winner_rows) == 1: actual = str(winner_rows.fighter_name.iloc[0])

    out = ROOT / fid; out.mkdir(parents=True, exist_ok=True)
    bundle = out / "adjusted_bundle"
    if bundle.exists(): shutil.rmtree(bundle)
    shutil.copytree(Path(DEFAULT_BUNDLE_DIR), bundle)
    audit = []

    kp = bundle / FILES["ko_prefight"]; k = pd.read_parquet(kp)
    kia, kib = _target_indices(k, fid, a, b)
    kra, krb = _target_rating(ko_r, fid, a), _target_rating(ko_r, fid, b)
    # Row A's opponent-defense rating is B's defense; row B already contains A's defense snapshot.
    _set_clock_from_ratings(k, kia, off_rating=float(kra.off_rating), def_rating=float(kra.opp_def_rating),
                            p0=float(kra.population_hazard), prior_events=2.0, audit=audit, fighter=a, label="ko_clock")
    _set_clock_from_ratings(k, kib, off_rating=float(krb.off_rating), def_rating=float(krb.opp_def_rating),
                            p0=float(krb.population_hazard), prior_events=2.0, audit=audit, fighter=b, label="ko_clock")
    k.to_parquet(kp, index=False)

    sp = bundle / FILES["sub_prefight"]; s = pd.read_parquet(sp)
    sia, sib = _target_indices(s, fid, a, b)
    sra, srb = _target_rating(sub_r, fid, a), _target_rating(sub_r, fid, b)
    _set_clock_from_ratings(s, sia, off_rating=float(sra.off_rating), def_rating=float(sra.opp_def_rating),
                            p0=float(sra.population_hazard), prior_events=1.0, audit=audit, fighter=a, label="sub_clock")
    _set_clock_from_ratings(s, sib, off_rating=float(srb.off_rating), def_rating=float(srb.opp_def_rating),
                            p0=float(srb.population_hazard), prior_events=1.0, audit=audit, fighter=b, label="sub_clock")
    s.to_parquet(sp, index=False)

    mp = bundle / "manifest.json"; m = json.loads(mp.read_text())
    for key, p in (("ko_prefight", kp), ("sub_prefight", sp)): m["files"][key]["sha256"] = sha256(p)
    m["research_shadow"] = {
        "type": "domain_hazard_glicko_ko_sub_clock_only", "fight_id": fid,
        "ko_off_a": float(kra.off_rating), "ko_def_b": float(kra.opp_def_rating),
        "ko_off_b": float(krb.off_rating), "ko_def_a": float(krb.opp_def_rating),
        "sub_off_a": float(sra.off_rating), "sub_def_b": float(sra.opp_def_rating),
        "sub_off_b": float(srb.off_rating), "sub_def_a": float(srb.opp_def_rating),
        "rd_used_in_brain": False, "production_changed": False,
    }
    mp.write_text(json.dumps(m, indent=2) + "\n")

    adjusted = run_locked(fid, bundle, PATHS, out / "adjusted_results.json")
    pd.DataFrame(audit).assign(fight_id=fid).to_csv(out / "domain_hazard_glicko_adjustments.csv", index=False)
    summary = {
        "fight_id": fid, "fighter_a": a, "fighter_b": b, "actual_winner": actual, "paths": PATHS,
        "ko": {
            a: {"off": float(kra.off_rating), "off_rd": float(kra.off_rd), "opp_def": float(kra.opp_def_rating), "opp_def_rd": float(kra.opp_def_rd), "rr": float(kra.matchup_rr)},
            b: {"off": float(krb.off_rating), "off_rd": float(krb.off_rd), "opp_def": float(krb.opp_def_rating), "opp_def_rd": float(krb.opp_def_rd), "rr": float(krb.matchup_rr)},
        },
        "sub": {
            a: {"off": float(sra.off_rating), "off_rd": float(sra.off_rd), "opp_def": float(sra.opp_def_rating), "opp_def_rd": float(sra.opp_def_rd), "rr": float(sra.matchup_rr)},
            b: {"off": float(srb.off_rating), "off_rd": float(srb.off_rd), "opp_def": float(srb.opp_def_rating), "opp_def_rd": float(srb.opp_def_rd), "rr": float(srb.matchup_rr)},
        },
        "adjusted": adjusted,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    shutil.rmtree(bundle)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fight-id", required=True, choices=FIGHT_IDS); args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    ko_ff = ko_surv.load_fighter_fights()
    sub_ff = sub_surv.load_fighter_fights()
    ko_r = build_hazard_ratings(ko_ff, "ko_event", prior_events=KO_PRIOR_EVENTS, prior_seconds=KO_PRIOR_SECONDS)
    sub_r = build_hazard_ratings(sub_ff, "sub_event", prior_events=SUB_PRIOR_EVENTS, prior_seconds=SUB_PRIOR_SECONDS)
    run_one(args.fight_id, ko_r, sub_r)

if __name__ == "__main__": main()
