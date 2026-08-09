"""Shadow Elo-style FSR family for clinch striking.

Adds three candidate persistent ratings:
- clinch_striking_pressure
- clinch_striking_precision
- clinch_striking_defense

Mirrors the locked ground-striking family. Pressure is a population-centered
fighter tendency; precision and defense are opponent-adjusted paired abilities.
All population pools/baselines are prior-date only and same-date updates are
simultaneous. Shadow/research only.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from math import exp, isfinite, log, sqrt
from pathlib import Path

import pandas as pd

RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_25_shadow")
BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4

SKILLS = (
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
)

C = {
    "clinch_attempts_per_round": "rfs_phase_base_fight_clinch_attempts_per_round",
    "clinch_attempt_share": "rfs_phase_base_fight_clinch_attempt_share",
    "clinch_attempts": "rfs_phase_interact_fight_clinch_attempts",
    "clinch_accuracy": "rfs_phase_interact_fight_clinch_accuracy",
    "opp_clinch_attempts": "rfs_phase_interact_fight_opp_clinch_attempts",
    "clinch_accuracy_allowed": "rfs_phase_interact_fight_clinch_accuracy_allowed",
}
POOL_KEYS = (
    "clinch_attempts_per_round",
    "clinch_attempt_share",
    "clinch_accuracy",
    "clinch_accuracy_allowed",
)

def finite(value):
    if value is None or pd.isna(value): return None
    try: out = float(value)
    except (TypeError, ValueError): return None
    return out if isfinite(out) else None

def clamp(v, lo, hi): return max(lo, min(hi, v))
def sigmoid(v): return 1.0 / (1.0 + exp(-v))
def logit(p):
    p = clamp(float(p), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return log(p / (1.0 - p))
def k_factor(n): return BASE_K / sqrt(1.0 + float(n) / 6.0)
def q_exp(u): return 1.0 - exp(-max(0.0, float(u)))
def percentile(pool, value):
    if value is None: return None
    if not pool: return 0.5
    return bisect_right(pool, float(value)) / len(pool)
def row_value(row, key): return finite(row.get(C[key]))
def weighted_available(parts):
    a = [(w, v) for w, v in parts if v is not None]
    if not a: return None
    total = sum(w for w, _ in a)
    return sum(w * float(v) for w, v in a) / total if total > 0 else None

def population_baseline(weighted_sum, quality_sum, skill):
    q = float(quality_sum[skill])
    return 0.50 if q <= 0 else clamp(float(weighted_sum[skill]) / q, 0.0, 1.0)
def expected_intrinsic(rating, baseline):
    return sigmoid(logit(baseline) + (float(rating) - BASE_RATING) / RATING_SCALE)
def expected_matchup(offense, defense, baseline):
    return sigmoid(logit(baseline) + (float(offense) - float(defense)) / RATING_SCALE)

def observation_bundle(row, pools):
    attempts = row_value(row, "clinch_attempts") or 0.0
    opp_attempts = row_value(row, "opp_clinch_attempts") or 0.0
    pressure = weighted_available((
        (0.60, percentile(pools["clinch_attempts_per_round"], row_value(row, "clinch_attempts_per_round"))),
        (0.40, percentile(pools["clinch_attempt_share"], row_value(row, "clinch_attempt_share"))),
    ))
    pq = q_exp(attempts / 10.0) if attempts > 0 else 0.0
    precision = percentile(pools["clinch_accuracy"], row_value(row, "clinch_accuracy"))
    allowed = percentile(pools["clinch_accuracy_allowed"], row_value(row, "clinch_accuracy_allowed"))
    defense = None if allowed is None else 1.0 - allowed
    dq = q_exp(opp_attempts / 10.0) if opp_attempts > 0 else 0.0
    return {
        "clinch_striking_pressure": (pressure if pq > 0 else None, pq),
        "clinch_striking_precision": (precision if pq > 0 else None, pq),
        "clinch_striking_defense": (defense if dq > 0 else None, dq),
    }

def append_date_to_pools(date_rows, pools):
    for _, row in date_rows.iterrows():
        for key in POOL_KEYS:
            value = row_value(row, key)
            if value is not None: insort(pools[key], value)

def validate_columns(df):
    required = {"fight_id", "fighter_id", "fighter_name", *C.values()}
    if "date" not in df.columns and "event_date" not in df.columns: required.add("date")
    missing = sorted(c for c in required if c not in df.columns)
    if missing: raise ValueError(f"RFS history missing required clinch-striking FSR columns: {missing}")

def build_prefight_snapshots(rfs: pd.DataFrame) -> pd.DataFrame:
    validate_columns(rfs)
    df = rfs.copy()
    date_col = "date" if "date" in df.columns else "event_date"
    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    df = df.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings = defaultdict(lambda: {s: BASE_RATING for s in SKILLS})
    updates = defaultdict(lambda: {s: 0 for s in SKILLS})
    fights = defaultdict(int)
    pools = {k: [] for k in POOL_KEYS}
    weighted_sum = defaultdict(float)
    quality_sum = defaultdict(float)
    snapshots = []

    for fight_date, date_rows in df.groupby("date", sort=True):
        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2: continue
            for row in fight.itertuples(index=False):
                fid = str(row.fighter_id); _ = ratings[fid]
                snap = {"fight_id": str(fight_id), "date": pd.Timestamp(fight_date),
                        "fighter_id": fid, "fighter_name": str(row.fighter_name),
                        "prior_ufc_fights": int(fights[fid])}
                snap.update({s: float(ratings[fid][s]) for s in SKILLS})
                snap.update({f"{s}_updates": int(updates[fid][s]) for s in SKILLS})
                snapshots.append(snap)

        deltas = defaultdict(lambda: {s: 0.0 for s in SKILLS})
        date_updates = defaultdict(lambda: {s: 0 for s in SKILLS})
        date_fights = defaultdict(int)
        date_weighted = defaultdict(float)
        date_quality = defaultdict(float)

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2: continue
            rows = [r for _, r in fight.iterrows()]
            for i, row in enumerate(rows):
                opp = rows[1-i]
                fid, oid = str(row["fighter_id"]), str(opp["fighter_id"])
                _ = ratings[fid]; _ = ratings[oid]
                bundle = observation_bundle(row, pools)
                for skill in SKILLS:
                    obs, q = bundle[skill]
                    if obs is None or q <= 0: continue
                    baseline = population_baseline(weighted_sum, quality_sum, skill)
                    if skill == "clinch_striking_pressure":
                        expected = expected_intrinsic(ratings[fid][skill], baseline)
                    elif skill == "clinch_striking_precision":
                        expected = expected_matchup(ratings[fid][skill], ratings[oid]["clinch_striking_defense"], baseline)
                    else:
                        expected = expected_matchup(ratings[fid][skill], ratings[oid]["clinch_striking_precision"], baseline)
                    delta = k_factor(updates[fid][skill]) * q * (float(obs) - expected)
                    deltas[fid][skill] += delta
                    date_updates[fid][skill] += 1
                    date_weighted[skill] += q * float(obs)
                    date_quality[skill] += q
                date_fights[fid] += 1

        for fid, skill_deltas in deltas.items():
            for skill, delta in skill_deltas.items():
                ratings[fid][skill] = clamp(ratings[fid][skill] + delta, MIN_RATING, MAX_RATING)
                updates[fid][skill] += date_updates[fid][skill]
        for fid, count in date_fights.items(): fights[fid] += count
        for skill in SKILLS:
            weighted_sum[skill] += date_weighted[skill]
            quality_sum[skill] += date_quality[skill]
        append_date_to_pools(date_rows, pools)

    out = pd.DataFrame(snapshots)
    if out.empty: raise RuntimeError("clinch-striking FSR replay produced no snapshots")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("clinch-striking FSR snapshots violate fighter-fight grain")
    return out

def main():
    if not RFS_PATH.exists(): raise RuntimeError(f"RFS history not found: {RFS_PATH}")
    snapshots = build_prefight_snapshots(pd.read_parquet(RFS_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "clinch_striking_fsr_v1_prefight_snapshots.parquet"
    snapshots.to_parquet(path, index=False)
    print(f"Wrote {len(snapshots):,} clinch-striking FSR pre-fight rows to {path}")

if __name__ == "__main__": main()
