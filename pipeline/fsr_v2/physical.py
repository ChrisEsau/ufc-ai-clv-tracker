"""Self-contained leakage-safe FSR V2 physical traits from raw UFC data.

The equations are the frozen FSR-32 stamina, fresh-power, and finish-reservoir
contracts.  Only their old enriched-RFS data boundary has been replaced: all
fight observations are reconstructed in memory from authoritative round stats
and master outcome metadata.
"""
from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH

STAMINA_CAPACITY = 100.0
PHYSICAL_COLUMNS = (
    "stamina_capacity", "stamina_depletion_resistance",
    "stamina_performance_resilience", "striking_power",
    "damage_durability", "knockdown_resistance",
)
LEARNED_PHYSICAL_COLUMNS = PHYSICAL_COLUMNS[1:]

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0
RATING_SCALE = 12.0
BASE_K = 7.0
LOGIT_EPSILON = 1e-4  # frozen dynamic-family value
CONFIDENCE_FIGHTS = 3.0
MIN_POPULATION_OBSERVATIONS = 100
KD_HIGH_EXPOSURE_QUANTILE = 0.67
DURABILITY_HIGH_EXPOSURE_QUANTILE = 0.70

POWER_MIN = 35.0
POWER_MAX = 90.0

# Frozen paired-power architecture.
#
# Historical power is learned as damaging-event production relative to the
# opponent's internally learned resistance. Age is deliberately NOT embedded
# here; current-age translation belongs at matchup/simulator time.
POWER_KO_WEIGHT = 0.50
POWER_UPDATE_K = 1.0
POWER_EVIDENCE_SATURATION = 20.0
POWER_BASE_PRIOR_RATE = 0.008
POWER_BASE_PRIOR_SIG = 1000.0

# Direct monotonic translation from validated paired raw state to the existing
# public 35-90 FSR striking-power contract.
POWER_RATING_SCALE = 200.0

STAMINA_POOL_MAP = {
    "far_late_early_workload": "late_early_workload_ratio",
    "far_sig_first_last": "sig_attempt_first_last_ratio",
    "far_total_first_last": "total_attempt_first_last_ratio",
    "far_sig_slope": "sig_attempt_slope",
    "far_total_slope": "total_attempt_slope",
    "far_td_slope": "td_attempt_slope",
    "far_control_slope": "control_slope",
    "fpr_late_early_output": "late_early_output_ratio",
    "fpr_sig_accuracy_change": "sig_accuracy_change",
    "fpr_total_accuracy_change": "total_accuracy_change",
    "fpr_sig_landed_slope": "sig_landed_slope",
    "fpr_total_landed_slope": "total_landed_slope",
}

@dataclass(frozen=True)
class PhysicalSnapshots:
    prefight: pd.DataFrame
    latest: pd.DataFrame


def _finite(value: object, default: float | None = None) -> float | None:
    if value is None or pd.isna(value): return default
    try: result = float(value)
    except (TypeError, ValueError): return default
    return result if isfinite(result) else default


def _safe_ratio(numerator: object, denominator: object) -> float:
    n, d = _finite(numerator), _finite(denominator)
    return np.nan if n is None or d is None or d <= 0 else n / d


def _mean_available(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _ols_slope(group: pd.DataFrame, column: str) -> float:
    x = pd.to_numeric(group["round"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[valid], y[valid], 1)[0]) if valid.sum() >= 2 and np.unique(x[valid]).size >= 2 else np.nan


def _first_last_ratio(group: pd.DataFrame, column: str) -> float:
    ordered = group.sort_values("round")
    return _safe_ratio(ordered.iloc[-1][column], ordered.iloc[0][column])


def _accuracy_change(group: pd.DataFrame, landed: str, attempted: str) -> float:
    ordered = group.sort_values("round")
    first = _safe_ratio(ordered.iloc[0][landed], ordered.iloc[0][attempted])
    last = _safe_ratio(ordered.iloc[-1][landed], ordered.iloc[-1][attempted])
    return last - first if np.isfinite(first) and np.isfinite(last) else np.nan


def build_physical_observations(rounds: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the exact old fight-level inputs without persisting RFS."""
    required = {"event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "round",
        "kd", "sig_str_landed", "sig_str_attempted", "total_str_landed", "total_str_attempted",
        "td_landed", "td_attempted", "ctrl_sec", "head_landed", "ground_landed"}
    missing = sorted(required - set(rounds.columns))
    if missing: raise ValueError(f"round stats missing physical source columns: {missing}")
    master_required = {"fight_id", "date", "method", "winner_id"}
    missing = sorted(master_required - set(master.columns))
    if missing: raise ValueError(f"master missing physical metadata columns: {missing}")

    r = rounds.copy()
    r["fight_id"] = r["fight_id"].astype(str); r["fighter_id"] = r["fighter_id"].astype(str); r["opponent_id"] = r["opponent_id"].astype(str)
    numeric = required - {"event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id"}
    for column in numeric: r[column] = pd.to_numeric(r[column], errors="coerce")
    opponent = r[["fight_id", "round", "fighter_id", "kd", "sig_str_landed", "head_landed", "ground_landed", "ctrl_sec"]].rename(
        columns={"fighter_id": "opponent_id", "kd": "opponent_kd", "sig_str_landed": "opponent_sig_landed",
                 "head_landed": "opponent_head_landed", "ground_landed": "opponent_ground_landed",
                 "ctrl_sec": "opponent_ctrl_sec"})
    r = r.merge(opponent, on=["fight_id", "round", "opponent_id"], how="left", validate="one_to_one")
    meta = master[list(master_required)].copy(); meta["fight_id"] = meta["fight_id"].astype(str)
    meta["date"] = pd.to_datetime(meta["date"], errors="raise")
    r = r.merge(meta, on="fight_id", how="left", validate="many_to_one")
    if r["date"].isna().any() or r["opponent_kd"].isna().any():
        raise ValueError("physical observations have missing master or reciprocal round rows")

    output = []
    for (_, _), group in r.groupby(["fight_id", "fighter_id"], sort=False):
        group = group.sort_values("round")
        first = group.iloc[0]; rounds_observed = float(group["round"].nunique())
        row = {"fight_id": str(first.fight_id), "fighter_id": str(first.fighter_id),
               "fighter_name": str(first.fighter_name), "opponent_id": str(first.opponent_id),
               "date": pd.Timestamp(first.date), "rounds_observed": rounds_observed,
               "kd_scored": float(group.kd.sum()),
               "sig_landed": float(group.sig_str_landed.sum()),
               "sig_attempts": float(group.sig_str_attempted.sum()),
               "td_attempts": float(group.td_attempted.sum()), "control_seconds": float(group.ctrl_sec.sum()),
               "kd_absorbed": float(group.opponent_kd.sum()), "sig_absorbed": float(group.opponent_sig_landed.sum()),
               "head_absorbed": float(group.opponent_head_landed.sum()),
               "ground_absorbed": float(group.opponent_ground_landed.sum()),
               "opponent_control_seconds": float(group.opponent_ctrl_sec.sum())}
        for name, column in (("sig_attempt_slope", "sig_str_attempted"),
            ("total_attempt_slope", "total_str_attempted"), ("td_attempt_slope", "td_attempted"),
            ("control_slope", "ctrl_sec"), ("sig_landed_slope", "sig_str_landed"),
            ("total_landed_slope", "total_str_landed")):
            row[name] = _ols_slope(group, column)
        if len(group) < 2:
            for name in ("sig_attempt_first_last_ratio", "total_attempt_first_last_ratio",
                         "late_early_workload_ratio", "late_early_output_ratio",
                         "sig_accuracy_change", "total_accuracy_change"): row[name] = np.nan
        else:
            row["sig_attempt_first_last_ratio"] = _first_last_ratio(group, "sig_str_attempted")
            row["total_attempt_first_last_ratio"] = _first_last_ratio(group, "total_str_attempted")
            row["late_early_workload_ratio"] = _mean_available(_first_last_ratio(group, c) for c in
                ("sig_str_attempted", "total_str_attempted", "td_attempted", "ctrl_sec"))
            row["late_early_output_ratio"] = _mean_available(_first_last_ratio(group, c) for c in
                ("sig_str_landed", "total_str_landed", "td_landed", "ctrl_sec"))
            row["sig_accuracy_change"] = _accuracy_change(group, "sig_str_landed", "sig_str_attempted")
            row["total_accuracy_change"] = _accuracy_change(group, "total_str_landed", "total_str_attempted")
        method = str(first.method).upper(); winner = str(first.winner_id)
        is_ko = "KO" in method or "TKO" in method
        row["ko_loss"] = float(is_ko and str(first.fighter_id) != winner)
        row["ko_win"] = float(is_ko and str(first.fighter_id) == winner)
        output.append(row)
    return pd.DataFrame(output).sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _clamp(value, low, high): return max(low, min(high, float(value)))
def _sigmoid(value): return 1/(1+exp(-value))
def _logit(p):
    p = _clamp(p, LOGIT_EPSILON, 1-LOGIT_EPSILON); return log(p/(1-p))
def stamina_k(n): return BASE_K / sqrt(1 + n/6)
def stamina_quality(row):
    rounds = float(row.rounds_observed)
    if rounds < 2: return 0.0
    units = row.sig_attempts/60 + row.td_attempts/4 + row.control_seconds/180
    return _clamp((1-exp(-(rounds-1)/2)) * (1-exp(-max(0, units))), 0, 1)
def _percentile(pool, value):
    return None if not np.isfinite(value) else (0.5 if not pool else bisect_right(pool, float(value))/len(pool))
def _weighted(parts):
    available=[(w,v) for w,v in parts if v is not None]
    return None if not available else sum(w*v for w,v in available)/sum(w for w,_ in available)
def _stamina_observations(row, pools):
    pct=lambda key: _percentile(pools[key], float(row[STAMINA_POOL_MAP[key]]))
    far=_weighted(((.40,pct("far_late_early_workload")),(.20,pct("far_sig_first_last")),
        (.15,pct("far_total_first_last")),(.0625,pct("far_sig_slope")),(.0625,pct("far_total_slope")),
        (.0625,pct("far_td_slope")),(.0625,pct("far_control_slope"))))
    fpr=_weighted(((.40,pct("fpr_late_early_output")),(.25,pct("fpr_sig_accuracy_change")),
        (.15,pct("fpr_total_accuracy_change")),(.10,pct("fpr_sig_landed_slope")),
        (.10,pct("fpr_total_landed_slope"))))
    return far, fpr


def _new_resistance_state():
    return {k:0.0 for k in ("fights","kd_absorbed","sig_absorbed","kd_free_fights",
        "kd_high_exposure_fights","kd_free_high_exposure","dur_high_exposure_fights",
        "dur_high_survivals","dur_high_exposure_sum","dur_high_survived_exposure_sum",
        "survived_exposure_sum","survived_fights","ko_losses")}
def _damage_exposure(row):
    rounds=max(1.0,float(row.rounds_observed))
    return float(np.mean([row.kd_absorbed/rounds,row.head_absorbed/rounds,row.ground_absorbed/rounds,row.opponent_control_seconds/(rounds*60)]))
def _midrank(value,pop):
    if not pop:return None
    a=np.asarray(pop,float);return float(((a<value).sum()+.5*(a==value).sum())/len(a))
def _rating(score,fights):
    if fights<=0:return 50.0
    confidence=1-exp(-fights/3);shrunk=.5+confidence*(_clamp(score,0,1)-.5)
    return float(np.clip(10+80*shrunk,10,90))
def _kd_components(s):
    return (-((s["kd_absorbed"]+.5)/(s["sig_absorbed"]+50)),s["kd_free_fights"]/s["fights"],
        s["kd_free_high_exposure"]/s["kd_high_exposure_fights"] if s["kd_high_exposure_fights"] else None)
def _kd_score(s,peers):
    values=[]
    for value,pop in zip(_kd_components(s),peers):
        if value is not None:
            p=_midrank(value,pop)
            if p is not None:values.append(p)
    return float(np.mean(values)) if values else .5
def _dur_score(s,threshold):
    parts=[]
    if s["dur_high_exposure_fights"]:
        parts.append((.35,s["dur_high_survivals"]/s["dur_high_exposure_fights"]))
        if s["dur_high_exposure_sum"]:parts.append((.30,s["dur_high_survived_exposure_sum"]/s["dur_high_exposure_sum"]))
    if threshold is not None and threshold>0 and s["survived_fights"]:
        parts.append((.20,float(np.clip((s["survived_exposure_sum"]/s["survived_fights"])/threshold,0,2)/2)))
    if s["fights"]:parts.append((.15,1-s["ko_losses"]/s["fights"]))
    return .5 if not parts else sum(w*v for w,v in parts)/sum(w for w,_ in parts)
def _threshold(values,q): return float(np.quantile(values,q)) if len(values)>=100 else None

def _power_rating(raw_state):
    """Translate paired raw power monotonically onto the public 35-90 scale."""
    return float(np.clip(
        BASE_RATING + POWER_RATING_SCALE * float(raw_state),
        POWER_MIN,
        POWER_MAX,
    ))


def build_physical_snapshots(*, rounds=None, master=None, round_path:Path=ROUND_STATS_PATH, master_path:Path=MASTER_PATH) -> PhysicalSnapshots:
    rounds=pd.read_parquet(round_path) if rounds is None else rounds.copy()
    master=pd.read_parquet(master_path) if master is None else master.copy()
    obs=build_physical_observations(rounds,master)
    stamina_ratings=defaultdict(lambda:{"far":50.0,"fpr":50.0});updates=defaultdict(lambda:{"far":0,"fpr":0})
    pools={key:[] for key in STAMINA_POOL_MAP}; weighted=defaultdict(float);quality_sum=defaultdict(float)
    resistance=defaultdict(_new_resistance_state);sig_exposures=[];damage_exposures=[]

    # Internal paired states. Defender state is used only to opponent-adjust
    # the evidence that learns striking power; it is not a published trait.
    power_attack=defaultdict(float)
    power_defense=defaultdict(float)
    power_population_events=0.0
    power_population_sig=0.0

    snapshots=[]
    for date,date_rows in obs.groupby("date",sort=True):
        # Expanding population baseline using only dates already processed.
        power_base_rate=(
            power_population_events + POWER_BASE_PRIOR_RATE * POWER_BASE_PRIOR_SIG
        ) / (
            power_population_sig + POWER_BASE_PRIOR_SIG
        )
        power_base_logit=_logit(power_base_rate)

        kd_threshold=_threshold(sig_exposures,.67);dur_threshold=_threshold(damage_exposures,.70)
        peer=[[],[],[]]
        for s in resistance.values():
            if s["fights"]:
                for i,v in enumerate(_kd_components(s)):
                    if v is not None:peer[i].append(v)
        for row in date_rows.itertuples(index=False):
            s=resistance[row.fighter_id];fights=int(s["fights"])
            snapshots.append({"fight_id":row.fight_id,"fighter_id":row.fighter_id,"fighter_name":row.fighter_name,"date":date,
                "stamina_capacity":100.0,"stamina_depletion_resistance":stamina_ratings[row.fighter_id]["far"],
                "stamina_performance_resilience":stamina_ratings[row.fighter_id]["fpr"],
                "striking_power":_power_rating(power_attack[row.fighter_id]),
                "damage_durability":_rating(_dur_score(s,dur_threshold),fights),
                "knockdown_resistance":_rating(_kd_score(s,peer) if fights else .5,fights)})
        deltas=defaultdict(lambda:{"far":0.0,"fpr":0.0});date_updates=defaultdict(lambda:{"far":0,"fpr":0})
        date_weighted=defaultdict(float);date_quality=defaultdict(float)

        power_attack_deltas=defaultdict(float)
        power_defense_deltas=defaultdict(float)
        date_power_events=0.0
        date_power_sig=0.0

        for _,row in date_rows.iterrows():
            fighter=str(row.fighter_id);far,fpr=_stamina_observations(row,pools);q=stamina_quality(row)
            for key,value in (("far",far),("fpr",fpr)):
                baseline=.5 if quality_sum[key]<=0 else _clamp(weighted[key]/quality_sum[key],0,1)
                expected=_sigmoid(_logit(baseline)+(stamina_ratings[fighter][key]-50)/12)
                if value is not None and q>0:
                    deltas[fighter][key]+=stamina_k(updates[fighter][key])*q*(value-expected);date_updates[fighter][key]+=1
                    date_weighted[key]+=q*value;date_quality[key]+=q
            s=resistance[fighter];kd=max(0,float(row.kd_absorbed));sig=max(0,float(row.sig_absorbed));damage=_damage_exposure(row);ko=float(row.ko_loss)
            s["fights"]+=1;s["kd_absorbed"]+=kd;s["sig_absorbed"]+=sig;s["kd_free_fights"]+=kd<=0
            if kd_threshold is not None and sig>=kd_threshold:s["kd_high_exposure_fights"]+=1;s["kd_free_high_exposure"]+=kd<=0
            if dur_threshold is not None and damage>=dur_threshold:
                s["dur_high_exposure_fights"]+=1;s["dur_high_exposure_sum"]+=damage
                if not ko:s["dur_high_survivals"]+=1;s["dur_high_survived_exposure_sum"]+=damage
            if ko:s["ko_losses"]+=1
            else:s["survived_fights"]+=1;s["survived_exposure_sum"]+=damage
            sig_exposures.append(sig);damage_exposures.append(damage)

            # Paired power update:
            #   observed damaging-event rate = (KD + 0.5 * KO win) / sig landed
            #   expected rate adjusts for attacker power and opponent resistance.
            power_sig=max(0.0,float(row.sig_landed))
            power_kd=max(0.0,float(row.kd_scored))
            power_ko=float(row.ko_win)
            power_events=power_kd + POWER_KO_WEIGHT * power_ko

            date_power_events += power_events
            date_power_sig += power_sig

            if power_sig > 0:
                observed=_clamp(power_events/power_sig,0.0,1.0)
                opponent=str(row.opponent_id)
                expected=_sigmoid(
                    power_base_logit
                    + power_attack[fighter]
                    - power_defense[opponent]
                )
                evidence=1-exp(-power_sig/POWER_EVIDENCE_SATURATION)
                delta=POWER_UPDATE_K * evidence * (observed-expected)
                power_attack_deltas[fighter] += delta
                power_defense_deltas[opponent] -= delta

        # Same-date isolation: apply paired updates only after every fighter on
        # the date has already received the pre-fight snapshot.
        for fighter,delta in power_attack_deltas.items():
            power_attack[fighter] += delta
        for fighter,delta in power_defense_deltas.items():
            power_defense[fighter] += delta

        power_population_events += date_power_events
        power_population_sig += date_power_sig

        for fighter in deltas:
            for key in ("far","fpr"):
                stamina_ratings[fighter][key]=_clamp(stamina_ratings[fighter][key]+deltas[fighter][key],10,90);updates[fighter][key]+=date_updates[fighter][key]
        for key in ("far","fpr"):
            weighted[key]+=date_weighted[key];quality_sum[key]+=date_quality[key]
        for key,column in STAMINA_POOL_MAP.items():
            for value in pd.to_numeric(date_rows[column],errors="coerce").dropna():insort(pools[key],float(value))
    prefight=_validate_physical(pd.DataFrame(snapshots))
    # True state after the final completed fight, using final peer populations.
    final_peer=[[],[],[]]
    for s in resistance.values():
        if s["fights"]:
            for i,v in enumerate(_kd_components(s)):
                if v is not None:final_peer[i].append(v)
    dur_threshold=_threshold(damage_exposures,.70)
    names=obs.sort_values(["date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["fighter_name"]
    latest=[]
    for fighter in sorted(set(obs.fighter_id)):
        s=resistance[fighter];fights=int(s["fights"])
        latest.append({"fighter_id":fighter,"fighter_name":names[fighter],"stamina_capacity":100.0,
            "stamina_depletion_resistance":stamina_ratings[fighter]["far"],"stamina_performance_resilience":stamina_ratings[fighter]["fpr"],
            "striking_power":_power_rating(power_attack[fighter]),"damage_durability":_rating(_dur_score(s,dur_threshold),fights),
            "knockdown_resistance":_rating(_kd_score(s,final_peer),fights)})
    return PhysicalSnapshots(prefight,pd.DataFrame(latest))


def _validate_physical(frame):
    numeric=frame[list(PHYSICAL_COLUMNS)].apply(pd.to_numeric,errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():raise RuntimeError("physical FSR contains missing/non-finite values")
    if not np.allclose(numeric.stamina_capacity,100):raise RuntimeError("stamina_capacity must remain fixed at 100")
    if ((numeric[["stamina_depletion_resistance","stamina_performance_resilience","damage_durability","knockdown_resistance"]]<10)|(numeric[["stamina_depletion_resistance","stamina_performance_resilience","damage_durability","knockdown_resistance"]]>90)).any().any():raise RuntimeError("physical 10-90 rating out of range")
    if ((numeric.striking_power<35)|(numeric.striking_power>90)).any():raise RuntimeError("striking_power outside 35-90")
    return frame

def attach_physical_prefight(core_prefight,physical_prefight):
    keys=["fight_id","fighter_id"]
    if core_prefight.duplicated(keys).any() or physical_prefight.duplicated(keys).any():raise ValueError("prefight snapshot violates fighter-fight grain")
    core_keys=set(map(tuple,core_prefight[keys].astype(str).to_numpy()));physical_keys=set(map(tuple,physical_prefight[keys].astype(str).to_numpy()))
    if core_keys!=physical_keys:raise RuntimeError(f"core/physical prefight key mismatch: missing={len(core_keys-physical_keys)}, extra={len(physical_keys-core_keys)}")
    return core_prefight.merge(physical_prefight[[*keys,*PHYSICAL_COLUMNS]],on=keys,how="left",validate="one_to_one")
def attach_physical_latest(core_latest,physical_latest):
    if core_latest.fighter_id.duplicated().any() or physical_latest.fighter_id.duplicated().any():raise ValueError("latest snapshot contains duplicate fighter_id")
    core=set(core_latest.fighter_id.astype(str));physical=set(physical_latest.fighter_id.astype(str))
    if core!=physical:raise RuntimeError(f"core/physical latest fighter mismatch: missing={len(core-physical)}, extra={len(physical-core)}")
    return core_latest.merge(physical_latest[["fighter_id",*PHYSICAL_COLUMNS]],on="fighter_id",how="left",validate="one_to_one")
