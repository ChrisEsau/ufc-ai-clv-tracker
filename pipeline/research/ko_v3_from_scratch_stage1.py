"""From-scratch KO/KD Stage 1: raw-data predictive validation.

Research only. This module deliberately excludes all FSR traits and all Monte
Carlo mechanics. It asks what raw, leakage-safe UFC history predicts three
separate future targets:

1. knockdown creation/susceptibility per landed significant strike;
2. KO/TKO conversion in fighter-fights where the attacker records a KD;
3. KO/TKO wins with no recorded attacker KD (a fight-level direct-finish proxy).

Every prefight state is same-date delayed: all fighters on an event date are
snapshotted before any result from that date is incorporated. Models are scored
with expanding calendar-year out-of-sample folds.

Important limitation: UFC round aggregates do not timestamp individual KDs or
finish sequences. Stage 1 therefore does NOT estimate acute-hurt decay or claim
that a zero-KD KO/TKO is literally a single-strike direct KO.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH

DEFAULT_OUT = Path("data/research/ko_v3_from_scratch_stage1")
EWM_DECAYS = (0.25, 0.50, 0.75, 0.90, 0.95)

RAW_ROUND_COLUMNS = [
    "event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "round",
    "kd", "sig_str_landed", "sig_str_attempted", "head_landed", "head_attempted",
    "distance_landed", "distance_attempted", "clinch_landed", "clinch_attempted",
    "ground_landed", "ground_attempted",
]

BASE_STATE_FIELDS = (
    "fights", "kd_scored", "sig_landed", "kd_absorbed", "sig_absorbed",
    "ko_wins", "ko_losses", "post_kd_opportunities", "post_kd_finishes",
    "recovery_opportunities", "recoveries", "direct_ko_wins", "direct_ko_losses",
)


def _tag(decay: float) -> str:
    return f"ewm{int(round(decay * 100)):02d}"


def _safe_rate(num, den):
    a = np.asarray(num, dtype=float)
    b = np.asarray(den, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan, dtype=float), where=b > 0)


def _normalize_date(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if out.isna().any():
        raise ValueError(f"invalid dates: {int(out.isna().sum())}")
    return out.dt.normalize()


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    for col in candidates:
        if col in colset:
            return col
    return None


def _corner_age(master: pd.DataFrame, corner: str, fight_date: pd.Series) -> pd.Series:
    direct = _first_existing(master.columns, (f"{corner}_age", f"{corner}_fighter_age", f"{corner}_age_years"))
    if direct:
        return pd.to_numeric(master[direct], errors="coerce")
    dob = _first_existing(master.columns, (
        f"{corner}_dob", f"{corner}_date_of_birth",
        f"{corner}_fighter_dob", f"{corner}_fighter_date_of_birth",
    ))
    if dob:
        birth = pd.to_datetime(master[dob], errors="coerce")
        return (fight_date - birth).dt.days / 365.25
    return pd.Series(np.nan, index=master.index, dtype=float)


def _ko_tko(method: pd.Series) -> pd.Series:
    text = method.fillna("").astype(str).str.upper()
    return text.str.contains(r"KO/TKO|\bTKO\b|\bKO\b", regex=True)


def load_raw_fighter_fights(round_path: Path = ROUND_STATS_PATH, master_path: Path = MASTER_PATH) -> tuple[pd.DataFrame, dict]:
    """Return one raw observation row per fighter-fight, with no FSR transforms."""
    rounds = pd.read_parquet(round_path, columns=RAW_ROUND_COLUMNS).copy()
    master = pd.read_parquet(master_path).drop_duplicates("fight_id").copy()

    rounds["event_date"] = _normalize_date(rounds["event_date"])
    for c in ("fight_id", "fighter_id", "opponent_id"):
        rounds[c] = rounds[c].astype(str)
    for c in RAW_ROUND_COLUMNS:
        if c not in {"event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id"}:
            rounds[c] = pd.to_numeric(rounds[c], errors="coerce").fillna(0.0)
    if rounds.duplicated(["fight_id", "round", "fighter_id"]).any():
        raise ValueError("raw round source contains duplicate fighter-round rows")

    agg = rounds.groupby(["event_date", "fight_id", "fighter_id"], as_index=False).agg(
        fighter_name=("fighter_name", "first"), opponent_id=("opponent_id", "first"),
        kd_scored=("kd", "sum"), sig_landed=("sig_str_landed", "sum"),
        sig_attempted=("sig_str_attempted", "sum"), head_landed=("head_landed", "sum"),
        head_attempted=("head_attempted", "sum"), distance_landed=("distance_landed", "sum"),
        distance_attempted=("distance_attempted", "sum"), clinch_landed=("clinch_landed", "sum"),
        clinch_attempted=("clinch_attempted", "sum"), ground_landed=("ground_landed", "sum"),
        ground_attempted=("ground_attempted", "sum"),
    )
    opp = agg[["fight_id", "fighter_id", "kd_scored", "sig_landed"]].rename(columns={
        "fighter_id": "opponent_id", "kd_scored": "kd_absorbed", "sig_landed": "sig_absorbed",
    })
    agg = agg.merge(opp, on=["fight_id", "opponent_id"], how="left", validate="one_to_one")
    if agg[["kd_absorbed", "sig_absorbed"]].isna().any().any():
        raise ValueError("raw round source contains non-reciprocal fighter rows")

    master["fight_id"] = master["fight_id"].astype(str)
    fight_date = _normalize_date(master["date"])
    master["fight_date"] = fight_date
    master["r_age_stage1"] = _corner_age(master, "r", fight_date)
    master["b_age_stage1"] = _corner_age(master, "b", fight_date)
    for c in ("r_id", "b_id", "winner_id"):
        if c in master:
            master[c] = master[c].astype(str)
    needed = [
        "fight_id", "fight_date", "division", "method", "winner_id", "r_id", "b_id",
        "r_name", "b_name", "r_age_stage1", "b_age_stage1",
    ]
    missing = [c for c in needed if c not in master.columns]
    if missing:
        raise ValueError(f"master missing required Stage-1 columns: {missing}")
    agg = agg.merge(master[needed], on="fight_id", how="left", validate="many_to_one")
    if agg["fight_date"].isna().any():
        raise ValueError("master metadata missing for raw round fights")

    date_mismatch = int((agg["event_date"] != agg["fight_date"]).sum())
    agg["event_date"] = agg["fight_date"]
    is_red = agg["fighter_id"].eq(agg["r_id"])
    is_blue = agg["fighter_id"].eq(agg["b_id"])
    if not (is_red | is_blue).all():
        raise ValueError("fighter IDs in round stats do not map to master corners")
    agg["fighter_age"] = np.where(is_red, agg["r_age_stage1"], agg["b_age_stage1"])
    agg["opponent_age"] = np.where(is_red, agg["b_age_stage1"], agg["r_age_stage1"])
    agg["won"] = agg["fighter_id"].eq(agg["winner_id"])
    agg["is_ko_tko"] = _ko_tko(agg["method"])
    agg["ko_win"] = agg["won"] & agg["is_ko_tko"]
    agg["ko_loss"] = (~agg["won"]) & agg["is_ko_tko"]
    agg["had_kd"] = agg["kd_scored"].gt(0)
    agg["was_dropped"] = agg["kd_absorbed"].gt(0)
    agg["post_kd_opportunity"] = agg["had_kd"].astype(float)
    agg["post_kd_finish"] = (agg["had_kd"] & agg["ko_win"]).astype(float)
    agg["recovery_opportunity"] = agg["was_dropped"].astype(float)
    agg["recovered_after_kd"] = (agg["was_dropped"] & ~agg["ko_loss"]).astype(float)
    agg["direct_ko_win"] = (agg["ko_win"] & ~agg["had_kd"]).astype(float)
    agg["direct_ko_loss"] = (agg["ko_loss"] & ~agg["was_dropped"]).astype(float)

    invalid_kd = int((agg["kd_scored"] > agg["sig_landed"]).sum())
    agg["valid_kd_trial"] = (agg["sig_landed"] > 0) & (agg["kd_scored"] <= agg["sig_landed"])
    audit = {
        "fighter_fight_rows": int(len(agg)), "fights": int(agg["fight_id"].nunique()),
        "date_mismatch_rows": date_mismatch, "kd_gt_sig_landed_rows": invalid_kd,
        "valid_kd_trial_rows": int(agg["valid_kd_trial"].sum()),
        "age_coverage": float(pd.Series(agg["fighter_age"]).notna().mean()),
        "ko_tko_fighter_wins": int(agg["ko_win"].sum()),
        "post_kd_opportunities": int(agg["post_kd_opportunity"].sum()),
        "post_kd_finishes": int(agg["post_kd_finish"].sum()),
        "direct_ko_proxy_wins": int(agg["direct_ko_win"].sum()),
    }
    return agg.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True), audit


def _empty_state() -> dict[str, float]:
    return {field: 0.0 for field in BASE_STATE_FIELDS}


def _observation_values(row) -> dict[str, float]:
    return {
        "fights": 1.0, "kd_scored": float(row.kd_scored), "sig_landed": float(row.sig_landed),
        "kd_absorbed": float(row.kd_absorbed), "sig_absorbed": float(row.sig_absorbed),
        "ko_wins": float(row.ko_win), "ko_losses": float(row.ko_loss),
        "post_kd_opportunities": float(row.post_kd_opportunity), "post_kd_finishes": float(row.post_kd_finish),
        "recovery_opportunities": float(row.recovery_opportunity), "recoveries": float(row.recovered_after_kd),
        "direct_ko_wins": float(row.direct_ko_win), "direct_ko_losses": float(row.direct_ko_loss),
    }


def build_prefight_states(ff: pd.DataFrame) -> pd.DataFrame:
    """Attach cumulative and EWM raw histories with same-date delayed updates."""
    cumulative: dict[str, dict[str, float]] = defaultdict(_empty_state)
    ewms: dict[float, dict[str, dict[str, float]]] = {d: defaultdict(_empty_state) for d in EWM_DECAYS}
    out: list[dict] = []
    for _, batch in ff.groupby("event_date", sort=True):
        for row in batch.itertuples(index=False):
            fid = str(row.fighter_id)
            rec = row._asdict()
            for field in BASE_STATE_FIELDS:
                rec[f"prior_{field}"] = cumulative[fid][field]
            for decay in EWM_DECAYS:
                tag = _tag(decay)
                for field in BASE_STATE_FIELDS:
                    rec[f"{tag}_{field}"] = ewms[decay][fid][field]
            out.append(rec)
        for row in batch.itertuples(index=False):
            fid = str(row.fighter_id)
            obs = _observation_values(row)
            for field, value in obs.items():
                cumulative[fid][field] += value
            for decay in EWM_DECAYS:
                for field, value in obs.items():
                    ewms[decay][fid][field] = decay * ewms[decay][fid][field] + value
    return pd.DataFrame(out).sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _feature_family(frame: pd.DataFrame, prefix: str, source_prefix: str) -> None:
    p = source_prefix
    frame[f"{prefix}_att_kd_rate"] = _safe_rate(frame[f"{p}kd_scored"], frame[f"{p}sig_landed"])
    frame[f"{prefix}_def_kd_suscept"] = _safe_rate(frame[f"opp_{p}kd_absorbed"], frame[f"opp_{p}sig_absorbed"])
    frame[f"{prefix}_att_ko_win_rate"] = _safe_rate(frame[f"{p}ko_wins"], frame[f"{p}fights"])
    frame[f"{prefix}_def_ko_loss_rate"] = _safe_rate(frame[f"opp_{p}ko_losses"], frame[f"opp_{p}fights"])
    frame[f"{prefix}_att_post_kd_conversion"] = _safe_rate(frame[f"{p}post_kd_finishes"], frame[f"{p}post_kd_opportunities"])
    frame[f"{prefix}_def_post_kd_recovery"] = _safe_rate(frame[f"opp_{p}recoveries"], frame[f"opp_{p}recovery_opportunities"])
    frame[f"{prefix}_att_direct_ko_rate"] = _safe_rate(frame[f"{p}direct_ko_wins"], frame[f"{p}fights"])
    frame[f"{prefix}_def_direct_ko_loss_rate"] = _safe_rate(frame[f"opp_{p}direct_ko_losses"], frame[f"opp_{p}fights"])
    frame[f"{prefix}_att_log_sig_landed"] = np.log1p(frame[f"{p}sig_landed"].astype(float))
    frame[f"{prefix}_def_log_sig_absorbed"] = np.log1p(frame[f"opp_{p}sig_absorbed"].astype(float))
    frame[f"{prefix}_att_log_fights"] = np.log1p(frame[f"{p}fights"].astype(float))
    frame[f"{prefix}_def_log_fights"] = np.log1p(frame[f"opp_{p}fights"].astype(float))


def build_matchup_frame(states: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_date", "fight_id", "fighter_id"]
    history_cols = [c for c in states.columns if c.startswith("prior_") or c.startswith("ewm")]
    opp = states[keys + history_cols].rename(columns={"fighter_id": "opponent_id", **{c: f"opp_{c}" for c in history_cols}})
    x = states.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")
    _feature_family(x, "cum", "prior_")
    for decay in EWM_DECAYS:
        tag = _tag(decay)
        _feature_family(x, tag, f"{tag}_")
    x["attacker_age"] = pd.to_numeric(x["fighter_age"], errors="coerce")
    x["defender_age"] = pd.to_numeric(x["opponent_age"], errors="coerce")
    x["division_cat"] = x["division"].fillna("unknown").astype(str)
    x["test_year"] = pd.to_datetime(x["event_date"]).dt.year.astype(int)
    return x.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class Arm:
    name: str
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()


class NumericCategoricalEncoder:
    def __init__(self, numeric: Iterable[str], categorical: Iterable[str]):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.medians, self.means, self.sds, self.categories = {}, {}, {}, {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame):
        names = []
        for c in self.numeric:
            s = pd.to_numeric(frame[c], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(med).astype(float)
            mean, sd = float(filled.mean()), float(filled.std(ddof=0))
            if not np.isfinite(sd) or sd < 1e-9:
                sd = 1.0
            self.medians[c], self.means[c], self.sds[c] = med, mean, sd
            names.extend([c, f"{c}__missing"])
        for c in self.categorical:
            cats = sorted(frame[c].fillna("unknown").astype(str).unique().tolist())
            self.categories[c] = cats
            names.extend([f"{c}={v}" for v in cats])
        self.feature_names = names
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts = []
        for c in self.numeric:
            s = pd.to_numeric(frame[c], errors="coerce")
            parts.append(((s.fillna(self.medians[c]).astype(float) - self.means[c]) / self.sds[c]).to_numpy()[:, None])
            parts.append(s.isna().astype(float).to_numpy()[:, None])
        for c in self.categorical:
            vals = frame[c].fillna("unknown").astype(str).to_numpy()
            for cat in self.categories[c]:
                parts.append((vals == cat).astype(float)[:, None])
        return np.hstack(parts) if parts else np.zeros((len(frame), 0), dtype=float)


def _fit_logit(train: pd.DataFrame, arm: Arm, y: np.ndarray, weights: np.ndarray | None = None):
    enc = NumericCategoricalEncoder(arm.numeric, arm.categorical).fit(train)
    X = enc.transform(train)
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    model.fit(X, y, sample_weight=weights)
    return enc, model


def _coef_rows(year: int, arm: Arm, enc, model) -> list[dict]:
    rows = [{"test_year": year, "arm": arm.name, "feature": "intercept", "coefficient": float(model.intercept_[0])}]
    rows += [{"test_year": year, "arm": arm.name, "feature": n, "coefficient": float(v)} for n, v in zip(enc.feature_names, model.coef_[0], strict=True)]
    return rows


def _kd_weighted_training_rows(frame: pd.DataFrame):
    pieces, ys, ws = [], [], []
    for idx, row in frame.iterrows():
        k, n = float(row["kd_scored"]), float(row["sig_landed"])
        if not (n > 0 and 0 <= k <= n):
            continue
        if k > 0:
            pieces.append(idx); ys.append(1); ws.append(k)
        if n-k > 0:
            pieces.append(idx); ys.append(0); ws.append(n-k)
    return frame.loc[pieces].reset_index(drop=True), np.asarray(ys, int), np.asarray(ws, float)


def _kd_arms() -> list[Arm]:
    arms = [
        Arm("division", (), ("division_cat",)),
        Arm("cum_attacker", ("cum_att_kd_rate", "cum_att_log_sig_landed")),
        Arm("cum_defender", ("cum_def_kd_suscept", "cum_def_log_sig_absorbed")),
        Arm("cum_both", ("cum_att_kd_rate", "cum_def_kd_suscept", "cum_att_log_sig_landed", "cum_def_log_sig_absorbed")),
        Arm("cum_both_age", ("cum_att_kd_rate", "cum_def_kd_suscept", "cum_att_log_sig_landed", "cum_def_log_sig_absorbed", "attacker_age", "defender_age")),
        Arm("cum_both_age_division", ("cum_att_kd_rate", "cum_def_kd_suscept", "cum_att_log_sig_landed", "cum_def_log_sig_absorbed", "attacker_age", "defender_age"), ("division_cat",)),
    ]
    for d in EWM_DECAYS:
        t = _tag(d)
        arms.append(Arm(f"{t}_both_age_division", (f"{t}_att_kd_rate", f"{t}_def_kd_suscept", f"{t}_att_log_sig_landed", f"{t}_def_log_sig_absorbed", "attacker_age", "defender_age"), ("division_cat",)))
    return arms


def _conversion_arms() -> list[Arm]:
    arms = [
        Arm("division", (), ("division_cat",)),
        Arm("cum_conversion_recovery", ("cum_att_post_kd_conversion", "cum_def_post_kd_recovery", "cum_att_log_fights", "cum_def_log_fights")),
        Arm("cum_conversion_recovery_age_division", ("cum_att_post_kd_conversion", "cum_def_post_kd_recovery", "cum_att_log_fights", "cum_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)),
    ]
    for d in EWM_DECAYS:
        t = _tag(d)
        arms.append(Arm(f"{t}_conversion_recovery_age_division", (f"{t}_att_post_kd_conversion", f"{t}_def_post_kd_recovery", f"{t}_att_log_fights", f"{t}_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)))
    return arms


def _direct_arms() -> list[Arm]:
    arms = [
        Arm("division", (), ("division_cat",)),
        Arm("cum_direct", ("cum_att_direct_ko_rate", "cum_def_direct_ko_loss_rate", "cum_att_log_fights", "cum_def_log_fights")),
        Arm("cum_direct_age_division", ("cum_att_direct_ko_rate", "cum_def_direct_ko_loss_rate", "cum_att_log_fights", "cum_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)),
    ]
    for d in EWM_DECAYS:
        t = _tag(d)
        arms.append(Arm(f"{t}_direct_age_division", (f"{t}_att_direct_ko_rate", f"{t}_def_direct_ko_loss_rate", f"{t}_att_log_fights", f"{t}_def_log_fights", "attacker_age", "defender_age"), ("division_cat",)))
    return arms


def _kd_fold_metrics(test, p, arm, year):
    k, n = test["kd_scored"].to_numpy(float), test["sig_landed"].to_numpy(float)
    p = np.clip(np.asarray(p, float), 1e-8, 1-1e-8); total_n = float(n.sum())
    any_y = (k > 0).astype(int); any_p = 1-np.power(1-p, n)
    return {
        "test_year": year, "arm": arm, "n_fighter_fights": int(len(test)), "landed_sig_strikes": total_n,
        "knockdowns": float(k.sum()), "actual_kd_per_landed": float(k.sum()/max(total_n,1.0)),
        "predicted_kd_per_landed": float(np.sum(n*p)/max(total_n,1.0)),
        "strike_log_loss": float(-np.sum(k*np.log(p)+(n-k)*np.log1p(-p))/max(total_n,1.0)),
        "strike_brier": float(np.sum(k*(1-p)**2+(n-k)*p**2)/max(total_n,1.0)),
        "any_kd_auc": float(roc_auc_score(any_y, any_p)) if np.unique(any_y).size == 2 else np.nan,
        "any_kd_brier": float(brier_score_loss(any_y, any_p)),
    }


def run_kd_walkforward(frame, first_test_year):
    usable = frame[frame["valid_kd_trial"]].copy()
    years = sorted(y for y in usable["test_year"].unique() if y >= first_test_year)
    details, metrics, coefs = [], [], []
    for year in years:
        train = usable[usable["event_date"] < pd.Timestamp(f"{year}-01-01")].copy()
        test = usable[usable["test_year"].eq(year)].copy()
        if len(train) < 500 or len(test) < 20 or train["sig_landed"].sum() <= 0:
            continue
        pop_p = float(train["kd_scored"].sum()/train["sig_landed"].sum())
        p = np.full(len(test), np.clip(pop_p,1e-8,1-1e-8))
        metrics.append(_kd_fold_metrics(test,p,"population",year))
        d = test[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division","kd_scored","sig_landed"]].copy()
        d["test_year"], d["arm"], d["p_kd_per_landed"] = year, "population", p; details.append(d)
        train_rows, y, w = _kd_weighted_training_rows(train)
        if np.unique(y).size < 2: continue
        for arm in _kd_arms():
            try:
                enc, model = _fit_logit(train_rows, arm, y, w); p = model.predict_proba(enc.transform(test))[:,1]
            except Exception as exc:
                print(f"KD ARM SKIP year={year} arm={arm.name}: {exc}"); continue
            metrics.append(_kd_fold_metrics(test,p,arm.name,year)); coefs.extend(_coef_rows(year,arm,enc,model))
            d = test[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division","kd_scored","sig_landed"]].copy()
            d["test_year"], d["arm"], d["p_kd_per_landed"] = year, arm.name, p; details.append(d)
    return (pd.concat(details,ignore_index=True) if details else pd.DataFrame(), pd.DataFrame(metrics), pd.DataFrame(coefs))


def _binary_fold_metrics(y,p,arm,year,n_train):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8); y=np.asarray(y,int)
    return {"test_year":year,"arm":arm,"n_train":int(n_train),"n_test":int(len(y)),"actual_rate":float(y.mean()),"predicted_rate":float(p.mean()),"auc":float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1]))}


def run_binary_walkforward(frame, *, first_test_year, eligible_mask, target, arms, label):
    usable=frame.loc[eligible_mask].copy(); usable[target]=usable[target].astype(int)
    years=sorted(y for y in usable["test_year"].unique() if y>=first_test_year)
    details,metrics,coefs=[],[],[]
    for year in years:
        train=usable[usable["event_date"]<pd.Timestamp(f"{year}-01-01")].copy(); test=usable[usable["test_year"].eq(year)].copy()
        if len(train)<200 or len(test)<10 or train[target].nunique()<2: continue
        ytr=train[target].to_numpy(int); yte=test[target].to_numpy(int); pop=float(ytr.mean())
        p=np.full(len(test),np.clip(pop,1e-8,1-1e-8)); metrics.append(_binary_fold_metrics(yte,p,"population",year,len(train)))
        d=test[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division",target]].copy(); d["test_year"],d["arm"],d[f"p_{label}"]=year,"population",p; details.append(d)
        for arm in arms:
            try:
                enc,model=_fit_logit(train,arm,ytr); p=model.predict_proba(enc.transform(test))[:,1]
            except Exception as exc:
                print(f"{label.upper()} ARM SKIP year={year} arm={arm.name}: {exc}"); continue
            metrics.append(_binary_fold_metrics(yte,p,arm.name,year,len(train))); coefs.extend(_coef_rows(year,arm,enc,model))
            d=test[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division",target]].copy(); d["test_year"],d["arm"],d[f"p_{label}"]=year,arm.name,p; details.append(d)
    return (pd.concat(details,ignore_index=True) if details else pd.DataFrame(),pd.DataFrame(metrics),pd.DataFrame(coefs))


def _aggregate_kd(detail):
    rows=[]
    if detail.empty: return pd.DataFrame()
    for arm,g in detail.groupby("arm",sort=False):
        p=np.clip(g["p_kd_per_landed"].to_numpy(float),1e-8,1-1e-8); k=g["kd_scored"].to_numpy(float); n=g["sig_landed"].to_numpy(float); total=float(n.sum())
        any_y=(k>0).astype(int); any_p=1-np.power(1-p,n)
        rows.append({"arm":arm,"n_fighter_fights":int(len(g)),"landed_sig_strikes":total,"knockdowns":float(k.sum()),"actual_kd_per_landed":float(k.sum()/max(total,1.0)),"predicted_kd_per_landed":float(np.sum(n*p)/max(total,1.0)),"strike_log_loss":float(-np.sum(k*np.log(p)+(n-k)*np.log1p(-p))/max(total,1.0)),"strike_brier":float(np.sum(k*(1-p)**2+(n-k)*p**2)/max(total,1.0)),"any_kd_auc":float(roc_auc_score(any_y,any_p)) if np.unique(any_y).size==2 else np.nan,"any_kd_brier":float(brier_score_loss(any_y,any_p))})
    return pd.DataFrame(rows).sort_values(["strike_log_loss","strike_brier"]).reset_index(drop=True)


def _aggregate_binary(detail,target,prob_col):
    rows=[]
    if detail.empty:return pd.DataFrame()
    for arm,g in detail.groupby("arm",sort=False):
        y=g[target].to_numpy(int); p=np.clip(g[prob_col].to_numpy(float),1e-8,1-1e-8)
        rows.append({"arm":arm,"n":int(len(g)),"actual_rate":float(y.mean()),"predicted_rate":float(p.mean()),"auc":float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1]))})
    return pd.DataFrame(rows).sort_values(["log_loss","brier"]).reset_index(drop=True)


def _kd_calibration(detail):
    rows=[]
    if detail.empty:return pd.DataFrame()
    for arm,g in detail.groupby("arm",sort=False):
        x=g.copy()
        try:x["bucket"]=pd.qcut(x["p_kd_per_landed"],q=10,duplicates="drop")
        except ValueError:continue
        for bucket,z in x.groupby("bucket",observed=True):
            n=z["sig_landed"].sum()
            if n<=0:continue
            rows.append({"arm":arm,"bucket":str(bucket),"fighter_fights":int(len(z)),"landed_sig_strikes":float(n),"predicted_kd_per_landed":float(np.average(z["p_kd_per_landed"],weights=z["sig_landed"])),"actual_kd_per_landed":float(z["kd_scored"].sum()/n)})
    return pd.DataFrame(rows)


def _best(summary,metric,maximize=False):
    if summary.empty or metric not in summary:return None
    z=summary.dropna(subset=[metric])
    if z.empty:return None
    row=z.loc[z[metric].idxmax() if maximize else z[metric].idxmin()]
    return {"arm":str(row["arm"]),metric:float(row[metric])}


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--round-path",type=Path,default=ROUND_STATS_PATH); ap.add_argument("--master-path",type=Path,default=MASTER_PATH); ap.add_argument("--out-dir",type=Path,default=DEFAULT_OUT); ap.add_argument("--first-test-year",type=int,default=2020); return ap.parse_args()


def main():
    args=parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    ff,audit=load_raw_fighter_fights(args.round_path,args.master_path); states=build_prefight_states(ff); frame=build_matchup_frame(states)
    kd_detail,kd_year,kd_coef=run_kd_walkforward(frame,args.first_test_year); kd_summary=_aggregate_kd(kd_detail); kd_calibration=_kd_calibration(kd_detail)
    post_detail,post_year,post_coef=run_binary_walkforward(frame,first_test_year=args.first_test_year,eligible_mask=frame["post_kd_opportunity"].gt(0),target="post_kd_finish",arms=_conversion_arms(),label="post_kd_finish"); post_summary=_aggregate_binary(post_detail,"post_kd_finish","p_post_kd_finish")
    direct_detail,direct_year,direct_coef=run_binary_walkforward(frame,first_test_year=args.first_test_year,eligible_mask=pd.Series(True,index=frame.index),target="direct_ko_win",arms=_direct_arms(),label="direct_ko"); direct_summary=_aggregate_binary(direct_detail,"direct_ko_win","p_direct_ko")
    ff.to_parquet(args.out_dir/"raw_fighter_fights.parquet",index=False); frame.to_parquet(args.out_dir/"prefight_matchup_states.parquet",index=False)
    outputs={"kd_oos_predictions.csv":kd_detail,"kd_oos_by_year.csv":kd_year,"kd_oos_summary.csv":kd_summary,"kd_calibration.csv":kd_calibration,"kd_coefficients_by_year.csv":kd_coef,"post_kd_oos_predictions.csv":post_detail,"post_kd_oos_by_year.csv":post_year,"post_kd_oos_summary.csv":post_summary,"post_kd_coefficients_by_year.csv":post_coef,"direct_ko_oos_predictions.csv":direct_detail,"direct_ko_oos_by_year.csv":direct_year,"direct_ko_oos_summary.csv":direct_summary,"direct_ko_coefficients_by_year.csv":direct_coef}
    for name,data in outputs.items():data.to_csv(args.out_dir/name,index=False)
    report={"stage":"KO V3 from scratch — Stage 1 raw predictive validation","uses_fsr_traits":False,"changes_mc_mechanics":False,"same_date_delayed":True,"first_test_year":int(args.first_test_year),"ewm_decays_screened":list(EWM_DECAYS),"raw_data_audit":audit,"best_kd_strike_log_loss":_best(kd_summary,"strike_log_loss"),"best_kd_any_event_auc":_best(kd_summary,"any_kd_auc",maximize=True),"best_post_kd_log_loss":_best(post_summary,"log_loss"),"best_post_kd_auc":_best(post_summary,"auc",maximize=True),"best_direct_ko_log_loss":_best(direct_summary,"log_loss"),"best_direct_ko_auc":_best(direct_summary,"auc",maximize=True),"limitations":["UFC round stats are aggregate, not timestamped strike events.","Post-KD conversion is measured at fighter-fight level because exact KD-to-finish timing is unavailable.","Direct KO means KO/TKO win with zero recorded attacker KDs; it is not an event-timestamped direct-strike label.","Hurt-state decay cannot be estimated from this aggregate dataset and is deliberately not invented in Stage 1.","Existing FSR power, durability, and KD resistance are excluded from Stage 1."]}
    (args.out_dir/"stage1_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KO V3 FROM SCRATCH — STAGE 1"); print(json.dumps(report,indent=2,sort_keys=True)); print("\nKD OOS SUMMARY"); print(kd_summary.to_string(index=False,float_format=lambda x:f"{x:.6f}")); print("\nPOST-KD OOS SUMMARY"); print(post_summary.to_string(index=False,float_format=lambda x:f"{x:.6f}")); print("\nDIRECT-KO PROXY OOS SUMMARY"); print(direct_summary.to_string(index=False,float_format=lambda x:f"{x:.6f}")); print(f"\nOUTPUT: {args.out_dir}")


if __name__=="__main__":main()
