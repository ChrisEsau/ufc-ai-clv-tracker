"""Research-only grouped-strike calibration for a better Event2 KO hazard.

The current Event2 KO resolver samples one logistic KO probability after every
landed modeled strike. UFCStats does not provide event timestamps for individual
landed strikes, so this study does NOT invent strike-level labels. Instead it
uses fighter-round significant-strike landings as grouped exposure and fits the
exact likelihood implied by an independent per-landed-strike logistic hazard:

    p_strike = sigmoid(X beta)
    P(KO in round | n landed sig. strikes) = 1 - (1 - p_strike) ** n

A round is positive only when that fighter won by KO/TKO in that round. Current-
round KDs are deliberately NOT predictors because their ordering relative to a
finish is unavailable in aggregate round stats. Prior KDs scored in completed
rounds are leakage-safe and are used as the available dynamic damage state.

This script does not modify production mechanics or FSR artifacts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    legacy_kdres_equivalent,
    legacy_power_equivalent,
)

DEFAULT_OUT = Path("data/research/event2_ko_hazard_v2")
SCHNELL_COSTA_FIGHT_ID = "5d2eedd05081ed23"

CURRENT = {
    "base_probability": 0.00250,
    "attacker_power_centered": 0.0200,
    "attacker_age_centered": -0.0300,
    "defender_age_centered": 0.0300,
    "prior_defender_kds": 1.00,
    "elapsed_minutes": 0.0,
}

MODEL_FEATURES = {
    "A_refit_current_inputs": [
        "attacker_power_centered",
        "attacker_age_centered",
        "defender_age_centered",
        "prior_defender_kds",
    ],
    "B_plus_kdres": [
        "attacker_power_centered",
        "attacker_age_centered",
        "defender_age_centered",
        "prior_defender_kds",
        "defender_kdres_centered",
    ],
    "C_plus_durability": [
        "attacker_power_centered",
        "attacker_age_centered",
        "defender_age_centered",
        "prior_defender_kds",
        "defender_durability_centered",
    ],
    "D_plus_both": [
        "attacker_power_centered",
        "attacker_age_centered",
        "defender_age_centered",
        "prior_defender_kds",
        "defender_kdres_centered",
        "defender_durability_centered",
    ],
    "E_plus_both_and_elapsed": [
        "attacker_power_centered",
        "attacker_age_centered",
        "defender_age_centered",
        "prior_defender_kds",
        "defender_kdres_centered",
        "defender_durability_centered",
        "elapsed_minutes",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master-path", default=str(MASTER_PATH))
    p.add_argument("--round-stats-path", default=str(ROUND_STATS_PATH))
    p.add_argument("--prefight-path", default=str(FSR_V3_PREFIGHT_SNAPSHOTS_PATH))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--bootstrap", type=int, default=500)
    return p.parse_args()


def _parse_dates(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    miss = x.isna()
    if miss.any():
        x.loc[miss] = pd.to_datetime(s.loc[miss], errors="coerce")
    return x


def _is_ko(method: object) -> bool:
    if pd.isna(method):
        return False
    s = str(method).strip().lower()
    if "doctor" in s:
        return False
    return "ko/tko" in s or s == "ko" or s.startswith("tko")


def _age(dob: object, date: pd.Timestamp) -> float:
    d = pd.to_datetime(dob, errors="coerce")
    if pd.isna(d) or pd.isna(date):
        return np.nan
    return float((date - d).days / 365.2425)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _require(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def build_rows(master: pd.DataFrame, rounds: pd.DataFrame, prefight: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _require(master, ["fight_id", "date", "method", "finish_round", "r_id", "b_id", "winner_id"], "master")
    _require(rounds, ["fight_id", "fighter_id", "opponent_id", "round", "sig_str_landed", "kd"], "round stats")
    _require(prefight, ["fight_id", "fighter_id", "striking_power_v3", "knockdown_resistance_v3", "damage_durability"], "prefight")

    m = master.copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["fight_date"] = _parse_dates(m["date"])
    m["finish_round"] = pd.to_numeric(m["finish_round"], errors="coerce")
    m = m.loc[m.fight_date.notna() & m.finish_round.notna()].drop_duplicates("fight_id", keep="last")
    midx = m.set_index("fight_id", drop=False)

    p = prefight.copy()
    p["fight_id"] = p["fight_id"].astype(str)
    p["fighter_id"] = p["fighter_id"].astype(str)
    pidx = p.set_index(["fight_id", "fighter_id"], drop=False)

    r = rounds.copy()
    r["fight_id"] = r["fight_id"].astype(str)
    r["fighter_id"] = r["fighter_id"].astype(str)
    r["opponent_id"] = r["opponent_id"].astype(str)
    r["round"] = pd.to_numeric(r["round"], errors="coerce")
    r["sig_str_landed"] = pd.to_numeric(r["sig_str_landed"], errors="coerce")
    r["kd"] = pd.to_numeric(r["kd"], errors="coerce")
    r = r.dropna(subset=["round", "sig_str_landed", "kd"])
    r = r.sort_values(["fight_id", "fighter_id", "round"]).copy()
    r["prior_defender_kds"] = r.groupby(["fight_id", "fighter_id"], sort=False)["kd"].cumsum() - r["kd"]

    rows: list[dict] = []
    skipped = {"master_missing": 0, "prefight_missing": 0, "bad_ids": 0}
    for rr in r.itertuples(index=False):
        fid, aid, did = str(rr.fight_id), str(rr.fighter_id), str(rr.opponent_id)
        if fid not in midx.index:
            skipped["master_missing"] += 1
            continue
        if (fid, aid) not in pidx.index or (fid, did) not in pidx.index:
            skipped["prefight_missing"] += 1
            continue
        mm = midx.loc[fid]
        if isinstance(mm, pd.DataFrame):
            mm = mm.iloc[-1]
        aa = pidx.loc[(fid, aid)]
        dd = pidx.loc[(fid, did)]
        if isinstance(aa, pd.DataFrame): aa = aa.iloc[-1]
        if isinstance(dd, pd.DataFrame): dd = dd.iloc[-1]

        red_id, blue_id = str(mm.r_id), str(mm.b_id)
        if aid == red_id:
            adob, ddob = getattr(mm, "r_dob", None), getattr(mm, "b_dob", None)
        elif aid == blue_id:
            adob, ddob = getattr(mm, "b_dob", None), getattr(mm, "r_dob", None)
        else:
            skipped["bad_ids"] += 1
            continue

        round_num = int(rr.round)
        is_finish_round = round_num == int(mm.finish_round)
        target = int(_is_ko(mm.method) and str(mm.winner_id) == aid and is_finish_round)
        rows.append({
            "fight_id": fid,
            "fight_date": mm.fight_date,
            "round": round_num,
            "attacker_id": aid,
            "defender_id": did,
            "target_ko_round": target,
            "target_ko_fight": int(_is_ko(mm.method) and str(mm.winner_id) == aid),
            "sig_landed_exposure": float(rr.sig_str_landed),
            "current_round_kds_audit_only": float(rr.kd),
            "prior_defender_kds": float(rr.prior_defender_kds),
            "elapsed_minutes": float((round_num - 1) * 5.0),
            "attacker_power_centered": float(legacy_power_equivalent(float(aa.striking_power_v3))) - 50.0,
            "attacker_age_centered": _age(adob, mm.fight_date) - 30.0,
            "defender_age_centered": _age(ddob, mm.fight_date) - 30.0,
            "defender_kdres_centered": float(legacy_kdres_equivalent(float(dd.knockdown_resistance_v3))) - 50.0,
            "defender_durability_centered": float(dd.damage_durability) - 50.0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no fighter-round rows joined")
    out = out.sort_values(["fight_date", "fight_id", "round", "attacker_id"]).reset_index(drop=True)
    return out, skipped


def chronological_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    fights = df[["fight_id", "fight_date"]].drop_duplicates().sort_values(["fight_date", "fight_id"])
    cut = int(np.floor(len(fights) * (1.0 - test_fraction)))
    cut = min(max(cut, 1), len(fights) - 1)
    train_ids = set(fights.iloc[:cut].fight_id)
    test_ids = set(fights.iloc[cut:].fight_id)
    return df[df.fight_id.isin(train_ids)].copy(), df[df.fight_id.isin(test_ids)].copy(), fights.iloc[cut].fight_date.date().isoformat()


@dataclass
class GroupedHazardModel:
    features: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    beta_scaled: np.ndarray
    intercept_raw: float
    coefficients_raw: dict[str, float]
    success: bool
    message: str

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.column_stack([
            pd.to_numeric(df[f], errors="coerce").fillna(self.medians[f]).to_numpy(float)
            for f in self.features
        ])
        eta = np.full(len(df), self.intercept_raw, dtype=float)
        for j, f in enumerate(self.features):
            eta += self.coefficients_raw[f] * x[:, j]
        p_strike = _sigmoid(eta)
        n = np.clip(pd.to_numeric(df.sig_landed_exposure, errors="coerce").fillna(0).to_numpy(float), 0, None)
        log_survival = n * np.log1p(-np.clip(p_strike, 1e-12, 1 - 1e-12))
        p_round = -np.expm1(log_survival)
        return p_strike, p_round, log_survival


def fit_grouped_logistic(train: pd.DataFrame, features: list[str]) -> GroupedHazardModel:
    medians, means, scales = {}, {}, {}
    cols = []
    for f in features:
        s = pd.to_numeric(train[f], errors="coerce")
        med = float(s.median()) if s.notna().any() else 0.0
        v = s.fillna(med).to_numpy(float)
        mean = float(v.mean())
        scale = float(v.std(ddof=0)) or 1.0
        medians[f], means[f], scales[f] = med, mean, scale
        cols.append((v - mean) / scale)
    X = np.column_stack(cols) if cols else np.empty((len(train), 0))
    n = np.clip(train.sig_landed_exposure.to_numpy(float), 0, None)
    y = train.target_ko_round.to_numpy(float)

    positive_zero = (y == 1) & (n <= 0)
    if positive_zero.any():
        keep = ~positive_zero
        X, n, y = X[keep], n[keep], y[keep]

    total_exposure = max(float(n.sum()), 1.0)
    crude = float(np.clip(y.sum() / total_exposure, 1e-5, 0.05))
    theta0 = np.zeros(X.shape[1] + 1, dtype=float)
    theta0[0] = _logit(crude)

    def objective(theta: np.ndarray) -> float:
        eta = theta[0] + X @ theta[1:]
        p = _sigmoid(eta)
        log_surv = n * np.log1p(-np.clip(p, 1e-12, 1 - 1e-12))
        p_round = -np.expm1(log_surv)
        ll = np.where(y > 0.5, np.log(np.clip(p_round, 1e-15, 1.0)), log_surv)
        ridge = 1e-6 * float(np.dot(theta[1:], theta[1:]))
        return -float(ll.sum()) + ridge

    opt = minimize(objective, theta0, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12})
    raw = {f: float(opt.x[j + 1] / scales[f]) for j, f in enumerate(features)}
    intercept = float(opt.x[0] - sum(raw[f] * means[f] for f in features))
    return GroupedHazardModel(features, medians, means, scales, opt.x, intercept, raw, bool(opt.success), str(opt.message))


def current_fixed_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.full(len(df), _logit(CURRENT["base_probability"]), dtype=float)
    for f, b in CURRENT.items():
        if f == "base_probability":
            continue
        vals = pd.to_numeric(df[f], errors="coerce").fillna(0).to_numpy(float)
        eta += float(b) * vals
    p = _sigmoid(eta)
    n = np.clip(df.sig_landed_exposure.to_numpy(float), 0, None)
    log_surv = n * np.log1p(-np.clip(p, 1e-12, 1 - 1e-12))
    return p, -np.expm1(log_surv), log_surv


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def aggregate_fight(df: pd.DataFrame, log_surv: np.ndarray) -> pd.DataFrame:
    x = df[["fight_id", "fight_date", "attacker_id", "defender_id", "target_ko_fight"]].copy()
    x["log_survival"] = log_surv
    g = x.groupby(["fight_id", "fight_date", "attacker_id", "defender_id"], as_index=False).agg(
        target_ko_fight=("target_ko_fight", "max"), log_survival=("log_survival", "sum")
    )
    g["p_ko_fight_exposure_conditioned"] = -np.expm1(g.log_survival.to_numpy(float))
    return g


def metrics(df: pd.DataFrame, p_round: np.ndarray, log_surv: np.ndarray) -> dict:
    y = df.target_ko_round.to_numpy(int)
    p = np.clip(p_round, 1e-15, 1 - 1e-15)
    fight = aggregate_fight(df, log_surv)
    yf = fight.target_ko_fight.to_numpy(int)
    pf = np.clip(fight.p_ko_fight_exposure_conditioned.to_numpy(float), 1e-15, 1 - 1e-15)
    return {
        "round_n": int(len(df)),
        "round_positive": int(y.sum()),
        "round_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "round_brier": float(brier_score_loss(y, p)),
        "round_auc": _safe_auc(y, p),
        "round_mean_pred": float(p.mean()),
        "round_observed_rate": float(y.mean()),
        "directional_fight_n": int(len(fight)),
        "fight_positive": int(yf.sum()),
        "fight_log_loss_exposure_conditioned": float(log_loss(yf, pf, labels=[0, 1])),
        "fight_brier_exposure_conditioned": float(brier_score_loss(yf, pf)),
        "fight_auc_exposure_conditioned": _safe_auc(yf, pf),
        "fight_mean_pred_exposure_conditioned": float(pf.mean()),
        "fight_observed_rate": float(yf.mean()),
    }


def bootstrap_prediction_delta(test: pd.DataFrame, base_p: np.ndarray, alt_p: np.ndarray, n_boot: int) -> dict:
    x = test[["fight_id", "target_ko_round"]].copy()
    x["base"] = np.clip(base_p, 1e-15, 1 - 1e-15)
    x["alt"] = np.clip(alt_p, 1e-15, 1 - 1e-15)
    by = {fid: g for fid, g in x.groupby("fight_id", sort=False)}
    ids = np.array(list(by), dtype=object)
    rng = np.random.default_rng(20260826)
    dll, dbr = [], []
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        g = pd.concat([by[f] for f in sampled], ignore_index=True)
        y = g.target_ko_round.to_numpy(int)
        dll.append(log_loss(y, g.alt, labels=[0, 1]) - log_loss(y, g.base, labels=[0, 1]))
        dbr.append(brier_score_loss(y, g.alt) - brier_score_loss(y, g.base))
    def summarize(v: list[float]) -> dict:
        a = np.asarray(v, float)
        return {"mean": float(a.mean()), "p2_5": float(np.quantile(a, .025)), "p97_5": float(np.quantile(a, .975)), "improvement_share": float(np.mean(a < 0))}
    return {"delta_round_log_loss": summarize(dll), "delta_round_brier": summarize(dbr)}


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    d = pd.DataFrame({"y": y, "p": p})
    try:
        d["bin"] = pd.qcut(d.p, q=bins, duplicates="drop")
    except ValueError:
        return []
    out = []
    for label, g in d.groupby("bin", observed=True):
        out.append({"bin": str(label), "n": int(len(g)), "mean_pred": float(g.p.mean()), "observed": float(g.y.mean())})
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(args.master_path)
    rounds = pd.read_parquet(args.round_stats_path)
    prefight = pd.read_parquet(args.prefight_path)

    data, skipped = build_rows(master, rounds, prefight)
    positive_zero = data[(data.target_ko_round == 1) & (data.sig_landed_exposure <= 0)].copy()
    clean = data.loc[~((data.target_ko_round == 1) & (data.sig_landed_exposure <= 0))].copy()
    train, test, cutoff = chronological_split(clean, args.test_fraction)

    summary = {
        "study": "event2_ko_hazard_v2_grouped_significant_strike_likelihood",
        "production_changed": False,
        "hazard_semantics": "one logistic KO roll per landed significant-strike-equivalent event; fighter-round grouped likelihood",
        "important_limitations": [
            "UFCStats has aggregate fighter-round strikes, not event timestamps.",
            "Current-round KD ordering is unknown, so current-round KD is audit-only and not a predictor.",
            "Fight-level probabilities below are conditioned on historically observed landed-strike exposure and observed rounds, not standalone prefight predictions.",
        ],
        "joined_rows": int(len(data)),
        "joined_fights": int(data.fight_id.nunique()),
        "skipped": skipped,
        "positive_ko_rounds_with_zero_sig_landed_excluded": int(len(positive_zero)),
        "train_fights": int(train.fight_id.nunique()),
        "test_fights": int(test.fight_id.nunique()),
        "test_starts": cutoff,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "models": {},
    }

    current_p_strike, current_p_round, current_log_surv = current_fixed_predictions(test)
    current_metrics = metrics(test, current_p_round, current_log_surv)
    current_metrics["coefficients_raw"] = {"intercept": _logit(CURRENT["base_probability"]), **{k: v for k, v in CURRENT.items() if k != "base_probability"}}
    current_metrics["round_calibration"] = calibration_bins(test.target_ko_round.to_numpy(int), current_p_round)
    summary["models"]["PRODUCTION_FIXED"] = current_metrics

    preds = {"PRODUCTION_FIXED": current_p_round}
    for name, features in MODEL_FEATURES.items():
        model = fit_grouped_logistic(train, features)
        p_strike, p_round, log_surv = model.predict(test)
        mm = metrics(test, p_round, log_surv)
        mm.update({
            "fit_success": model.success,
            "fit_message": model.message,
            "features": features,
            "intercept_raw": model.intercept_raw,
            "base_per_strike_probability_at_all_centered_zero": float(_sigmoid(np.array([model.intercept_raw]))[0]),
            "coefficients_raw": model.coefficients_raw,
            "round_calibration": calibration_bins(test.target_ko_round.to_numpy(int), p_round),
        })
        mm["bootstrap_vs_production_fixed"] = bootstrap_prediction_delta(test, current_p_round, p_round, args.bootstrap)
        summary["models"][name] = mm
        preds[name] = p_round

    base = preds["A_refit_current_inputs"]
    for name in MODEL_FEATURES:
        if name == "A_refit_current_inputs":
            continue
        summary["models"][name]["bootstrap_vs_A_refit"] = bootstrap_prediction_delta(test, base, preds[name], args.bootstrap)

    # Schnell-Costa is in the chronological holdout. Audit its historical round exposures
    # and per-round predictions; this is diagnostic only and is not used for fitting.
    sc = test[test.fight_id.eq(SCHNELL_COSTA_FIGHT_ID)].copy()
    audit = []
    if not sc.empty:
        for idx, row in sc.iterrows():
            rec = {
                "round": int(row["round"]),
                "attacker_id": str(row.attacker_id),
                "defender_id": str(row.defender_id),
                "sig_landed_exposure": float(row.sig_landed_exposure),
                "current_round_kds_audit_only": float(row.current_round_kds_audit_only),
                "prior_defender_kds": float(row.prior_defender_kds),
                "target_ko_round": int(row.target_ko_round),
                "defender_kdres_legacy": float(row.defender_kdres_centered + 50),
                "defender_durability": float(row.defender_durability_centered + 50),
            }
            pos = test.index.get_loc(idx)
            rec["p_round_PRODUCTION_FIXED"] = float(preds["PRODUCTION_FIXED"][pos])
            for name in MODEL_FEATURES:
                rec[f"p_round_{name}"] = float(preds[name][pos])
            audit.append(rec)
    summary["schnell_costa_holdout_round_audit"] = audit

    predictions = test[["fight_id", "fight_date", "round", "attacker_id", "defender_id", "target_ko_round", "target_ko_fight", "sig_landed_exposure", "current_round_kds_audit_only", "prior_defender_kds"]].copy()
    for name, p in preds.items():
        predictions[f"p_round_{name}"] = p
    predictions.to_csv(outdir / "holdout_round_predictions.csv", index=False)
    if len(positive_zero):
        positive_zero.to_csv(outdir / "zero_sig_landed_ko_rounds.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("EVENT2_KO_HAZARD_V2_STUDY")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
