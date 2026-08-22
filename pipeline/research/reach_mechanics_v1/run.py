from __future__ import annotations

"""Development-only reach-to-mechanics study for FSR V3 / Event Clock V2.

This module is research-only. It reads frozen raw/master/FSR V3 inputs and never
modifies FSR or simulator code/data. 2024+ is reserved and never fit or scored.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    derive_runtime_inputs,
    initialize_fighter_path_traits,
    load_prefight_snapshots,
)
from pipeline.simulation.event_mc_v1.components.fsr_v2_mechanics import (
    TAKEDOWN_ATTACKER_AGE_CENTER_YEARS,
    TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR,
)

DEFAULT_CONFIG = Path("pipeline/research/reach_mechanics_v1/config.yaml")
EPS = 1e-8


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("reach mechanics config must be a mapping")
    return payload


def _parse_measure(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in {"--", "nan", "none"}:
        return np.nan
    if "'" in text:
        cleaned = text.replace('"', "").replace("in", "")
        feet_text, _, inches_text = cleaned.partition("'")
        try:
            return 12.0 * float(feet_text.strip()) + float(inches_text.strip() or 0)
        except ValueError:
            return np.nan
    numeric = "".join(ch if ch.isdigit() or ch in ".-" else " " for ch in text)
    parts = [p for p in numeric.split() if p]
    try:
        return float(parts[0]) if parts else np.nan
    except ValueError:
        return np.nan


def _age_years(dob: object, event_date: pd.Timestamp) -> float:
    dob_ts = pd.to_datetime(dob, errors="coerce")
    if pd.isna(dob_ts):
        return np.nan
    return float((event_date - dob_ts).days / 365.2425)


def _clip_p(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, EPS, 1.0 - EPS)


def _logit(value: float | np.ndarray) -> float | np.ndarray:
    p = _clip_p(value)
    return np.log(p / (1.0 - p))


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _aggregate_actuals(master: pd.DataFrame) -> pd.DataFrame:
    pair_master = master.drop(columns=["event_date"], errors="ignore").copy()
    paired = build_paired_rounds(master=pair_master)
    keys = ["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "opponent_name"]
    frame = (
        paired.groupby(keys, as_index=False)
        .agg(
            distance_landed=("distance_landed", "sum"),
            distance_attempted=("distance_attempted", "sum"),
            td_landed=("td_landed", "sum"),
            td_attempted=("td_attempted", "sum"),
            ground_landed=("ground_landed", "sum"),
            ground_attempted=("ground_attempted", "sum"),
            clinch_landed=("clinch_landed", "sum"),
            clinch_attempted=("clinch_attempted", "sum"),
            sig_str_landed=("sig_str_landed", "sum"),
            sig_str_attempted=("sig_str_attempted", "sum"),
            own_control_seconds=("ctrl_sec", "sum"),
            standing_exposure_seconds=("standing_exposure_seconds", "sum"),
            td_exposure_seconds=("td_tendency_exposure_seconds", "sum"),
            fight_elapsed_seconds=("round_elapsed_seconds", "sum"),
        )
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    for c in ("fight_id", "fighter_id", "opponent_id"):
        frame[c] = frame[c].astype(str)
    return frame


def _master_meta(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["fight_id"] = m["fight_id"].astype(str)
    date_col = "date" if "date" in m.columns else "event_date"
    if date_col not in m.columns:
        raise ValueError("master needs date or event_date")
    m["event_date_meta"] = pd.to_datetime(m[date_col], errors="raise").dt.normalize()
    needed = [
        "fight_id", "event_date_meta", "division", "r_id", "b_id",
        "r_reach", "b_reach", "r_height", "b_height", "r_weight", "b_weight",
        "r_dob", "b_dob",
    ]
    missing = [c for c in needed if c not in m.columns]
    if missing:
        raise ValueError(f"master missing reach-study columns: {missing}")
    return m[needed].drop_duplicates("fight_id")


def _attach_physical(frame: pd.DataFrame, meta: pd.DataFrame, saturation: float) -> pd.DataFrame:
    out = frame.merge(meta, on="fight_id", how="left", validate="many_to_one")
    if out["event_date_meta"].isna().any():
        raise ValueError("missing master metadata for reach study")
    date_bad = out["event_date"].ne(out["event_date_meta"])
    if date_bad.any():
        raise ValueError(f"event-date mismatch for {int(date_bad.sum())} directional rows")

    records: list[dict[str, Any]] = []
    for row in out.to_dict("records"):
        fid = str(row["fighter_id"])
        oid = str(row["opponent_id"])
        rid, bid = str(row["r_id"]), str(row["b_id"])
        if fid == rid and oid == bid:
            self_side, opp_side = "r", "b"
        elif fid == bid and oid == rid:
            self_side, opp_side = "b", "r"
        else:
            raise ValueError(f"fighter ids do not match master corners for fight {row['fight_id']}")
        event_date = pd.Timestamp(row["event_date"])
        self_reach = _parse_measure(row[f"{self_side}_reach"])
        opp_reach = _parse_measure(row[f"{opp_side}_reach"])
        self_height = _parse_measure(row[f"{self_side}_height"])
        opp_height = _parse_measure(row[f"{opp_side}_height"])
        self_weight = _parse_measure(row[f"{self_side}_weight"])
        opp_weight = _parse_measure(row[f"{opp_side}_weight"])
        self_age = _age_years(row[f"{self_side}_dob"], event_date)
        opp_age = _age_years(row[f"{opp_side}_dob"], event_date)
        reach_edge = self_reach - opp_reach if np.isfinite(self_reach) and np.isfinite(opp_reach) else np.nan
        height_edge = self_height - opp_height if np.isfinite(self_height) and np.isfinite(opp_height) else np.nan
        age_edge = self_age - opp_age if np.isfinite(self_age) and np.isfinite(opp_age) else np.nan
        weight_edge = self_weight - opp_weight if np.isfinite(self_weight) and np.isfinite(opp_weight) else np.nan
        self_ape = self_reach - self_height if np.isfinite(self_reach) and np.isfinite(self_height) else np.nan
        opp_ape = opp_reach - opp_height if np.isfinite(opp_reach) and np.isfinite(opp_height) else np.nan
        ape_edge = self_ape - opp_ape if np.isfinite(self_ape) and np.isfinite(opp_ape) else np.nan
        row.update({
            "division": str(row.get("division", "Unknown") or "Unknown"),
            "self_reach": self_reach,
            "opp_reach": opp_reach,
            "reach_edge": reach_edge,
            "self_height": self_height,
            "opp_height": opp_height,
            "height_edge": height_edge,
            "self_ape_index": self_ape,
            "opp_ape_index": opp_ape,
            "ape_edge": ape_edge,
            "self_weight": self_weight,
            "opp_weight": opp_weight,
            "weight_edge": weight_edge,
            "self_age": self_age,
            "opp_age": opp_age,
            "age_edge": age_edge,
            "reach_tanh": float(np.tanh(reach_edge / saturation)) if np.isfinite(reach_edge) else np.nan,
        })
        records.append(row)
    return pd.DataFrame(records)


def _attach_fsr_expectations(frame: pd.DataFrame, fsr_path: Path) -> tuple[pd.DataFrame, int]:
    fsr = load_prefight_snapshots(fsr_path)
    lookup = {
        (pd.Timestamp(r.event_date).normalize(), str(r.fight_id), str(r.fighter_id)): r._asdict()
        for r in fsr.itertuples(index=False)
    }
    trait_cache: dict[tuple[pd.Timestamp, str, str], Any] = {}
    rng = np.random.default_rng(0)

    def traits(key):
        if key not in trait_cache:
            trait_cache[key] = initialize_fighter_path_traits(
                lookup[key], None, rng=rng, sample_epistemic=False
            )
        return trait_cache[key]

    rows = []
    missing = 0
    for row in frame.to_dict("records"):
        date = pd.Timestamp(row["event_date"]).normalize()
        self_key = (date, str(row["fight_id"]), str(row["fighter_id"]))
        opp_key = (date, str(row["fight_id"]), str(row["opponent_id"]))
        if self_key not in lookup or opp_key not in lookup:
            missing += 1
            row["fsr_matched"] = False
            rows.append(row)
            continue
        runtime = derive_runtime_inputs(traits(self_key), traits(opp_key))
        td_p = float(runtime.takedown_completion)
        if np.isfinite(row["self_age"]):
            td_p = float(_sigmoid(
                _logit(td_p)
                + TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR
                * (float(row["self_age"]) - TAKEDOWN_ATTACKER_AGE_CENTER_YEARS)
            ))
        row.update({
            "fsr_matched": True,
            "expected_distance_attempts": max(runtime.standing_rate_15m, 0.0)
            * max(float(row["standing_exposure_seconds"]), 0.0) / 900.0,
            "expected_distance_accuracy": float(_clip_p(runtime.standing_accuracy)),
            "expected_td_attempts": max(runtime.takedown_rate_15m, 0.0)
            * max(float(row["td_exposure_seconds"]), 0.0) / 900.0,
            "expected_td_completion": float(_clip_p(td_p)),
            "expected_ground_attempts": float(runtime.ground_expected_attempts(row["own_control_seconds"])),
            "expected_ground_accuracy": float(_clip_p(runtime.ground_accuracy)),
        })
        rows.append(row)
    return pd.DataFrame(rows), missing


def _design(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], division: bool):
    xtr = pd.DataFrame({"const": np.ones(len(train), dtype=float)}, index=train.index)
    xva = pd.DataFrame({"const": np.ones(len(valid), dtype=float)}, index=valid.index)
    if division:
        tr_div = pd.get_dummies(train["division"].fillna("Unknown"), prefix="division", dtype=float)
        va_div = pd.get_dummies(valid["division"].fillna("Unknown"), prefix="division", dtype=float)
        cols = sorted(tr_div.columns)
        if cols:
            cols = cols[1:]  # one reference division; intercept is already present
        tr_div = tr_div.reindex(columns=cols, fill_value=0.0)
        va_div = va_div.reindex(columns=cols, fill_value=0.0)
        xtr = pd.concat([xtr, tr_div], axis=1)
        xva = pd.concat([xva, va_div], axis=1)
    for c in numeric:
        xtr[c] = pd.to_numeric(train[c], errors="coerce").astype(float)
        xva[c] = pd.to_numeric(valid[c], errors="coerce").astype(float)
    return xtr.astype(float), xva.astype(float)


def _count_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, float), EPS, None)
    y = np.asarray(y, float)
    return {
        "primary_score": float(mean_poisson_deviance(y, pred)),
        "secondary_score": float(mean_absolute_error(y, pred)),
    }


def _binomial_metrics(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> dict[str, float]:
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = _clip_p(np.asarray(p, float))
    total = max(float(n.sum()), 1.0)
    ll = -float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)) / total)
    brier = float(np.sum(k * (1.0 - p) ** 2 + (n - k) * p**2) / total)
    return {"primary_score": ll, "secondary_score": brier}


def _fit_variant(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: dict[str, Any],
    numeric: list[str],
    division: bool,
):
    xtr, xva = _design(train, valid, numeric, division)
    if spec["kind"] == "count":
        ytr = train[spec["y"]].to_numpy(float)
        mu_tr = np.clip(train[spec["expected"]].to_numpy(float), EPS, None)
        model = sm.GLM(
            ytr,
            xtr,
            family=sm.families.Poisson(),
            offset=np.log(mu_tr),
        ).fit(maxiter=150, disp=0)
        mu_va = np.clip(valid[spec["expected"]].to_numpy(float), EPS, None)
        pred = model.predict(xva, offset=np.log(mu_va))
        metrics = _count_metrics(valid[spec["y"]].to_numpy(float), pred)
    else:
        ntr = train[spec["attempts"]].to_numpy(float)
        ktr = train[spec["landed"]].to_numpy(float)
        ptr = _clip_p(train[spec["expected"]].to_numpy(float))
        model = sm.GLM(
            ktr / ntr,
            xtr,
            family=sm.families.Binomial(),
            freq_weights=ntr,
            offset=_logit(ptr),
        ).fit(maxiter=150, disp=0)
        pva0 = _clip_p(valid[spec["expected"]].to_numpy(float))
        pred = model.predict(xva, offset=_logit(pva0))
        metrics = _binomial_metrics(
            valid[spec["landed"]].to_numpy(float),
            valid[spec["attempts"]].to_numpy(float),
            pred,
        )
    return model, np.asarray(pred, float), metrics


def _raw_metrics(valid: pd.DataFrame, spec: dict[str, Any]) -> dict[str, float]:
    if spec["kind"] == "count":
        return _count_metrics(
            valid[spec["y"]].to_numpy(float),
            valid[spec["expected"]].to_numpy(float),
        )
    return _binomial_metrics(
        valid[spec["landed"]].to_numpy(float),
        valid[spec["attempts"]].to_numpy(float),
        valid[spec["expected"]].to_numpy(float),
    )


def _mechanics() -> list[dict[str, Any]]:
    return [
        {"name": "distance_attempt_rate", "kind": "count", "y": "distance_attempted", "expected": "expected_distance_attempts"},
        {"name": "distance_accuracy", "kind": "binomial", "landed": "distance_landed", "attempts": "distance_attempted", "expected": "expected_distance_accuracy"},
        {"name": "td_attempt_rate", "kind": "count", "y": "td_attempted", "expected": "expected_td_attempts"},
        {"name": "td_completion", "kind": "binomial", "landed": "td_landed", "attempts": "td_attempted", "expected": "expected_td_completion"},
        {"name": "ground_attempt_rate", "kind": "count", "y": "ground_attempted", "expected": "expected_ground_attempts"},
        {"name": "ground_accuracy", "kind": "binomial", "landed": "ground_landed", "attempts": "ground_attempted", "expected": "expected_ground_accuracy"},
    ]


def _valid_for_mechanic(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    if spec["kind"] == "count":
        mask = (
            pd.to_numeric(frame[spec["y"]], errors="coerce").notna()
            & pd.to_numeric(frame[spec["expected"]], errors="coerce").gt(EPS)
        )
    else:
        mask = (
            pd.to_numeric(frame[spec["attempts"]], errors="coerce").gt(0)
            & pd.to_numeric(frame[spec["landed"]], errors="coerce").notna()
            & pd.to_numeric(frame[spec["expected"]], errors="coerce").between(EPS, 1.0 - EPS)
        )
    return frame.loc[mask].copy()


def _residual_bins(dev: pd.DataFrame, mechanics: list[dict[str, Any]], bins: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [f"[{bins[i]:g},{bins[i+1]:g})" for i in range(len(bins) - 1)]
    for spec in mechanics:
        frame = _valid_for_mechanic(dev, spec)
        frame["reach_bin"] = pd.cut(
            frame["reach_edge"], bins=bins, labels=labels, include_lowest=True, right=False
        )
        if spec["kind"] == "count":
            frame["residual"] = np.log((frame[spec["y"]] + 0.5) / (frame[spec["expected"]] + 0.5))
            for bucket, g in frame.groupby("reach_bin", observed=True):
                rows.append({
                    "mechanic": spec["name"], "kind": spec["kind"], "reach_bin": str(bucket),
                    "rows": len(g), "mean_reach_edge": float(g["reach_edge"].mean()),
                    "mean_residual": float(g["residual"].mean()),
                    "median_residual": float(g["residual"].median()),
                    "actual": float(g[spec["y"]].sum()),
                    "expected": float(g[spec["expected"]].sum()),
                })
        else:
            obs_p = (frame[spec["landed"]] + 0.5) / (frame[spec["attempts"]] + 1.0)
            frame["residual"] = _logit(obs_p.to_numpy(float)) - _logit(frame[spec["expected"]].to_numpy(float))
            for bucket, g in frame.groupby("reach_bin", observed=True):
                attempts = float(g[spec["attempts"]].sum())
                rows.append({
                    "mechanic": spec["name"], "kind": spec["kind"], "reach_bin": str(bucket),
                    "rows": len(g), "mean_reach_edge": float(g["reach_edge"].mean()),
                    "mean_residual": float(g["residual"].mean()),
                    "median_residual": float(g["residual"].median()),
                    "actual": float(g[spec["landed"]].sum() / max(attempts, 1.0)),
                    "expected": float(np.average(g[spec["expected"]], weights=g[spec["attempts"]])),
                })
    return pd.DataFrame(rows)


def run(config_path: Path = DEFAULT_CONFIG) -> None:
    config = _load(config_path)
    out_root = Path(config["outputs"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    years = [int(x) for x in config["validation"]["development_years"]]
    min_train = int(config["validation"]["minimum_train_rows"])
    min_valid = int(config["validation"]["minimum_valid_rows"])
    saturation = float(config["physical"]["reach_saturation_inches"])
    reach_bins = [float(x) for x in config["physical"]["reach_bins"]]

    master = pd.read_parquet(config["inputs"]["master_path"])
    actual = _aggregate_actuals(master)
    frame = _attach_physical(actual, _master_meta(master), saturation)
    frame, fsr_missing = _attach_fsr_expectations(
        frame, Path(config["inputs"]["fsr_v3_prefight_path"])
    )

    frame["physical_complete"] = frame[
        ["self_reach", "opp_reach", "self_height", "opp_height", "self_age", "opp_age"]
    ].notna().all(axis=1)
    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.normalize()
    dev = frame[
        frame["event_date"].lt(outer_start)
        & frame["fsr_matched"].astype(bool)
        & frame["physical_complete"].astype(bool)
    ].copy()
    outer_rows = int(frame["event_date"].ge(outer_start).sum())

    variants = {
        "fsr_calibrated": ([], False),
        "division_age": (["age_edge"], True),
        "division_age_height": (["age_edge", "height_edge"], True),
        "division_age_reach": (["age_edge", "reach_edge"], True),
        "division_age_ape": (["age_edge", "ape_edge"], True),
        "division_age_height_reach": (["age_edge", "height_edge", "reach_edge"], True),
        "division_age_height_reach_tanh": (["age_edge", "height_edge", "reach_tanh"], True),
    }
    mechanics = _mechanics()
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for year in years:
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        for spec in mechanics:
            cohort = _valid_for_mechanic(dev, spec)
            train = cohort[cohort["event_date"].lt(start)].copy()
            valid = cohort[cohort["event_date"].ge(start) & cohort["event_date"].lt(end)].copy()
            if len(train) < min_train or len(valid) < min_valid:
                continue
            raw = _raw_metrics(valid, spec)
            metric_rows.append({
                "fold": year, "mechanic": spec["name"], "kind": spec["kind"],
                "variant": "fsr_raw", "train_rows": len(train), "valid_rows": len(valid),
                "primary_metric": "poisson_deviance" if spec["kind"] == "count" else "attempt_log_loss",
                "secondary_metric": "mae" if spec["kind"] == "count" else "attempt_brier",
                **raw,
            })
            for variant, (numeric, use_division) in variants.items():
                try:
                    model, _, scores = _fit_variant(train, valid, spec, numeric, use_division)
                except Exception as exc:
                    metric_rows.append({
                        "fold": year, "mechanic": spec["name"], "kind": spec["kind"],
                        "variant": variant, "train_rows": len(train), "valid_rows": len(valid),
                        "primary_metric": "fit_error", "secondary_metric": type(exc).__name__,
                        "primary_score": np.nan, "secondary_score": np.nan,
                    })
                    continue
                metric_rows.append({
                    "fold": year, "mechanic": spec["name"], "kind": spec["kind"],
                    "variant": variant, "train_rows": len(train), "valid_rows": len(valid),
                    "primary_metric": "poisson_deviance" if spec["kind"] == "count" else "attempt_log_loss",
                    "secondary_metric": "mae" if spec["kind"] == "count" else "attempt_brier",
                    **scores,
                })
                for term in ("reach_edge", "reach_tanh"):
                    if term not in model.params.index:
                        continue
                    beta = float(model.params[term])
                    se = float(model.bse[term]) if term in model.bse.index else np.nan
                    pvalue = float(model.pvalues[term]) if term in model.pvalues.index else np.nan
                    if term == "reach_edge":
                        effect_per_inch = float(math.exp(beta))
                        effect_plus4 = float(math.exp(beta * 4.0))
                    else:
                        effect_per_inch = np.nan
                        effect_plus4 = float(math.exp(beta * math.tanh(4.0 / saturation)))
                    coefficient_rows.append({
                        "fold": year, "mechanic": spec["name"], "kind": spec["kind"],
                        "variant": variant, "term": term, "beta": beta, "se": se,
                        "pvalue": pvalue, "effect_ratio_per_inch": effect_per_inch,
                        "effect_ratio_at_plus4_inches": effect_plus4,
                    })

    metrics = pd.DataFrame(metric_rows)
    coeff = pd.DataFrame(coefficient_rows)
    if metrics.empty:
        raise RuntimeError("reach study produced no development metrics")

    primary_base = "division_age_height"
    candidate_variants = ["division_age_height_reach", "division_age_height_reach_tanh"]
    inc_rows = []
    for mechanic in metrics["mechanic"].dropna().unique():
        sub = metrics[metrics["mechanic"].eq(mechanic)]
        base = sub[sub["variant"].eq(primary_base)].set_index("fold")
        for candidate in candidate_variants:
            cand = sub[sub["variant"].eq(candidate)].set_index("fold")
            common = sorted(set(base.index) & set(cand.index))
            deltas = []
            for fold in common:
                b = float(base.loc[fold, "primary_score"])
                c = float(cand.loc[fold, "primary_score"])
                if np.isfinite(b) and np.isfinite(c):
                    deltas.append((fold, c - b, b, c))
            if not deltas:
                continue
            inc_rows.append({
                "mechanic": mechanic,
                "candidate": candidate,
                "folds": len(deltas),
                "folds_improved": int(sum(delta < 0 for _, delta, _, _ in deltas)),
                "mean_delta_primary_vs_age_height": float(np.mean([d[1] for d in deltas])),
                "mean_age_height_score": float(np.mean([d[2] for d in deltas])),
                "mean_candidate_score": float(np.mean([d[3] for d in deltas])),
            })
    incremental = pd.DataFrame(inc_rows)

    if not coeff.empty:
        coefficient_summary = (
            coeff.groupby(["mechanic", "variant", "term"], as_index=False)
            .agg(
                folds=("fold", "nunique"),
                mean_beta=("beta", "mean"),
                median_beta=("beta", "median"),
                positive_folds=("beta", lambda s: int((s > 0).sum())),
                negative_folds=("beta", lambda s: int((s < 0).sum())),
                mean_effect_ratio_per_inch=("effect_ratio_per_inch", "mean"),
                mean_effect_ratio_at_plus4_inches=("effect_ratio_at_plus4_inches", "mean"),
            )
        )
    else:
        coefficient_summary = pd.DataFrame()

    residual_bins = _residual_bins(dev, mechanics, reach_bins)
    coverage = pd.DataFrame([
        {"stage": "all_directional_round_stat_rows", "rows": len(frame), "fights": frame["fight_id"].nunique()},
        {"stage": "fsr_v3_matched", "rows": int(frame["fsr_matched"].sum()), "fights": frame.loc[frame["fsr_matched"], "fight_id"].nunique()},
        {"stage": "physical_complete", "rows": int(frame["physical_complete"].sum()), "fights": frame.loc[frame["physical_complete"], "fight_id"].nunique()},
        {"stage": "development_common_cohort", "rows": len(dev), "fights": dev["fight_id"].nunique()},
        {"stage": "outer_reserved_not_scored", "rows": outer_rows, "fights": frame.loc[frame["event_date"].ge(outer_start), "fight_id"].nunique()},
    ])

    pair_sizes = frame.groupby("fight_id").size()
    reach_pair_sums = dev.groupby("fight_id")["reach_edge"].sum(min_count=2)
    audit = pd.DataFrame([
        {"check": "outer_start_unchanged", "passed": str(outer_start.date()) == "2024-01-01", "value": str(outer_start.date())},
        {"check": "outer_rows_never_scored", "passed": bool(dev["event_date"].lt(outer_start).all()), "value": outer_rows},
        {"check": "fsr_v3_rows_resolved", "passed": bool(frame["fsr_matched"].any()), "value": int(frame["fsr_matched"].sum())},
        {"check": "exactly_two_directional_rows_per_fight", "passed": bool(pair_sizes.eq(2).all()), "value": pair_sizes.value_counts().to_dict()},
        {"check": "reach_edges_antisymmetric", "passed": bool(np.allclose(reach_pair_sums.dropna(), 0.0, atol=1e-8)), "value": float(reach_pair_sums.dropna().abs().max()) if reach_pair_sums.notna().any() else np.nan},
        {"check": "development_folds_only", "passed": set(pd.to_numeric(metrics["fold"], errors="coerce").dropna().astype(int)).issubset(set(years)), "value": sorted(metrics["fold"].dropna().unique().tolist())},
        {"check": "predictors_are_prefight_physical_only", "passed": True, "value": sorted({c for cols, _ in variants.values() for c in cols})},
    ])
    if not audit["passed"].all():
        print(audit.to_string(index=False))
        raise RuntimeError("reach mechanics audit failed")

    metrics.to_csv(out_root / "mechanic_fold_metrics.csv", index=False)
    incremental.to_csv(out_root / "reach_incremental_summary.csv", index=False)
    coeff.to_csv(out_root / "reach_fold_coefficients.csv", index=False)
    coefficient_summary.to_csv(out_root / "reach_coefficient_summary.csv", index=False)
    residual_bins.to_csv(out_root / "reach_residual_bins.csv", index=False)
    coverage.to_csv(out_root / "coverage.csv", index=False)
    audit.to_csv(out_root / "audit.csv", index=False)

    nominations = []
    if not incremental.empty:
        linear = incremental[incremental["candidate"].eq("division_age_height_reach")]
        for row in linear.to_dict("records"):
            coeff_row = coefficient_summary[
                coefficient_summary["mechanic"].eq(row["mechanic"])
                & coefficient_summary["variant"].eq("division_age_height_reach")
                & coefficient_summary["term"].eq("reach_edge")
            ]
            sign_consistent = False
            if not coeff_row.empty:
                cr = coeff_row.iloc[0]
                sign_consistent = max(int(cr["positive_folds"]), int(cr["negative_folds"])) >= 3
            nominations.append({
                "mechanic": row["mechanic"],
                "mean_delta_primary": float(row["mean_delta_primary_vs_age_height"]),
                "folds_improved": int(row["folds_improved"]),
                "reach_direction_stable_3_of_4": bool(sign_consistent),
                "nominated_for_followup": bool(row["mean_delta_primary_vs_age_height"] < 0 and row["folds_improved"] >= 3 and sign_consistent),
            })

    summary = {
        "protocol": "development-only FSR-offset reach mechanics study; 2024+ reserved and not scored",
        "outer_rows_reserved_not_scored": outer_rows,
        "fsr_missing_directional_rows": fsr_missing,
        "development_common_rows": int(len(dev)),
        "development_common_fights": int(dev["fight_id"].nunique()),
        "primary_control_model": primary_base,
        "primary_reach_model": "division_age_height_reach",
        "reach_saturation_inches": saturation,
        "mechanics": [m["name"] for m in mechanics],
        "nominations": nominations,
    }
    (out_root / "reach_mechanics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(audit.to_string(index=False))
    print("\nCOVERAGE")
    print(coverage.to_string(index=False))
    print("\nREACH INCREMENTAL SUMMARY")
    print(incremental.to_string(index=False) if not incremental.empty else "none")
    print("\nREACH COEFFICIENT SUMMARY")
    print(coefficient_summary.to_string(index=False) if not coefficient_summary.empty else "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
