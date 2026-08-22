from __future__ import annotations

"""Robustness checks for Reach Mechanics V1.

Research-only. Reconstructs the same pre-2024 temporary FSR V3 state used by
Reach Mechanics V1, then tests the two nominated reach mechanics against:
- clipped reach edges;
- removal of large/extreme reach matchups;
- within-division chronological fits.

No FSR or simulator files are modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.research.reach_mechanics_v1.entrypoint import build_temporary_prefight
from pipeline.research.reach_mechanics_v1.run import (
    DEFAULT_CONFIG,
    _aggregate_actuals,
    _attach_fsr_expectations,
    _attach_physical,
    _fit_variant,
    _master_meta,
    _mechanics,
    _valid_for_mechanic,
)

OUT = Path("data/research/reach_mechanics_robustness_v1")
TMP_FSR = Path("/tmp/reach_mechanics_robustness_v1_fsr.parquet")
SHORTLIST = {"distance_attempt_rate", "td_completion"}
YEARS = [2020, 2021, 2022, 2023]


def _metric_delta(train: pd.DataFrame, valid: pd.DataFrame, spec: dict, reach_term: str):
    base, _, base_score = _fit_variant(
        train, valid, spec, ["age_edge", "height_edge"], True
    )
    model, _, reach_score = _fit_variant(
        train, valid, spec, ["age_edge", "height_edge", reach_term], True
    )
    beta = float(model.params[reach_term])
    return {
        "base_score": float(base_score["primary_score"]),
        "reach_score": float(reach_score["primary_score"]),
        "delta": float(reach_score["primary_score"] - base_score["primary_score"]),
        "beta": beta,
    }


def _prepare_frame(config: dict) -> tuple[pd.DataFrame, int]:
    master = pd.read_parquet(config["inputs"]["master_path"])
    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    temporary = build_temporary_prefight(master, outer_start)
    temporary.to_parquet(TMP_FSR, index=False)

    actual = _aggregate_actuals(master)
    saturation = float(config["physical"]["reach_saturation_inches"])
    frame = _attach_physical(actual, _master_meta(master), saturation)
    frame, fsr_missing = _attach_fsr_expectations(frame, TMP_FSR)
    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.normalize()
    frame["physical_complete"] = frame[
        ["self_reach", "opp_reach", "self_height", "opp_height", "self_age", "opp_age"]
    ].notna().all(axis=1)
    dev = frame[
        frame["event_date"].lt(outer_start)
        & frame["fsr_matched"].astype(bool)
        & frame["physical_complete"].astype(bool)
    ].copy()
    dev["reach_clip4"] = dev["reach_edge"].clip(-4.0, 4.0)
    dev["reach_clip6"] = dev["reach_edge"].clip(-6.0, 6.0)
    return dev, fsr_missing


def _pooled_scenarios(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [s for s in _mechanics() if s["name"] in SHORTLIST]
    for spec in specs:
        cohort = _valid_for_mechanic(dev, spec)
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01")
            end = pd.Timestamp(f"{year + 1}-01-01")
            base_train = cohort[cohort["event_date"].lt(start)].copy()
            base_valid = cohort[
                cohort["event_date"].ge(start) & cohort["event_date"].lt(end)
            ].copy()
            if len(base_train) < 1000 or len(base_valid) < 150:
                continue

            scenarios = [
                ("full_linear", base_train, base_valid, "reach_edge", np.nan),
                ("clip4", base_train, base_valid, "reach_clip4", 4.0),
                ("clip6", base_train, base_valid, "reach_clip6", 6.0),
            ]
            for threshold in (4.0, 6.0):
                tr = base_train[base_train["reach_edge"].abs().le(threshold)].copy()
                va = base_valid[base_valid["reach_edge"].abs().le(threshold)].copy()
                scenarios.append((f"trim_abs{int(threshold)}", tr, va, "reach_edge", threshold))

            p95 = float(base_train["reach_edge"].abs().quantile(0.95))
            tr95 = base_train[base_train["reach_edge"].abs().le(p95)].copy()
            va95 = base_valid[base_valid["reach_edge"].abs().le(p95)].copy()
            scenarios.append(("trim_train_p95", tr95, va95, "reach_edge", p95))

            for scenario, train, valid, term, threshold in scenarios:
                if len(train) < 800 or len(valid) < 100:
                    continue
                try:
                    result = _metric_delta(train, valid, spec, term)
                    rows.append({
                        "mechanic": spec["name"],
                        "fold": year,
                        "scenario": scenario,
                        "reach_term": term,
                        "threshold_inches": threshold,
                        "train_rows": len(train),
                        "valid_rows": len(valid),
                        **result,
                    })
                except Exception as exc:
                    rows.append({
                        "mechanic": spec["name"],
                        "fold": year,
                        "scenario": scenario,
                        "reach_term": term,
                        "threshold_inches": threshold,
                        "train_rows": len(train),
                        "valid_rows": len(valid),
                        "base_score": np.nan,
                        "reach_score": np.nan,
                        "delta": np.nan,
                        "beta": np.nan,
                        "error": type(exc).__name__,
                    })
    return pd.DataFrame(rows)


def _scenario_summary(folds: pd.DataFrame) -> pd.DataFrame:
    valid = folds[np.isfinite(folds["delta"]) & np.isfinite(folds["beta"])].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["mechanic", "scenario"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            folds_improved=("delta", lambda s: int((s < 0).sum())),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            positive_beta_folds=("beta", lambda s: int((s > 0).sum())),
            negative_beta_folds=("beta", lambda s: int((s < 0).sum())),
            mean_beta=("beta", "mean"),
            median_beta=("beta", "median"),
            mean_valid_rows=("valid_rows", "mean"),
        )
    )


def _division_folds(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [s for s in _mechanics() if s["name"] in SHORTLIST]
    for spec in specs:
        cohort = _valid_for_mechanic(dev, spec)
        divisions = sorted(cohort["division"].dropna().astype(str).unique())
        for division in divisions:
            div = cohort[cohort["division"].astype(str).eq(division)].copy()
            for year in YEARS:
                start = pd.Timestamp(f"{year}-01-01")
                end = pd.Timestamp(f"{year + 1}-01-01")
                train = div[div["event_date"].lt(start)].copy()
                valid = div[div["event_date"].ge(start) & div["event_date"].lt(end)].copy()
                if len(train) < 200 or len(valid) < 30:
                    continue
                try:
                    _, _, base_score = _fit_variant(
                        train, valid, spec, ["age_edge", "height_edge"], False
                    )
                    model, _, reach_score = _fit_variant(
                        train, valid, spec,
                        ["age_edge", "height_edge", "reach_edge"], False
                    )
                    rows.append({
                        "mechanic": spec["name"],
                        "division": division,
                        "fold": year,
                        "train_rows": len(train),
                        "valid_rows": len(valid),
                        "delta": float(reach_score["primary_score"] - base_score["primary_score"]),
                        "beta": float(model.params["reach_edge"]),
                    })
                except Exception as exc:
                    rows.append({
                        "mechanic": spec["name"], "division": division, "fold": year,
                        "train_rows": len(train), "valid_rows": len(valid),
                        "delta": np.nan, "beta": np.nan, "error": type(exc).__name__,
                    })
    return pd.DataFrame(rows)


def _division_summary(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    valid = folds[np.isfinite(folds["delta"]) & np.isfinite(folds["beta"])].copy()
    return (
        valid.groupby(["mechanic", "division"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            folds_improved=("delta", lambda s: int((s < 0).sum())),
            mean_delta=("delta", "mean"),
            positive_beta_folds=("beta", lambda s: int((s > 0).sum())),
            negative_beta_folds=("beta", lambda s: int((s < 0).sum())),
            mean_beta=("beta", "mean"),
            total_valid_rows=("valid_rows", "sum"),
        )
    )


def _extreme_fighters(dev: pd.DataFrame) -> pd.DataFrame:
    extreme = dev[dev["reach_edge"].abs().ge(6.0)].copy()
    if extreme.empty:
        return pd.DataFrame()
    out = (
        extreme.groupby(["fighter_id", "fighter_name"], as_index=False)
        .agg(
            extreme_rows=("fight_id", "size"),
            extreme_fights=("fight_id", "nunique"),
            mean_abs_reach_edge=("reach_edge", lambda s: float(s.abs().mean())),
            max_abs_reach_edge=("reach_edge", lambda s: float(s.abs().max())),
        )
        .sort_values(["extreme_fights", "max_abs_reach_edge"], ascending=[False, False])
        .reset_index(drop=True)
    )
    out["share_of_extreme_rows"] = out["extreme_rows"] / max(len(extreme), 1)
    return out


def main() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    dev, fsr_missing = _prepare_frame(config)
    OUT.mkdir(parents=True, exist_ok=True)

    folds = _pooled_scenarios(dev)
    summary = _scenario_summary(folds)
    div_folds = _division_folds(dev)
    div_summary = _division_summary(div_folds)
    extremes = _extreme_fighters(dev)

    pair_sums = dev.groupby("fight_id")["reach_edge"].sum(min_count=2)
    audit = pd.DataFrame([
        {"check": "outer_start_unchanged", "passed": str(outer_start.date()) == "2024-01-01", "value": str(outer_start.date())},
        {"check": "development_rows_only", "passed": bool(dev["event_date"].lt(outer_start).all()), "value": len(dev)},
        {"check": "fsr_rows_resolved", "passed": bool(dev["fsr_matched"].all()), "value": len(dev)},
        {"check": "reach_edges_antisymmetric", "passed": bool(np.allclose(pair_sums.dropna(), 0.0, atol=1e-8)), "value": float(pair_sums.dropna().abs().max())},
        {"check": "shortlist_only", "passed": set(folds["mechanic"].dropna()).issubset(SHORTLIST), "value": sorted(folds["mechanic"].dropna().unique().tolist())},
        {"check": "chronological_folds_only", "passed": set(folds["fold"].dropna().astype(int)).issubset(set(YEARS)), "value": sorted(folds["fold"].dropna().unique().tolist())},
    ])
    if not audit["passed"].all():
        raise RuntimeError("reach robustness audit failed")

    gate_rows = []
    for mechanic in sorted(SHORTLIST):
        s = summary[summary["mechanic"].eq(mechanic)].set_index("scenario")
        required = ["full_linear", "clip4", "clip6", "trim_abs6", "trim_train_p95"]
        available = [x for x in required if x in s.index]
        scenario_passes = 0
        for scenario in available:
            r = s.loc[scenario]
            if int(r["folds_improved"]) >= 3 and float(r["mean_delta"]) < 0 and int(r["positive_beta_folds"]) >= 3:
                scenario_passes += 1
        d = div_summary[(div_summary["mechanic"].eq(mechanic)) & (div_summary["folds"].ge(2))]
        positive_divisions = int((d["mean_beta"] > 0).sum()) if not d.empty else 0
        evaluable_divisions = int(len(d))
        improving_divisions = int((d["mean_delta"] < 0).sum()) if not d.empty else 0
        gate_rows.append({
            "mechanic": mechanic,
            "robust_scenarios_available": len(available),
            "robust_scenarios_passed": scenario_passes,
            "evaluable_divisions": evaluable_divisions,
            "positive_beta_divisions": positive_divisions,
            "improving_divisions": improving_divisions,
            "robustness_pass": bool(
                len(available) >= 5
                and scenario_passes >= 4
                and evaluable_divisions > 0
                and positive_divisions / evaluable_divisions >= 0.60
            ),
        })
    gates = pd.DataFrame(gate_rows)

    folds.to_csv(OUT / "robustness_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "robustness_scenario_summary.csv", index=False)
    div_folds.to_csv(OUT / "division_fold_metrics.csv", index=False)
    div_summary.to_csv(OUT / "division_summary.csv", index=False)
    extremes.to_csv(OUT / "extreme_reach_fighters.csv", index=False)
    gates.to_csv(OUT / "robustness_gate.csv", index=False)
    audit.to_csv(OUT / "audit.csv", index=False)

    payload = {
        "protocol": "development-only reach robustness; 2024+ sealed",
        "fsr_missing_directional_rows_outside_reconstructed_period": int(fsr_missing),
        "development_rows": int(len(dev)),
        "development_fights": int(dev["fight_id"].nunique()),
        "shortlist": sorted(SHORTLIST),
        "gates": gates.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("AUDIT")
    print(audit.to_string(index=False))
    print("\nSCENARIO SUMMARY")
    print(summary.to_string(index=False))
    print("\nDIVISION SUMMARY")
    print(div_summary.to_string(index=False))
    print("\nROBUSTNESS GATE")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
