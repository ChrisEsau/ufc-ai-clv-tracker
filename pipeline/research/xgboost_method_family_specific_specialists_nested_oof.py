from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.xgboost_method_family_specific_specialists_oof import (
    ROOT, FAMILIES, FEATURE_COUNTS, CAPACITY, FEATURE_LIST_PATH,
    load_features, rank_family_features, family_candidate_oof, fit_family_fold,
    combine_family_probs, _metrics, _calibration, _build_rows, MARKET_COLS,
    CLASS_ORDER, EPS,
)

OUT_PRED = ROOT / "xgboost_method_market_offset__family_specific_specialists_nested_oof_predictions.csv"
OUT_SUMMARY = ROOT / "xgboost_method_market_offset__family_specific_specialists_nested_oof_summary.json"
OUT_SELECTIONS = ROOT / "xgboost_method_market_offset__family_specific_specialists_nested_selections.csv"

TEST_YEARS = [2022, 2023, 2024]
FROZEN_SIXWAY_PRED = ROOT / "xgboost_method_market_offset__oof_predictions.csv"


def select_configs(selection_years: list[int], df: pd.DataFrame, ranked: dict[str, list[str]]) -> dict:
    selected = {}
    for family in FAMILIES:
        rows = []
        for nfeat in FEATURE_COUNTS:
            feats = ranked[family][: min(nfeat, len(ranked[family]))]
            for cap in CAPACITY:
                result = family_candidate_oof(
                    family=family,
                    df=df,
                    features=feats,
                    max_depth=cap["max_depth"],
                    rounds=cap["rounds"],
                    validation_years=selection_years,
                )
                rows.append({
                    "family": family,
                    "feature_count_requested": nfeat,
                    "max_depth": cap["max_depth"],
                    "rounds": cap["rounds"],
                    "binary_log_loss": result["binary_log_loss"],
                    "market_binary_log_loss": result["market_binary_log_loss"],
                    "delta_vs_market": result["delta_vs_market"],
                    "candidate": f"{family}_top{nfeat}_{cap['name']}",
                })
        selected[family] = sorted(rows, key=lambda r: (r["binary_log_loss"], r["feature_count_requested"], r["max_depth"], r["rounds"]))[0]
    return selected


def main() -> None:
    features = load_features()
    df, _, _ = _build_rows(development_only=True, include_targets=True, forced_features=features)
    ranked = {family: rank_family_features(df, features, family) for family in FAMILIES}

    all_rows = []
    selections = []
    yearly = []

    for test_year in TEST_YEARS:
        selection_years = list(range(2021, test_year))
        selected = select_configs(selection_years, df, ranked)
        for family, cfg in selected.items():
            selections.append({"test_year": test_year, "selection_years": ",".join(map(str, selection_years)), **cfg})

        train_end = pd.Timestamp(f"{test_year-1}-12-31")
        val_start = pd.Timestamp(f"{test_year}-01-01")
        val_end = pd.Timestamp(f"{test_year}-12-31")
        train = df[df["date"] <= train_end].copy()
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()

        fam_probs = {}
        for family, cfg in selected.items():
            feats = ranked[family][: min(int(cfg["feature_count_requested"]), len(ranked[family]))]
            fam_probs[family] = fit_family_fold(
                family=family,
                train=train,
                val=val,
                features=feats,
                max_depth=int(cfg["max_depth"]),
                rounds=int(cfg["rounds"]),
            )

        market = val[MARKET_COLS].to_numpy(float)
        combined = combine_family_probs(market, fam_probs)
        y = val["target"].to_numpy(int)
        yearly.append({
            "year": test_year,
            "n": len(val),
            "market": _metrics(y, market),
            "nested": _metrics(y, combined),
        })

        for i, (_, r) in enumerate(val.iterrows()):
            out = {
                "fight_id": r["fight_id"],
                "date": r["date"].date().isoformat(),
                "event_name": r["event_name"],
                "red_fighter": r["red_fighter"],
                "blue_fighter": r["blue_fighter"],
                "target": int(r["target"]),
                "actual_class": CLASS_ORDER[int(r["target"])],
                "test_year": test_year,
            }
            for j, cname in enumerate(CLASS_ORDER):
                slug = cname.lower()
                out[f"market_{slug}"] = float(market[i, j])
                out[f"nested_{slug}"] = float(combined[i, j])
            all_rows.append(out)

    pred = pd.DataFrame(all_rows).sort_values(["date", "fight_id"]).reset_index(drop=True)
    y = pred["target"].to_numpy(int)
    market_cols = [f"market_{c.lower()}" for c in CLASS_ORDER]
    nested_cols = [f"nested_{c.lower()}" for c in CLASS_ORDER]
    market = pred[market_cols].to_numpy(float)
    nested = pred[nested_cols].to_numpy(float)

    frozen = pd.read_csv(FROZEN_SIXWAY_PRED)
    frozen = frozen[frozen["fold"].astype(int).isin(TEST_YEARS)].copy()
    frozen = frozen.sort_values(["date", "fight_id"]).reset_index(drop=True)
    if list(frozen["fight_id"].astype(str)) != list(pred["fight_id"].astype(str)):
        raise RuntimeError("frozen six-way comparison rows do not align with nested predictions")
    frozen_cols = [f"model_{c.lower().replace('_tko','')}" for c in CLASS_ORDER]
    # existing ledger uses red_ko/red_sub/red_dec/blue_ko/blue_sub/blue_dec
    frozen_cols = ["model_red_ko","model_red_sub","model_red_dec","model_blue_ko","model_blue_sub","model_blue_dec"]
    frozen_p = frozen[frozen_cols].to_numpy(float)

    summary = {
        "experiment": "six_way_method_family_specific_specialists_nested_oof_v1",
        "period": "nested chronological 2022-2024 OOF",
        "reads_2025_plus": False,
        "uses_roi": False,
        "selection_rule": "for each test year, select KO/SUB/DEC configs only from earlier OOF years",
        "test_years": TEST_YEARS,
        "pooled_n": int(len(pred)),
        "pooled_market": _metrics(y, market),
        "pooled_nested": _metrics(y, nested),
        "pooled_frozen_sixway_same_rows": _metrics(y, frozen_p),
        "delta_nested_minus_market_log_loss": float(_metrics(y, nested)["log_loss"] - _metrics(y, market)["log_loss"]),
        "delta_nested_minus_frozen_sixway_log_loss": float(_metrics(y, nested)["log_loss"] - _metrics(y, frozen_p)["log_loss"]),
        "beats_frozen_sixway_same_rows": bool(_metrics(y, nested)["log_loss"] < _metrics(y, frozen_p)["log_loss"]),
        "yearly": yearly,
        "calibration": _calibration(y, nested),
    }

    pd.DataFrame(selections).to_csv(OUT_SELECTIONS, index=False)
    pred.to_csv(OUT_PRED, index=False)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
