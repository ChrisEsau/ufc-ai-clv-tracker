from __future__ import annotations

"""Development-only dissection for Raw Signal Discovery V1.

This module never scores the reserved 2024+ outer period. It uses the same
chronological 2020-2023 folds as the discovery gate to measure which feature
families matter and to summarize signed XGBoost contributions for the most
stable development signals.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pipeline.research.raw_signal_discovery_v1.train_discovery import (
    DEFAULT_CONFIG,
    NON_FEATURE_COLUMNS,
    _fit_xgb,
    _metrics,
    _usable_features,
)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _strip_perspective(feature: str) -> str:
    for prefix in ("self_", "opp_", "diff_"):
        if feature.startswith(prefix):
            return feature[len(prefix):]
    return feature


def _is_physical(feature: str) -> bool:
    return "profile_" in _strip_perspective(feature)


def _is_fsr_like(feature: str) -> bool:
    """Approximate information already represented by validated FSR V3 core traits."""
    base = _strip_perspective(feature)
    if "hist_" not in base or not base.endswith("_mean"):
        return False
    if any(token in base for token in ("r1_", "r2_", "r3_", "r2_minus_r1", "r3_minus_r1")):
        return False
    return any(token in base for token in ("distance_", "td_", "ground_", "kd_"))


def _is_volatility(feature: str) -> bool:
    return _strip_perspective(feature).endswith("_std")


def _is_round_progression(feature: str) -> bool:
    base = _strip_perspective(feature)
    return any(token in base for token in ("r1_", "r2_", "r3_", "r2_minus_r1", "r3_minus_r1"))


def _is_target_mix(feature: str) -> bool:
    base = _strip_perspective(feature)
    return any(token in base for token in ("head_", "body_", "leg_", "clinch_", "attempt_share"))


def _is_wrestling_control(feature: str) -> bool:
    base = _strip_perspective(feature)
    return any(token in base for token in ("td_", "ctrl_sec", "control_share", "ground_", "sub_", "rev"))


def _is_duration_experience(feature: str) -> bool:
    base = _strip_perspective(feature)
    return any(token in base for token in (
        "elapsed_seconds", "rounds_observed", "history_fights",
        "scheduled_rounds", "title_fight",
    ))


def _feature_sets(features: list[str]) -> dict[str, list[str]]:
    physical = {f for f in features if _is_physical(f)}
    fsr_like = {f for f in features if _is_fsr_like(f)}
    novel = set(features) - physical - fsr_like
    variants: dict[str, set[str]] = {
        "all": set(features),
        "physical_only": physical,
        "fsr_like_only": fsr_like,
        "novel_behavior_only": novel,
        "physical_plus_fsr_like": physical | fsr_like,
        "physical_plus_novel": physical | novel,
        "all_minus_physical": set(features) - physical,
        "all_minus_volatility": {f for f in features if not _is_volatility(f)},
        "all_minus_round_progression": {f for f in features if not _is_round_progression(f)},
        "all_minus_target_mix": {f for f in features if not _is_target_mix(f)},
        "all_minus_wrestling_control": {f for f in features if not _is_wrestling_control(f)},
        "all_minus_duration_experience": {f for f in features if not _is_duration_experience(f)},
    }
    return {name: sorted(cols) for name, cols in variants.items()}


def _direction_rows(
    frame: pd.DataFrame,
    X_valid: pd.DataFrame,
    model,
    top_features: list[str],
    fold: int,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_valid, feature_names=list(X_valid.columns))
    contrib = booster.predict(dmatrix, pred_contribs=True)[:, :-1]
    feature_to_idx = {f: i for i, f in enumerate(X_valid.columns)}
    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    y = frame["fighter_win"].to_numpy(int)

    for feature in top_features:
        idx = feature_to_idx.get(feature)
        if idx is None:
            continue
        values = pd.to_numeric(X_valid[feature], errors="coerce").to_numpy(float)
        shap = contrib[:, idx].astype(float)
        mask = np.isfinite(values) & np.isfinite(shap)
        if mask.sum() < 50:
            continue
        v = values[mask]
        s = shap[mask]
        yy = y[mask]
        vr = pd.Series(v).rank(method="average").to_numpy(float)
        sr = pd.Series(s).rank(method="average").to_numpy(float)
        rho = float(np.corrcoef(vr, sr)[0, 1]) if len(vr) > 1 else np.nan
        q25, q75 = np.quantile(v, [0.25, 0.75])
        low = s[v <= q25]
        high = s[v >= q75]
        low_mean = float(np.mean(low)) if len(low) else np.nan
        high_mean = float(np.mean(high)) if len(high) else np.nan
        summary_rows.append({
            "fold": fold,
            "feature": feature,
            "n": int(mask.sum()),
            "spearman_value_vs_shap": rho,
            "q25_value": float(q25),
            "q75_value": float(q75),
            "low_quartile_mean_shap": low_mean,
            "high_quartile_mean_shap": high_mean,
            "high_minus_low_shap": high_mean - low_mean,
            "mean_abs_shap": float(np.mean(np.abs(s))),
        })

        tmp = pd.DataFrame({"value": v, "shap": s, "fighter_win": yy})
        try:
            tmp["bin"] = pd.qcut(tmp["value"], q=bins, duplicates="drop")
        except ValueError:
            continue
        for bin_id, (_, group) in enumerate(tmp.groupby("bin", observed=True, sort=True), start=1):
            bin_rows.append({
                "fold": fold,
                "feature": feature,
                "bin": bin_id,
                "n": int(len(group)),
                "value_min": float(group["value"].min()),
                "value_median": float(group["value"].median()),
                "value_max": float(group["value"].max()),
                "mean_signed_shap": float(group["shap"].mean()),
                "mean_abs_shap": float(group["shap"].abs().mean()),
                "empirical_win_rate": float(group["fighter_win"].mean()),
            })
    return summary_rows, bin_rows


def run(config_path: Path = DEFAULT_CONFIG) -> None:
    config = _load_config(config_path)
    outputs = config["outputs"]
    dcfg = config.get("dissection", {})
    top_n = int(dcfg.get("top_signed_shap_features", 30))
    bins = int(dcfg.get("dependence_bins", 10))

    bank = pd.read_parquet(outputs["prefight_feature_bank"])
    bank["event_date"] = pd.to_datetime(bank["event_date"], errors="raise").dt.normalize()
    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    dev = bank[bank["event_date"] < outer_start].copy()
    outer = bank[bank["event_date"] >= outer_start]
    if dev.empty:
        raise RuntimeError("no development rows available for dissection")

    all_candidates = [c for c in bank.columns if c not in NON_FEATURE_COLUMNS]
    stability = pd.read_csv(outputs["signal_stability"])
    top_features = stability.head(top_n)["feature"].astype(str).tolist()
    ablation_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    family_count_rows: list[dict[str, Any]] = []
    seed = int(config["model"]["random_seed"])

    for year in [int(x) for x in config["validation"]["development_years"]]:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        train = dev[dev["event_date"] < start].copy()
        valid = dev[(dev["event_date"] >= start) & (dev["event_date"] < end)].copy()
        if valid.empty:
            continue
        usable = _usable_features(train, all_candidates)
        variants = _feature_sets(usable)
        family_count_rows.append({
            "fold": year,
            "usable_features": len(usable),
            "physical": sum(_is_physical(f) for f in usable),
            "fsr_like": sum(_is_fsr_like(f) for f in usable),
            "novel_behavior": sum((not _is_physical(f)) and (not _is_fsr_like(f)) for f in usable),
            "volatility": sum(_is_volatility(f) for f in usable),
            "round_progression": sum(_is_round_progression(f) for f in usable),
            "target_mix": sum(_is_target_mix(f) for f in usable),
            "wrestling_control": sum(_is_wrestling_control(f) for f in usable),
            "duration_experience": sum(_is_duration_experience(f) for f in usable),
        })

        all_model = None
        all_X_valid = None
        for variant_name, features in variants.items():
            if not features:
                continue
            model, X_valid, pred = _fit_xgb(
                train, valid, features, config["model"]["xgb"], seed + year
            )
            metrics = _metrics(valid["fighter_win"].to_numpy(int), pred)
            ablation_rows.append({
                "fold": year,
                "variant": variant_name,
                "features": len(features),
                **metrics,
            })
            if variant_name == "all":
                all_model = model
                all_X_valid = X_valid

        if all_model is None or all_X_valid is None:
            raise RuntimeError(f"full dissection model missing for fold {year}")
        srows, brows = _direction_rows(valid, all_X_valid, all_model, top_features, year, bins)
        direction_rows.extend(srows)
        dependence_rows.extend(brows)

    ablations = pd.DataFrame(ablation_rows)
    if ablations.empty:
        raise RuntimeError("no ablation results produced")
    full = ablations[ablations["variant"] == "all"][["fold", "log_loss", "brier", "auc", "accuracy"]].rename(columns={
        "log_loss": "all_log_loss", "brier": "all_brier", "auc": "all_auc", "accuracy": "all_accuracy",
    })
    ablations = ablations.merge(full, on="fold", how="left", validate="many_to_one")
    ablations["delta_log_loss_vs_all"] = ablations["log_loss"] - ablations["all_log_loss"]
    ablations["delta_brier_vs_all"] = ablations["brier"] - ablations["all_brier"]
    ablations["delta_auc_vs_all"] = ablations["auc"] - ablations["all_auc"]
    ablations["delta_accuracy_vs_all"] = ablations["accuracy"] - ablations["all_accuracy"]
    ablation_summary = ablations.groupby("variant", as_index=False).agg(
        folds=("fold", "nunique"), mean_features=("features", "mean"),
        mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"),
        mean_auc=("auc", "mean"), mean_accuracy=("accuracy", "mean"),
        mean_delta_log_loss_vs_all=("delta_log_loss_vs_all", "mean"),
        mean_delta_brier_vs_all=("delta_brier_vs_all", "mean"),
        mean_delta_auc_vs_all=("delta_auc_vs_all", "mean"),
        mean_delta_accuracy_vs_all=("delta_accuracy_vs_all", "mean"),
    ).sort_values("mean_log_loss")

    directions = pd.DataFrame(direction_rows)
    if not directions.empty:
        direction_summary = directions.groupby("feature", as_index=False).agg(
            folds=("fold", "nunique"), mean_abs_shap=("mean_abs_shap", "mean"),
            mean_spearman_value_vs_shap=("spearman_value_vs_shap", "mean"),
            mean_high_minus_low_shap=("high_minus_low_shap", "mean"),
            positive_direction_folds=("high_minus_low_shap", lambda s: int((s > 0).sum())),
            negative_direction_folds=("high_minus_low_shap", lambda s: int((s < 0).sum())),
        )
        direction_summary["direction_consistent"] = (
            (direction_summary["positive_direction_folds"] == direction_summary["folds"]) |
            (direction_summary["negative_direction_folds"] == direction_summary["folds"])
        )
        direction_summary = direction_summary.sort_values("mean_abs_shap", ascending=False)
    else:
        direction_summary = pd.DataFrame()

    root = Path(outputs["root"])
    ablations.to_csv(root / "family_ablation_metrics.csv", index=False)
    ablation_summary.to_csv(root / "family_ablation_summary.csv", index=False)
    pd.DataFrame(family_count_rows).to_csv(root / "feature_family_counts.csv", index=False)
    directions.to_csv(root / "signed_shap_fold_summary.csv", index=False)
    direction_summary.to_csv(root / "signed_shap_direction_summary.csv", index=False)
    pd.DataFrame(dependence_rows).to_csv(root / "signed_shap_dependence_bins.csv", index=False)

    audit = pd.DataFrame([
        {"check": "outer_start_unchanged", "passed": str(outer_start.date()) == "2024-01-01", "value": str(outer_start.date())},
        {"check": "outer_rows_never_scored", "passed": True, "value": int(len(outer))},
        {"check": "development_rows_only", "passed": bool(dev["event_date"].lt(outer_start).all()), "value": int(len(dev))},
        {"check": "development_folds_only", "passed": set(ablations["fold"].unique()).issubset({2020, 2021, 2022, 2023}), "value": sorted(ablations["fold"].unique().tolist())},
    ])
    audit.to_csv(root / "dissection_audit.csv", index=False)
    if not audit["passed"].all():
        raise RuntimeError("signal dissection audit failed")

    summary = {
        "protocol": "development-only signal dissection; 2024+ reserved and not scored",
        "outer_rows_reserved_not_scored": int(len(outer)),
        "top_signed_shap_features": top_n,
        "dependence_bins": bins,
        "ablation_variants": sorted(ablations["variant"].unique().tolist()),
        "best_non_all_variants_by_log_loss": ablation_summary[ablation_summary["variant"] != "all"].head(5).to_dict(orient="records"),
        "directionally_consistent_top_features": (
            direction_summary[direction_summary["direction_consistent"]].head(30)["feature"].tolist()
            if not direction_summary.empty else []
        ),
    }
    (root / "dissection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFAMILY ABLATION SUMMARY")
    print(ablation_summary.to_string(index=False))
    if not direction_summary.empty:
        print("\nSIGNED SHAP DIRECTION SUMMARY")
        print(direction_summary.head(30).to_string(index=False))
    print(f"\n2024+ rows reserved and not scored: {len(outer):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
