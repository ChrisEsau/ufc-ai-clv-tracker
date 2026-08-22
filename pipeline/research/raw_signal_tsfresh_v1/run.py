from __future__ import annotations

"""Development-only tsfresh challenger for UFC raw-signal discovery.

Every target fighter-fight sequence contains only that fighter's fights from
strictly earlier calendar dates. The reserved 2024+ period is counted but never
extracted, fit, selected, or scored.
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tsfresh import extract_features

from pipeline.research.raw_signal_discovery_v1.build_feature_bank import (
    _build_directional_matchups,
    _build_fight_observations,
    _build_prefight_snapshots,
)
from pipeline.research.raw_signal_discovery_v1.dissect_signals import (
    _is_duration_experience,
    _is_physical,
    _is_target_mix,
)
from pipeline.research.raw_signal_discovery_v1.train_discovery import (
    NON_FEATURE_COLUMNS,
    _fit_xgb,
    _fold_importance,
    _metrics,
    _usable_features,
)

DEFAULT_CONFIG = Path("pipeline/research/raw_signal_tsfresh_v1/config.yaml")


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("tsfresh research config must be a mapping")
    return payload


def _eligible_master(master: pd.DataFrame) -> pd.DataFrame:
    required = {"fight_id", "date", "winner_id", "r_id", "b_id"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"master missing required tsfresh columns: {missing}")
    m = master.copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["event_date"] = pd.to_datetime(m["date"], errors="raise").dt.normalize()
    m = m[m["winner_id"].notna()].copy()
    valid_ids = m.apply(
        lambda x: str(x["winner_id"]) in {str(x["r_id"]), str(x["b_id"])},
        axis=1,
    )
    return m[valid_ids].sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def _fc_parameters() -> dict[str, Any]:
    """Bounded automatic sequence search: richer than mean/std, smaller than full tsfresh."""
    return {
        "mean": None,
        "median": None,
        "minimum": None,
        "maximum": None,
        "standard_deviation": None,
        "variance": None,
        "root_mean_square": None,
        "abs_energy": None,
        "absolute_sum_of_changes": None,
        "mean_abs_change": None,
        "mean_change": None,
        "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
        "c3": [{"lag": 1}, {"lag": 2}],
        "cid_ce": [{"normalize": False}, {"normalize": True}],
        "count_above_mean": None,
        "count_below_mean": None,
        "first_location_of_maximum": None,
        "first_location_of_minimum": None,
        "last_location_of_maximum": None,
        "last_location_of_minimum": None,
        "linear_trend": [
            {"attr": "slope"}, {"attr": "rvalue"}, {"attr": "stderr"},
        ],
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "quantile": [{"q": 0.1}, {"q": 0.25}, {"q": 0.75}, {"q": 0.9}],
        "ratio_beyond_r_sigma": [{"r": 1.0}, {"r": 2.0}],
        "skewness": None,
        "kurtosis": None,
        "time_reversal_asymmetry_statistic": [{"lag": 1}, {"lag": 2}],
        "binned_entropy": [{"max_bins": 5}],
        "number_peaks": [{"n": 1}, {"n": 2}],
    }


def _safe_feature_name(raw_name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw_name)).strip("_")
    return f"tsf_{clean}"


def _build_long_sequences(
    targets: pd.DataFrame,
    obs: pd.DataFrame,
    metrics: list[str],
    max_history_fights: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(metrics) - set(obs.columns))
    if missing:
        raise ValueError(f"fight observations missing configured tsfresh metrics: {missing}")

    grouped = {
        str(fid): g.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
        for fid, g in obs.groupby("fighter_id", sort=False)
    }
    chunks: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for row in targets.itertuples(index=False):
        fighter_id = str(row.fighter_id)
        target_date = pd.Timestamp(row.event_date)
        target_id = str(row.target_id)
        hist = grouped.get(fighter_id)
        if hist is None:
            hist = pd.DataFrame(columns=obs.columns)
        else:
            hist = hist[pd.to_datetime(hist["event_date"]).lt(target_date)]
        if max_history_fights > 0:
            hist = hist.tail(max_history_fights)

        history_max = (
            pd.Timestamp(hist["event_date"].max()).normalize()
            if not hist.empty else pd.NaT
        )
        coverage_rows.append({
            "target_id": target_id,
            "fight_id": str(row.fight_id),
            "fighter_id": fighter_id,
            "event_date": target_date,
            "history_fights_used": int(len(hist)),
            "history_max_date": history_max,
        })
        if hist.empty:
            continue

        values = hist[metrics].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        values.insert(0, "sequence_pos", np.arange(len(values), dtype=int))
        values.insert(0, "target_id", target_id)
        chunks.append(values)

    long = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["target_id", "sequence_pos", *metrics])
    )
    return long, pd.DataFrame(coverage_rows)


def _extract_tsfresh(
    long: pd.DataFrame,
    all_target_ids: pd.Index,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if long.empty:
        raise RuntimeError("no historical sequences available for tsfresh extraction")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extracted = extract_features(
            long,
            column_id="target_id",
            column_sort="sequence_pos",
            default_fc_parameters=_fc_parameters(),
            n_jobs=n_jobs,
            disable_progressbar=True,
            show_warnings=False,
            impute_function=None,
        )

    extracted = extracted.replace([np.inf, -np.inf], np.nan)
    raw_names = [str(c) for c in extracted.columns]
    safe_names = [_safe_feature_name(c) for c in raw_names]
    if len(set(safe_names)) != len(safe_names):
        duplicates = pd.Series(safe_names).value_counts()
        duplicates = duplicates[duplicates.gt(1)].index.tolist()
        raise RuntimeError(f"tsfresh feature-name sanitization collision: {duplicates[:20]}")

    manifest_rows = []
    for raw, safe in zip(raw_names, safe_names):
        parts = raw.split("__")
        manifest_rows.append({
            "feature": safe,
            "raw_feature": raw,
            "source_metric": parts[0] if parts else raw,
            "calculator": parts[1] if len(parts) > 1 else "unknown",
        })
    extracted.columns = safe_names
    extracted = extracted.reindex(all_target_ids)
    extracted.index.name = "target_id"
    return extracted, pd.DataFrame(manifest_rows)


def _directional_tsfresh(targets: pd.DataFrame, extracted: pd.DataFrame) -> pd.DataFrame:
    base_features = list(extracted.columns)
    own = targets[["fight_id", "fighter_id", "opponent_id", "target_id"]].merge(
        extracted.reset_index(), on="target_id", how="left", validate="one_to_one"
    )

    self_frame = own[["fight_id", "fighter_id", "opponent_id"]].reset_index(drop=True)
    self_values = own[base_features].copy()
    self_values.columns = [f"self_{c}" for c in base_features]
    self_frame = pd.concat([self_frame, self_values.reset_index(drop=True)], axis=1)

    opponent_values = own[["fight_id", "fighter_id", *base_features]].copy()
    opponent_values = opponent_values.rename(columns={"fighter_id": "opponent_id"})
    opponent_values = opponent_values.rename(columns={c: f"opp_{c}" for c in base_features})
    paired = self_frame.merge(
        opponent_values,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    diff_payload = {}
    for c in base_features:
        a = pd.to_numeric(paired[f"self_{c}"], errors="coerce")
        b = pd.to_numeric(paired[f"opp_{c}"], errors="coerce")
        diff_payload[f"diff_{c}"] = a - b
    return pd.concat([paired, pd.DataFrame(diff_payload, index=paired.index)], axis=1)


def _stability(
    importance: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if importance.empty:
        return pd.DataFrame(), pd.DataFrame()

    imp = importance.copy()
    imp["base_feature"] = imp["feature"].str.replace(
        r"^(self_|opp_|diff_)", "", regex=True
    )
    lookup = manifest.set_index("feature")
    imp["source_metric"] = imp["base_feature"].map(lookup["source_metric"])
    imp["calculator"] = imp["base_feature"].map(lookup["calculator"])

    stability = (
        imp.groupby("feature", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            median_shap_rank=("shap_rank", "median"),
            top100_folds=("shap_rank", lambda s: int((s <= 100).sum())),
        )
        .sort_values(["top100_folds", "mean_abs_shap"], ascending=[False, False])
    )
    calculator = (
        imp.dropna(subset=["calculator"])
        .groupby("calculator", as_index=False)
        .agg(
            directional_features=("feature", "nunique"),
            folds=("fold", "nunique"),
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            top100_hits=("shap_rank", lambda s: int((s <= 100).sum())),
        )
        .sort_values(["top100_hits", "total_mean_abs_shap"], ascending=[False, False])
    )
    return stability, calculator


def run(config_path: Path = DEFAULT_CONFIG) -> None:
    config = _load(config_path)
    discovery_config = _load(Path(config["inputs"]["discovery_config"]))
    round_path = Path(discovery_config["inputs"]["round_stats_path"])
    master_path = Path(discovery_config["inputs"]["master_path"])
    out_root = Path(config["outputs"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    development_years = [int(x) for x in config["validation"]["development_years"]]
    max_history_fights = int(config["sequence"]["max_history_fights"])
    sequence_metrics = [str(x) for x in config["sequence"]["metrics"]]
    n_jobs = int(config["tsfresh"]["n_jobs"])

    rounds = pd.read_parquet(round_path)
    master = pd.read_parquet(master_path)
    eligible = _eligible_master(master)
    outer_rows = int(2 * eligible["event_date"].ge(outer_start).sum())
    dev_master = eligible[eligible["event_date"].lt(outer_start)].copy()

    obs = _build_fight_observations(rounds, master)
    obs = obs[pd.to_datetime(obs["event_date"]).lt(outer_start)].copy()

    # Manual targets are also rebuilt only through 2023 so the combined challenger
    # can measure incremental tsfresh signal without ever constructing outer targets.
    snapshots = _build_prefight_snapshots(obs, dev_master, discovery_config)
    manual_bank = _build_directional_matchups(snapshots)
    snapshots = snapshots.copy()
    snapshots["target_id"] = (
        snapshots["fight_id"].astype(str) + "|" + snapshots["fighter_id"].astype(str)
    )

    target_cols = ["fight_id", "event_date", "fighter_id", "opponent_id", "target_id"]
    long, coverage = _build_long_sequences(
        snapshots[target_cols], obs, sequence_metrics, max_history_fights
    )
    extracted, manifest = _extract_tsfresh(
        long,
        pd.Index(snapshots["target_id"].astype(str), name="target_id"),
        n_jobs=n_jobs,
    )
    ts_directional = _directional_tsfresh(snapshots[target_cols], extracted)

    frame = manual_bank.merge(
        ts_directional,
        on=["fight_id", "fighter_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()

    manual_candidates = [c for c in manual_bank.columns if c not in NON_FEATURE_COLUMNS]
    physical = [c for c in manual_candidates if _is_physical(c)]
    manual_pruned = [
        c for c in manual_candidates
        if not _is_duration_experience(c) and not _is_target_mix(c)
    ]
    ts_candidates = [
        c for c in frame.columns
        if c.startswith(("self_tsf_", "opp_tsf_", "diff_tsf_"))
    ]

    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    seed = int(discovery_config["model"]["random_seed"])
    xgb_cfg = discovery_config["model"]["xgb"]

    for year in development_years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        train = frame[frame["event_date"].lt(start)].copy()
        valid = frame[frame["event_date"].ge(start) & frame["event_date"].lt(end)].copy()
        if valid.empty:
            continue

        usable_manual = _usable_features(train, manual_candidates)
        usable_pruned = _usable_features(train, manual_pruned)
        usable_physical = _usable_features(train, physical)
        usable_ts = _usable_features(train, ts_candidates)
        variants = {
            "physical_only": usable_physical,
            "tsfresh_only": usable_ts,
            "physical_plus_tsfresh": sorted(set(usable_physical) | set(usable_ts)),
            "manual_full": usable_manual,
            "manual_full_plus_tsfresh": sorted(set(usable_manual) | set(usable_ts)),
            "manual_pruned": usable_pruned,
            "manual_pruned_plus_tsfresh": sorted(set(usable_pruned) | set(usable_ts)),
        }

        for variant, features in variants.items():
            if not features:
                continue
            model, X_valid, pred = _fit_xgb(train, valid, features, xgb_cfg, seed + year)
            metric_rows.append({
                "fold": year,
                "variant": variant,
                "train_fights": int(train["fight_id"].nunique()),
                "valid_fights": int(valid["fight_id"].nunique()),
                "features": len(features),
                **_metrics(valid["fighter_win"].to_numpy(int), pred),
            })
            if variant == "manual_pruned_plus_tsfresh":
                imp = _fold_importance(model, X_valid, year)
                importance_rows.append(imp[imp["feature"].isin(usable_ts)].copy())

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("tsfresh challenger produced no development metrics")
    summary_df = (
        metrics_df.groupby("variant", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_features=("features", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_auc=("auc", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
        .sort_values("mean_log_loss")
    )

    importance = (
        pd.concat(importance_rows, ignore_index=True)
        if importance_rows
        else pd.DataFrame(columns=["fold", "feature", "gain", "mean_abs_shap", "shap_rank"])
    )
    stability, calculator_summary = _stability(importance, manifest)

    current = pd.to_datetime(coverage["event_date"])
    history_max = pd.to_datetime(coverage["history_max_date"], errors="coerce")
    bad_history = history_max.notna() & history_max.ge(current)
    audit = pd.DataFrame([
        {"check": "outer_start_unchanged", "passed": str(outer_start.date()) == "2024-01-01", "value": str(outer_start.date())},
        {"check": "outer_rows_reserved_not_scored", "passed": bool(frame["event_date"].lt(outer_start).all()), "value": outer_rows},
        {"check": "history_strictly_prior_date", "passed": not bad_history.any(), "value": int(bad_history.sum())},
        {"check": "exactly_two_directional_rows_per_fight", "passed": frame.groupby("fight_id").size().eq(2).all(), "value": frame.groupby("fight_id").size().value_counts().to_dict()},
        {"check": "targets_complement_within_fight", "passed": frame.groupby("fight_id")["fighter_win"].sum().eq(1).all(), "value": int((~frame.groupby("fight_id")["fighter_win"].sum().eq(1)).sum())},
        {"check": "development_folds_only", "passed": set(metrics_df["fold"].unique()).issubset(set(development_years)), "value": sorted(metrics_df["fold"].unique().tolist())},
    ])
    if not audit["passed"].all():
        print(audit.to_string(index=False))
        raise RuntimeError("tsfresh challenger audit failed")

    manifest["nonmissing_target_share"] = [
        float(extracted[c].notna().mean()) for c in manifest["feature"]
    ]
    coverage_summary = pd.DataFrame([
        {"bucket": "zero_history", "targets": int(coverage["history_fights_used"].eq(0).sum())},
        {"bucket": "one_history", "targets": int(coverage["history_fights_used"].eq(1).sum())},
        {"bucket": "two_plus_history", "targets": int(coverage["history_fights_used"].ge(2).sum())},
        {"bucket": "five_plus_history", "targets": int(coverage["history_fights_used"].ge(5).sum())},
    ])

    metrics_df.to_csv(out_root / "tsfresh_fold_metrics.csv", index=False)
    summary_df.to_csv(out_root / "tsfresh_metric_summary.csv", index=False)
    manifest.to_csv(out_root / "tsfresh_feature_manifest.csv", index=False)
    coverage_summary.to_csv(out_root / "tsfresh_history_coverage.csv", index=False)
    audit.to_csv(out_root / "tsfresh_audit.csv", index=False)
    importance.to_csv(out_root / "tsfresh_fold_importance.csv", index=False)
    stability.to_csv(out_root / "tsfresh_signal_stability.csv", index=False)
    calculator_summary.to_csv(out_root / "tsfresh_calculator_summary.csv", index=False)

    by_variant = {
        row["variant"]: {
            "log_loss": float(row["mean_log_loss"]),
            "brier": float(row["mean_brier"]),
            "auc": float(row["mean_auc"]),
            "accuracy": float(row["mean_accuracy"]),
            "mean_features": float(row["mean_features"]),
        }
        for row in summary_df.to_dict(orient="records")
    }
    summary = {
        "protocol": "development-only tsfresh challenger; 2024+ reserved and not scored",
        "outer_rows_reserved_not_scored": outer_rows,
        "development_rows": int(len(frame)),
        "development_fights": int(frame["fight_id"].nunique()),
        "sequence_metrics": sequence_metrics,
        "max_history_fights": max_history_fights,
        "tsfresh_base_features_extracted": int(len(extracted.columns)),
        "tsfresh_directional_candidates": int(len(ts_candidates)),
        "mean_metrics_by_variant": by_variant,
        "top_tsfresh_signals": stability.head(30).to_dict(orient="records") if not stability.empty else [],
        "top_calculators": calculator_summary.head(20).to_dict(orient="records") if not calculator_summary.empty else [],
    }
    (out_root / "tsfresh_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(audit.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(
        f"tsfresh extracted base features={len(extracted.columns):,} | "
        f"directional candidates={len(ts_candidates):,} | "
        f"development fights={frame['fight_id'].nunique():,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
