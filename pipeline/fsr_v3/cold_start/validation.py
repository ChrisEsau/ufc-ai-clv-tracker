"""Held-out native-target validation for the FSR V3 cold-start layer.

Nothing in this module changes published FSR. It trains an external-evidence
model on historical UFC debut outcomes, calibrates external equivalent evidence
strength on a later time window, and scores a still-later holdout. The same
external prior is also tested after one and two UFC observations by adding the
exact accumulated V3 UFC likelihood state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.math import nb2_log_likelihood, normalize_log_weights, weighted_mean_sd
from pipeline.fsr_v3.replay.rate_families import (
    RateFamilySpec,
    build_rate_fighter_fights,
    replay_tendency,
    standing_spec,
    takedown_spec,
)

from .features import build_external_feature_snapshots
from .model import ColdStartNB2RateModel, calibrate_extra_evidence_seconds
from .priors import combine_positive_rate_prior, gamma_logweights


@dataclass(frozen=True)
class ColdStartSplit:
    train_start: str = "2012-01-01"
    calibration_start: str = "2022-01-01"
    test_start: str = "2024-01-01"
    test_end: str | None = "2025-12-31"


@dataclass
class FamilyValidationResult:
    family: str
    scores: pd.DataFrame
    summary: pd.DataFrame
    bootstrap: pd.DataFrame
    calibration: pd.DataFrame
    coefficients: pd.DataFrame
    chosen_extra_seconds: dict[str, float]
    coverage: pd.DataFrame


def _prior_counts(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.sort_values(["event_date", "fight_id", "fighter_id"]).copy().reset_index(drop=True)
    counts: dict[str, int] = {}
    prior = np.zeros(len(x), dtype=int)
    for _, idx in x.groupby("event_date", sort=True).groups.items():
        idx = list(idx)
        for i in idx:
            prior[i] = counts.get(str(x.at[i, "fighter_id"]), 0)
        for i in idx:
            fighter = str(x.at[i, "fighter_id"])
            counts[fighter] = counts.get(fighter, 0) + 1
    x["prior_ufc_fights"] = prior
    x["ufc_bucket"] = np.where(prior >= 3, "3plus", prior.astype(str))
    return x


def _target_frame(spec: RateFamilySpec) -> pd.DataFrame:
    fights = build_rate_fighter_fights(spec)
    replay = replay_tendency(fights, spec).rename(columns={"denominator": "exposure_seconds"})
    replay = _prior_counts(replay)
    replay["as_of_date"] = replay["event_date"]
    return replay


def _grid(spec: RateFamilySpec) -> np.ndarray:
    return np.linspace(spec.tendency_grid_min, spec.tendency_grid_max, spec.tendency_grid_points)


def _score_history(
    frame: pd.DataFrame,
    *,
    spec: RateFamilySpec,
    extra_seconds: dict[str, float],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Score baseline vs cold-start prior with exact same accumulated UFC state."""
    grid = _grid(spec)
    states: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    source = frame.sort_values(["event_date", "fight_id", "fighter_id"]).copy()
    for event_date, batch in source.groupby("event_date", sort=True):
        pending: list[tuple[str, np.ndarray | None]] = []
        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            y = float(record["numerator"])
            exposure = float(record["exposure_seconds"])
            q_pop = float(record["population_rate_15m"])
            alpha = float(record["observation_alpha"])
            state = states.get(fighter)
            obs_ll = None
            if exposure > 0.0:
                obs_ll = nb2_log_likelihood(y, exposure / 900.0 * grid, alpha)

            base_prior = combine_positive_rate_prior(
                population_mean_rate_15m=q_pop,
                population_seconds=spec.tendency_prior_seconds,
            )
            base_lp = gamma_logweights(grid, base_prior)
            if state is not None:
                base_lp = base_lp + state
            base_w = normalize_log_weights(base_lp)
            base_mean, base_sd = weighted_mean_sd(grid, base_w)

            # The zero-external branch must be mathematically identical to the
            # validated V3 production replay before this study can be trusted.
            production_pre = record.get("pre_rating", np.nan)
            if pd.notna(production_pre) and abs(base_mean - float(production_pre)) > 1e-7:
                raise AssertionError(
                    f"{spec.name} cold-start baseline parity failed for {fighter} "
                    f"on {event_date}: replay={base_mean} production={production_pre}"
                )

            bucket = str(record.get("evidence_bucket", "none"))
            k_ext = float(extra_seconds.get(bucket, 0.0))
            q_ext = record.get("external_predicted_rate_15m", np.nan)
            ext_prior = combine_positive_rate_prior(
                population_mean_rate_15m=q_pop,
                population_seconds=spec.tendency_prior_seconds,
                external_mean_rate_15m=float(q_ext) if pd.notna(q_ext) else None,
                extra_seconds=k_ext,
            )
            ext_lp = gamma_logweights(grid, ext_prior)
            if state is not None:
                ext_lp = ext_lp + state
            ext_w = normalize_log_weights(ext_lp)
            ext_mean, ext_sd = weighted_mean_sd(grid, ext_w)

            if obs_ll is not None:
                base_pred_ll = float(logsumexp(base_lp + obs_ll) - logsumexp(base_lp))
                ext_pred_ll = float(logsumexp(ext_lp + obs_ll) - logsumexp(ext_lp))
                base_plugin_ll = float(nb2_log_likelihood(y, exposure / 900.0 * base_mean, alpha))
                ext_plugin_ll = float(nb2_log_likelihood(y, exposure / 900.0 * ext_mean, alpha))
            else:
                base_pred_ll = ext_pred_ll = base_plugin_ll = ext_plugin_ll = np.nan

            in_test = pd.Timestamp(event_date) >= test_start
            if test_end is not None:
                in_test = in_test and pd.Timestamp(event_date) <= test_end
            if in_test and int(record["prior_ufc_fights"]) <= 2:
                rows.append(
                    {
                        "family": spec.name,
                        "event_date": pd.Timestamp(event_date),
                        "test_year": int(pd.Timestamp(event_date).year),
                        "fight_id": str(record["fight_id"]),
                        "fighter_id": fighter,
                        "fighter_name": record["fighter_name"],
                        "opponent_name": record["opponent_name"],
                        "prior_ufc_fights": int(record["prior_ufc_fights"]),
                        "ufc_bucket": str(record["ufc_bucket"]),
                        "evidence_bucket": bucket,
                        "coverage_signature": record.get("coverage_signature", "none"),
                        "ext_bouts": int(record.get("ext_bouts", 0) or 0),
                        "external_extra_seconds": k_ext,
                        "population_rate_15m": q_pop,
                        "external_predicted_rate_15m": float(q_ext) if pd.notna(q_ext) else np.nan,
                        "baseline_pre_rate_15m": base_mean,
                        "cold_start_pre_rate_15m": ext_mean,
                        "baseline_pre_sd": base_sd,
                        "cold_start_pre_sd": ext_sd,
                        "actual_count": y,
                        "exposure_seconds": exposure,
                        "baseline_predictive_ll": base_pred_ll,
                        "cold_start_predictive_ll": ext_pred_ll,
                        "delta_predictive_ll": ext_pred_ll - base_pred_ll,
                        "baseline_plugin_ll": base_plugin_ll,
                        "cold_start_plugin_ll": ext_plugin_ll,
                        "delta_plugin_ll": ext_plugin_ll - base_plugin_ll,
                        "baseline_predicted_count": exposure / 900.0 * base_mean,
                        "cold_start_predicted_count": exposure / 900.0 * ext_mean,
                    }
                )
            pending.append((fighter, obs_ll))

        # Exact same-date delayed UFC evidence update as V3 production replay.
        for fighter, obs_ll in pending:
            if obs_ll is None:
                continue
            if fighter in states:
                states[fighter] = states[fighter] + obs_ll
                states[fighter] -= np.max(states[fighter])
            else:
                states[fighter] = obs_ll - np.max(obs_ll)
    return pd.DataFrame(rows)


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    records = []
    for coverage in ("ALL", "HAS_EXTERNAL"):
        c = scores if coverage == "ALL" else scores[scores["ext_bouts"] > 0]
        for bucket in ("ALL_0_2", "0", "1", "2"):
            g = c if bucket == "ALL_0_2" else c[c["ufc_bucket"] == bucket]
            g = g[np.isfinite(g["delta_predictive_ll"])]
            if g.empty:
                continue
            base_err = np.abs(g["baseline_predicted_count"] - g["actual_count"])
            cold_err = np.abs(g["cold_start_predicted_count"] - g["actual_count"])
            records.append(
                {
                    "family": str(g["family"].iloc[0]),
                    "coverage": coverage,
                    "test_year": "ALL",
                    "ufc_bucket": bucket,
                    "rows": int(len(g)),
                    "fights": int(g["fight_id"].nunique()),
                    "predictive_ll_delta": float(g["delta_predictive_ll"].sum()),
                    "mean_predictive_ll_delta": float(g["delta_predictive_ll"].mean()),
                    "plugin_ll_delta": float(g["delta_plugin_ll"].sum()),
                    "baseline_mae_count": float(base_err.mean()),
                    "cold_start_mae_count": float(cold_err.mean()),
                    "mae_delta": float(cold_err.mean() - base_err.mean()),
                    "mean_extra_seconds": float(g["external_extra_seconds"].mean()),
                }
            )
        for year, y in c.groupby("test_year", sort=True):
            g = y[y["prior_ufc_fights"].isin([0, 1])]
            g = g[np.isfinite(g["delta_predictive_ll"])]
            if g.empty:
                continue
            base_err = np.abs(g["baseline_predicted_count"] - g["actual_count"])
            cold_err = np.abs(g["cold_start_predicted_count"] - g["actual_count"])
            records.append(
                {
                    "family": str(g["family"].iloc[0]),
                    "coverage": coverage,
                    "test_year": str(int(year)),
                    "ufc_bucket": "EARLY_0_1",
                    "rows": int(len(g)),
                    "fights": int(g["fight_id"].nunique()),
                    "predictive_ll_delta": float(g["delta_predictive_ll"].sum()),
                    "mean_predictive_ll_delta": float(g["delta_predictive_ll"].mean()),
                    "plugin_ll_delta": float(g["delta_plugin_ll"].sum()),
                    "baseline_mae_count": float(base_err.mean()),
                    "cold_start_mae_count": float(cold_err.mean()),
                    "mae_delta": float(cold_err.mean() - base_err.mean()),
                    "mean_extra_seconds": float(g["external_extra_seconds"].mean()),
                }
            )
    return pd.DataFrame(records)


def _bootstrap(scores: pd.DataFrame, reps: int = 5000, seed: int = 20260821) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for coverage in ("ALL", "HAS_EXTERNAL"):
        c = scores if coverage == "ALL" else scores[scores["ext_bouts"] > 0]
        for bucket in ("0", "1", "2", "ALL_0_2"):
            g = c if bucket == "ALL_0_2" else c[c["ufc_bucket"] == bucket]
            if g.empty:
                continue
            clustered = g.groupby("fight_id")["delta_predictive_ll"].sum().to_numpy(float)
            n = len(clustered)
            if n == 0:
                continue
            draws = rng.integers(0, n, size=(reps, n))
            vals = clustered[draws].sum(axis=1)
            records.append(
                {
                    "coverage": coverage,
                    "ufc_bucket": bucket,
                    "fights": n,
                    "predictive_ll_delta": float(clustered.sum()),
                    "ci_2_5": float(np.quantile(vals, 0.025)),
                    "ci_97_5": float(np.quantile(vals, 0.975)),
                    "p_delta_gt_zero": float(np.mean(vals > 0.0)),
                }
            )
    return pd.DataFrame(records)


def validate_rate_family(
    external_bouts: pd.DataFrame,
    *,
    spec: RateFamilySpec,
    split: ColdStartSplit = ColdStartSplit(),
    ridge_alpha: float = 20.0,
) -> FamilyValidationResult:
    targets = _target_frame(spec)
    snapshots = build_external_feature_snapshots(
        targets[
            [
                "event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id",
                "opponent_name", "numerator", "exposure_seconds", "population_rate_15m",
                "observation_alpha", "pre_rating", "prior_ufc_fights", "ufc_bucket", "as_of_date",
            ]
        ],
        external_bouts,
    )

    train_start = pd.Timestamp(split.train_start)
    cal_start = pd.Timestamp(split.calibration_start)
    test_start = pd.Timestamp(split.test_start)
    test_end = pd.Timestamp(split.test_end) if split.test_end is not None else None
    if not (train_start < cal_start < test_start):
        raise ValueError("cold-start split must satisfy train_start < calibration_start < test_start")
    if test_end is not None and test_end < test_start:
        raise ValueError("cold-start test_end must be on/after test_start")

    train = snapshots[
        (snapshots["event_date"] >= train_start)
        & (snapshots["event_date"] < cal_start)
        & (snapshots["prior_ufc_fights"] == 0)
        & (snapshots["ext_bouts"] > 0)
        & (snapshots["exposure_seconds"] > 0)
    ].copy()
    calibration = snapshots[
        (snapshots["event_date"] >= cal_start)
        & (snapshots["event_date"] < test_start)
        & (snapshots["prior_ufc_fights"] == 0)
        & (snapshots["exposure_seconds"] > 0)
    ].copy()
    if len(train) < 30:
        raise ValueError(f"{spec.name}: insufficient external-evidence debut training rows: {len(train)}")

    model = ColdStartNB2RateModel(ridge_alpha=ridge_alpha).fit(train)
    snapshots["external_predicted_rate_15m"] = model.predict_rate(snapshots)
    calibration["external_predicted_rate_15m"] = model.predict_rate(calibration)

    chosen, calibration_table = calibrate_extra_evidence_seconds(
        calibration,
        population_seconds=spec.tendency_prior_seconds,
        grid=_grid(spec),
        candidates=(0.0, 30.0, 60.0, 90.0, 135.0, 180.0, 270.0, 360.0, 540.0, 720.0, 1080.0, 1440.0),
    )
    # Sparse calibration cells are not allowed to create confident priors.
    calibration_counts = calibration.groupby("evidence_bucket").size().to_dict()
    for bucket in list(chosen):
        if bucket == "none" or int(calibration_counts.get(bucket, 0)) < 25:
            chosen[bucket] = 0.0

    scores = _score_history(
        snapshots,
        spec=spec,
        extra_seconds=chosen,
        test_start=test_start,
        test_end=test_end,
    )

    coverage_source = snapshots
    if test_end is not None:
        coverage_source = coverage_source[coverage_source["event_date"] <= test_end]
    coverage = (
        coverage_source.assign(period=np.select(
            [coverage_source["event_date"] < cal_start, coverage_source["event_date"] < test_start],
            ["train", "calibration"],
            default="test",
        ))
        .groupby(["period", "ufc_bucket", "evidence_bucket"], as_index=False)
        .agg(rows=("fight_id", "size"), fighters=("fighter_id", "nunique"))
    )
    return FamilyValidationResult(
        family=spec.name,
        scores=scores,
        summary=_summary(scores),
        bootstrap=_bootstrap(scores),
        calibration=calibration_table,
        coefficients=model.coefficient_frame(),
        chosen_extra_seconds=chosen,
        coverage=coverage,
    )


def validate_standing_and_takedown(
    external_bouts: pd.DataFrame,
    *,
    split: ColdStartSplit = ColdStartSplit(),
) -> dict[str, FamilyValidationResult]:
    config = FSRV3Config()
    return {
        "standing_striking": validate_rate_family(external_bouts, spec=standing_spec(config), split=split),
        "takedown": validate_rate_family(external_bouts, spec=takedown_spec(config), split=split),
    }


def write_results(results: dict[str, FamilyValidationResult], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result.scores.to_csv(out / f"{name}_scores.csv", index=False)
        result.summary.to_csv(out / f"{name}_summary.csv", index=False)
        result.bootstrap.to_csv(out / f"{name}_bootstrap.csv", index=False)
        result.calibration.to_csv(out / f"{name}_strength_calibration.csv", index=False)
        result.coefficients.to_csv(out / f"{name}_coefficients.csv", index=False)
        result.coverage.to_csv(out / f"{name}_coverage.csv", index=False)
        pd.DataFrame(
            [{"evidence_bucket": k, "extra_seconds": v} for k, v in result.chosen_extra_seconds.items()]
        ).to_csv(out / f"{name}_chosen_strength.csv", index=False)
