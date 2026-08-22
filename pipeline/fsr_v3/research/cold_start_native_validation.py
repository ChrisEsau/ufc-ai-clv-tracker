"""Held-out validation of external-evidence cold-start priors for FSR V3.

This diagnostic is measurement-only. It does not write production FSR outputs
or alter validated trait configuration.

Protocol
--------
* external source: dated non-UFC MMA fight facts only
* train external->native mapping: 2016-2022 early-career UFC observations
* calibrate equivalent external evidence seconds: 2023-2024 UFC debuts only
* final untouched test: calendar 2025
* native targets: exact FSR V3 standing and takedown tendency NB2 likelihoods
* report debut (0 prior UFC) and fight-2 (1 prior UFC) separately
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.cold_start.features import build_external_feature_snapshots
from pipeline.fsr_v3.cold_start.mma_global import load_mma_global_fighter_bouts
from pipeline.fsr_v3.cold_start.model import ColdStartNB2RateModel, calibrate_extra_evidence_seconds
from pipeline.fsr_v3.cold_start.replay import add_prior_ufc_fight_count, paired_fight_bootstrap, score_seeded_tendency
from pipeline.fsr_v3.replay.rate_families import (
    RateFamilySpec,
    build_rate_fighter_fights,
    replay_tendency,
    standing_spec,
    takedown_spec,
)

TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
CAL_START = pd.Timestamp("2023-01-01")
CAL_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")


def _candidate_seconds(spec: RateFamilySpec) -> tuple[float, ...]:
    k = float(spec.tendency_prior_seconds)
    raw = [0.0, 0.125 * k, 0.25 * k, 0.5 * k, k, 2.0 * k, 4.0 * k]
    return tuple(sorted({round(float(v), 6) for v in raw}))


def _early_target_frame(spec: RateFamilySpec) -> pd.DataFrame:
    fights = build_rate_fighter_fights(spec)
    history = replay_tendency(fights, spec)
    history = add_prior_ufc_fight_count(history)
    history["as_of_date"] = history["event_date"]
    return history


def _fit_external_model(frame: pd.DataFrame) -> ColdStartNB2RateModel:
    train = frame[
        frame["event_date"].between(TRAIN_START, TRAIN_END)
        & frame["prior_ufc_fights"].isin([0, 1])
        & pd.to_numeric(frame["ext_bouts"], errors="coerce").gt(0)
        & pd.to_numeric(frame["denominator"], errors="coerce").gt(0)
    ].copy()
    if len(train) < 100:
        raise RuntimeError(f"insufficient cold-start training rows: {len(train)}")
    model = ColdStartNB2RateModel(ridge_alpha=20.0)
    model.fit(
        train,
        count_column="numerator",
        exposure_column="denominator",
        population_rate_column="population_rate_15m",
        alpha_column="observation_alpha",
    )
    return model


def _predict_external(model: ColdStartNB2RateModel, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["external_predicted_rate_15m"] = model.predict_rate(
        out,
        population_rate_column="population_rate_15m",
    )
    no_evidence = pd.to_numeric(out["ext_bouts"], errors="coerce").fillna(0).le(0)
    out.loc[no_evidence, "external_predicted_rate_15m"] = out.loc[
        no_evidence, "population_rate_15m"
    ]
    return out


def _calibrate_strengths(frame: pd.DataFrame, spec: RateFamilySpec) -> tuple[dict[str, float], pd.DataFrame]:
    # Calibrate only on true UFC debuts. That makes the strength estimate a
    # clean external-prior question, uncontaminated by previous UFC evidence.
    calibration = frame[
        frame["event_date"].between(CAL_START, CAL_END)
        & frame["prior_ufc_fights"].eq(0)
        & pd.to_numeric(frame["ext_bouts"], errors="coerce").gt(0)
        & pd.to_numeric(frame["denominator"], errors="coerce").gt(0)
    ].copy()
    grid = np.linspace(
        spec.tendency_grid_min,
        spec.tendency_grid_max,
        spec.tendency_grid_points,
    )
    chosen, table = calibrate_extra_evidence_seconds(
        calibration,
        population_seconds=float(spec.tendency_prior_seconds),
        grid=grid,
        candidates=_candidate_seconds(spec),
        external_rate_column="external_predicted_rate_15m",
        count_column="numerator",
        exposure_column="denominator",
        population_rate_column="population_rate_15m",
        alpha_column="observation_alpha",
    )
    # Do not allow a tiny calibration cell to create a production-sized prior.
    bucket_counts = calibration.groupby("evidence_bucket").size().to_dict()
    for bucket, strength in list(chosen.items()):
        if bucket == "none":
            chosen[bucket] = 0.0
        elif int(bucket_counts.get(bucket, 0)) < 25:
            chosen[bucket] = 0.0
    return chosen, table


def _evaluate(
    frame: pd.DataFrame,
    spec: RateFamilySpec,
    strengths: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = score_seeded_tendency(
        frame,
        spec,
        extra_seconds_by_bucket={},
    )
    parity = np.nanmax(np.abs(baseline["cold_pre_rating"].to_numpy(float) - baseline["pre_rating"].to_numpy(float)))
    if not np.isfinite(parity) or parity > 1e-8:
        raise AssertionError(f"zero-external replay failed production parity: max pre-rating error={parity}")

    cold = score_seeded_tendency(
        frame,
        spec,
        extra_seconds_by_bucket=strengths,
    )
    keys = ["event_date", "fight_id", "fighter_id"]
    b = baseline[keys + ["cold_predictive_ll", "cold_pre_rating", "cold_pre_posterior_sd"]].rename(
        columns={
            "cold_predictive_ll": "baseline_predictive_ll",
            "cold_pre_rating": "baseline_pre_rating_replayed",
            "cold_pre_posterior_sd": "baseline_pre_sd_replayed",
        }
    )
    scored = cold.merge(b, on=keys, how="left", validate="one_to_one")
    scored["delta_predictive_ll"] = scored["cold_predictive_ll"] - scored["baseline_predictive_ll"]
    scored["delta_pre_rating"] = scored["cold_pre_rating"] - scored["baseline_pre_rating_replayed"]
    scored["delta_pre_sd"] = scored["cold_pre_posterior_sd"] - scored["baseline_pre_sd_replayed"]
    test = scored[
        scored["event_date"].between(TEST_START, TEST_END)
        & scored["prior_ufc_fights"].isin([0, 1])
        & pd.to_numeric(scored["denominator"], errors="coerce").gt(0)
    ].copy()

    summaries: list[dict[str, object]] = []
    slices = [
        ("all_early", test),
        ("debut_0", test[test["prior_ufc_fights"] == 0]),
        ("fight2_1", test[test["prior_ufc_fights"] == 1]),
        ("external_only", test[pd.to_numeric(test["ext_bouts"], errors="coerce").gt(0)]),
    ]
    for label, part in slices:
        result = paired_fight_bootstrap(part, delta_column="delta_predictive_ll")
        result.update(
            {
                "trait": spec.tendency_trait,
                "slice": label,
                "fighter_rows": int(len(part)),
                "external_coverage": float(pd.to_numeric(part["ext_bouts"], errors="coerce").gt(0).mean()) if len(part) else np.nan,
                "mean_baseline_ll": float(part["baseline_predictive_ll"].mean()) if len(part) else np.nan,
                "mean_cold_ll": float(part["cold_predictive_ll"].mean()) if len(part) else np.nan,
                "mean_delta_pre_sd": float(part["delta_pre_sd"].mean()) if len(part) else np.nan,
            }
        )
        summaries.append(result)

    for bucket, part in test.groupby("evidence_bucket", sort=True):
        result = paired_fight_bootstrap(part, delta_column="delta_predictive_ll")
        result.update(
            {
                "trait": spec.tendency_trait,
                "slice": f"bucket_{bucket}",
                "fighter_rows": int(len(part)),
                "external_coverage": float(pd.to_numeric(part["ext_bouts"], errors="coerce").gt(0).mean()) if len(part) else np.nan,
                "mean_baseline_ll": float(part["baseline_predictive_ll"].mean()) if len(part) else np.nan,
                "mean_cold_ll": float(part["cold_predictive_ll"].mean()) if len(part) else np.nan,
                "mean_delta_pre_sd": float(part["delta_pre_sd"].mean()) if len(part) else np.nan,
            }
        )
        summaries.append(result)
    return test, pd.DataFrame(summaries)


def run_trait(
    spec: RateFamilySpec,
    external_bouts: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    print("=" * 120)
    print(f"COLD START NATIVE VALIDATION — {spec.tendency_trait}")
    print("=" * 120)
    base = _early_target_frame(spec)
    targets = base[[
        "event_date", "fight_id", "fighter_id", "fighter_name", "prior_ufc_fights"
    ]].rename(columns={"event_date": "as_of_date"})
    features = build_external_feature_snapshots(targets, external_bouts)
    features = features.rename(columns={"as_of_date": "event_date"})
    join_keys = ["event_date", "fight_id", "fighter_id"]
    frame = base.merge(
        features.drop(columns=["fighter_name", "prior_ufc_fights"], errors="ignore"),
        on=join_keys,
        how="left",
        validate="one_to_one",
    )

    model = _fit_external_model(frame)
    frame = _predict_external(model, frame)
    strengths, strength_table = _calibrate_strengths(frame, spec)
    test_rows, summary = _evaluate(frame, spec, strengths)

    stem = spec.tendency_trait
    test_rows.to_csv(output_dir / f"{stem}_test_rows.csv", index=False)
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    strength_table.to_csv(output_dir / f"{stem}_strength_grid.csv", index=False)
    model.coefficient_frame().to_csv(output_dir / f"{stem}_coefficients.csv", index=False)

    print("chosen equivalent external evidence seconds:", strengths)
    print(summary.to_string(index=False))
    return {
        "trait": stem,
        "strengths": strengths,
        "test_rows": int(len(test_rows)),
        "summary": summary.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mma-db", required=True)
    parser.add_argument("--output-dir", default="data/diagnostics/fsr_v3/cold_start_native_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    external_bouts = load_mma_global_fighter_bouts(args.mma_db)
    print(f"canonical external fighter-bout rows: {len(external_bouts):,}")

    config = FSRV3Config()
    results = []
    for spec in (standing_spec(config), takedown_spec(config)):
        results.append(run_trait(spec, external_bouts, output_dir))

    manifest = {
        "status": "measurement_only",
        "train_window": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "calibration_window": [str(CAL_START.date()), str(CAL_END.date())],
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "source": "MMA Global Database / fights_career_longitudinal, non-UFC rows only",
        "traits": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
