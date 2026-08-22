"""V3-fitted direct inference for Event Clock MC V2.

The model classes, regularization constants, hurdle/control architecture, and
standing-free-time machinery are imported unchanged from frozen Event Clock V1.
Only the input feature builder/feature columns and V3 submission-baseline source
differ. A V1 bundle must never be used with these V3 feature semantics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import (
    FAMILIES,
    POISSON_ALPHA,
    BINOMIAL_ALPHA,
    CONTROL_OCCURRENCE_ALPHA,
    CONTROL_AMOUNT_ALPHA,
    PoissonExposureRidge,
    BinomialRidge,
    ControlHurdle,
)
from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    build_pair_frame,
    physical_hurdle_probability,
    CONTROL_OCCURRENCE_FEATURES,
    CONTROL_OWNERSHIP_FEATURES,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage7_budget_timeline import (
    add_historical_free_time,
    fit_standing_free_time_model,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_full_fight_predictive_replay_shadow_ko_kd import (
    base_submission_rate,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11c_submission_conversion import (
    clip_probability,
    logistic,
    logit,
)
from pipeline.simulation.event_clock_mc_v2.feature_builder import (
    build_feature_rows_v3,
    direct_feature_columns_v3,
)


REACH_MULTIPLIER_COLUMN = "distance_reach_multiplier"


def _distance_reach_multipliers(frame: pd.DataFrame) -> np.ndarray:
    """Return validated distance-volume multipliers, neutral for legacy frames."""
    if REACH_MULTIPLIER_COLUMN not in frame.columns:
        return np.ones(len(frame), dtype=float)
    values = pd.to_numeric(frame[REACH_MULTIPLIER_COLUMN], errors="coerce").to_numpy(float)
    values = np.where(np.isfinite(values), values, 1.0)
    if np.any(values <= 0.0):
        raise ValueError("distance reach multipliers must be positive")
    return values


def _apply_distance_reach_translation(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply reach only to distance volume and derive the standing-rate change.

    Event Clock Stage 9 draws a combined standing attempt budget from
    ``pred_standing_rate_free_15m``.  Reach was validated only for DISTANCE
    attempt volume, not clinch volume or landing accuracy.  We therefore:

    1. multiply predicted distance attempts and landings by the reach multiplier;
    2. leave clinch predictions unchanged;
    3. compute the implied combined-standing volume ratio so the same effect can
       be applied to the Stage-9 free-time standing rate later.

    Scaling both distance attempts and distance landings preserves predicted
    distance accuracy exactly.
    """
    f = frame.copy()
    multiplier = _distance_reach_multipliers(f)

    base_distance_attempted = f["pred_distance_attempted"].to_numpy(float).copy()
    base_distance_landed = f["pred_distance_landed"].to_numpy(float).copy()
    clinch_attempted = f["pred_clinch_attempted"].to_numpy(float)

    f["pred_distance_attempted_base_no_reach"] = base_distance_attempted
    f["pred_distance_landed_base_no_reach"] = base_distance_landed
    f["pred_distance_attempted"] = base_distance_attempted * multiplier
    f["pred_distance_landed"] = base_distance_landed * multiplier

    base_standing_attempted = base_distance_attempted + clinch_attempted
    translated_standing_attempted = (
        f["pred_distance_attempted"].to_numpy(float) + clinch_attempted
    )
    standing_multiplier = np.divide(
        translated_standing_attempted,
        base_standing_attempted,
        out=np.ones_like(translated_standing_attempted),
        where=base_standing_attempted > 1e-12,
    )
    f["standing_reach_rate_multiplier"] = standing_multiplier
    return f


def _add_fitted_direct_predictions(
    train: pd.DataFrame,
    x: np.ndarray,
    exposure: np.ndarray,
    attempt_models: dict,
    completion_models: dict,
    control_direct_model,
) -> pd.DataFrame:
    """Attach the in-sample direct predictions required by V1 pair builders.

    Frozen V1 ``fit_inference_models`` receives a training frame that already
    came through ``prepare_direct_predictions`` and therefore already contains
    ``pred_*`` columns. ECV2 builds its V3 training frame directly from
    historical features/targets, so those fitted prediction columns must be
    created here before calling the unchanged V1 ``build_pair_frame`` helper.

    Reach is intentionally NOT folded into these fitted base models.  It is a
    separately validated post-model matchup translation applied only in forward
    inference, preserving the frozen direct-model schema and coefficients.
    """
    fitted = train.copy()

    for fam in FAMILIES:
        mu = attempt_models[fam].predict(x, exposure)
        p = completion_models[fam].predict_probability(x)
        fitted[f"pred_{fam}_attempted"] = mu
        fitted[f"pred_{fam}_landed"] = mu * p

    control, _, _ = control_direct_model.predict(x, exposure)
    fitted["pred_qualified_control_inflicted_seconds"] = np.minimum(
        control,
        fitted["duration"].to_numpy(float),
    )

    # Preserve the same physical fight-level cap used during forward inference:
    # red + blue predicted control cannot exceed total fight duration.
    for _, group in fitted.groupby("fight_id"):
        idx = group.index
        total = float(
            fitted.loc[idx, "pred_qualified_control_inflicted_seconds"].sum()
        )
        duration = float(group["duration"].iloc[0])
        if total > duration and total > 0.0:
            fitted.loc[idx, "pred_qualified_control_inflicted_seconds"] *= (
                duration / total
            )

    fitted["pred_standing_attempted"] = (
        fitted["pred_distance_attempted"] + fitted["pred_clinch_attempted"]
    )
    fitted["pred_standing_landed"] = (
        fitted["pred_distance_landed"] + fitted["pred_clinch_landed"]
    )
    return fitted


def fit_inference_models_v3(train: pd.DataFrame) -> dict:
    """Fit the frozen V1 direct-model architecture on V3 semantic features."""
    train = train.copy()
    cols = direct_feature_columns_v3()
    x = train[cols].to_numpy(float)
    exposure = train["duration"].to_numpy(float) / 900.0

    attempt_models, completion_models = {}, {}
    for fam in FAMILIES:
        attempted = f"{fam}_attempted"
        landed = f"{fam}_landed"
        attempt_models[fam] = PoissonExposureRidge(alpha=POISSON_ALPHA).fit(
            x,
            train[attempted].to_numpy(float),
            exposure,
        )
        completion_models[fam] = BinomialRidge(alpha=BINOMIAL_ALPHA).fit(
            x,
            train[landed].to_numpy(float),
            train[attempted].to_numpy(float),
        )

    control_direct_model = ControlHurdle(
        occurrence_alpha=CONTROL_OCCURRENCE_ALPHA,
        amount_alpha=CONTROL_AMOUNT_ALPHA,
    ).fit(
        x,
        train["qualified_control_inflicted_seconds"].to_numpy(float),
        exposure,
    )

    hurdle_occurrence_models = {
        fam: BinomialRidge(alpha=BINOMIAL_ALPHA).fit(
            x,
            (train[f"{fam}_attempted"].to_numpy(float) > 0).astype(float),
            np.ones(len(train)),
        )
        for fam in ("td", "ground")
    }

    # V1's pair/control models are trained from fitted direct predictions.
    # Our V3 frame did not previously contain those columns, which caused the
    # bundle build to fail at build_pair_frame().
    fitted_train = _add_fitted_direct_predictions(
        train,
        x,
        exposure,
        attempt_models,
        completion_models,
        control_direct_model,
    )

    standing_train = fitted_train.copy()
    standing_train["standing_attempted"] = (
        standing_train["distance_attempted"] + standing_train["clinch_attempted"]
    )
    standing_train["standing_landed"] = (
        standing_train["distance_landed"] + standing_train["clinch_landed"]
    )
    standing_train = add_historical_free_time(standing_train)
    _, _, standing_model, standing_alpha = fit_standing_free_time_model(
        standing_train,
        standing_train.copy(),
        cols,
    )

    pair = build_pair_frame(standing_train)
    control_occurrence_model = BinomialRidge(alpha=20.0).fit(
        pair[CONTROL_OCCURRENCE_FEATURES].to_numpy(float),
        (pair["actual_total_control"].to_numpy(float) > 0).astype(float),
        np.ones(len(pair)),
    )
    positive = pair["actual_total_control"] > 0
    control_ownership_model = BinomialRidge(alpha=20.0).fit(
        pair.loc[positive, CONTROL_OWNERSHIP_FEATURES].to_numpy(float),
        pair.loc[positive, "actual_red_control"].to_numpy(float) / 10.0,
        pair.loc[positive, "actual_total_control"].to_numpy(float) / 10.0,
    )

    return {
        "schema": "event_clock_mc_v2_fsr_v3_direct_v1",
        "feature_cols": cols,
        "attempt_models": attempt_models,
        "completion_models": completion_models,
        "control_direct_model": control_direct_model,
        "hurdle_occurrence_models": hurdle_occurrence_models,
        "standing_model": standing_model,
        "standing_alpha": float(standing_alpha),
        "control_occurrence_model": control_occurrence_model,
        "control_ownership_model": control_ownership_model,
    }


def load_submission_baseline_v3() -> pd.DataFrame:
    snapshots = pd.read_parquet(
        FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
        columns=["fight_id", "submission_conversion_baseline"],
    ).copy()
    snapshots["fight_id"] = snapshots["fight_id"].astype(str)
    snapshots["submission_conversion_baseline"] = pd.to_numeric(
        snapshots["submission_conversion_baseline"], errors="coerce"
    )
    if snapshots["submission_conversion_baseline"].isna().any():
        raise RuntimeError("FSR V3 submission_conversion_baseline contains nulls")

    check = snapshots.groupby("fight_id")["submission_conversion_baseline"].agg(
        baseline_min="min",
        baseline_max="max",
        submission_conversion_baseline="median",
    ).reset_index()
    max_range = float((check["baseline_max"] - check["baseline_min"]).max())
    if max_range > 1e-10:
        raise RuntimeError(
            "FSR V3 submission_conversion_baseline differs within a fight; "
            f"max range={max_range:.12g}"
        )
    return check[["fight_id", "submission_conversion_baseline"]]


def predict_feature_frame_v3(
    features: pd.DataFrame,
    models: dict,
    submission_scale: float,
    conversion_offset: float,
    *,
    submission_baseline: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a V3-fitted direct bundle to already-constructed feature rows."""
    f = features.copy()
    cols = models["feature_cols"]
    x = f[cols].to_numpy(float)
    exposure = f["duration"].to_numpy(float) / 900.0

    for fam in FAMILIES:
        mu = models["attempt_models"][fam].predict(x, exposure)
        p = models["completion_models"][fam].predict_probability(x)
        f[f"pred_{fam}_attempted"] = mu
        f[f"pred_{fam}_landed"] = mu * p
        # build_pair_frame expects these historical columns, but prediction
        # never consumes actual outcomes. Zero placeholders prevent leakage.
        f[f"{fam}_attempted"] = 0.0
        f[f"{fam}_landed"] = 0.0

    # Apply the empirically validated reach translation after the frozen direct
    # models.  This keeps model fitting/schema unchanged and changes only the
    # supported distance-volume mechanic.
    f = _apply_distance_reach_translation(f)

    control, _, _ = models["control_direct_model"].predict(x, exposure)
    f["pred_qualified_control_inflicted_seconds"] = np.minimum(
        control,
        f["duration"].to_numpy(float),
    )
    f["qualified_control_inflicted_seconds"] = 0.0

    for _, group in f.groupby("fight_id"):
        idx = group.index
        total = float(f.loc[idx, "pred_qualified_control_inflicted_seconds"].sum())
        duration = float(group["duration"].iloc[0])
        if total > duration:
            f.loc[idx, "pred_qualified_control_inflicted_seconds"] *= duration / total

    f["standing_attempted"] = 0.0
    f["standing_landed"] = 0.0
    f["pred_standing_attempted"] = f["pred_distance_attempted"] + f["pred_clinch_attempted"]
    f["pred_standing_landed"] = f["pred_distance_landed"] + f["pred_clinch_landed"]
    base_standing_rate = np.maximum(
        models["standing_model"].predict(f[cols]),
        0.0,
    )
    f["pred_standing_rate_free_15m_base_no_reach"] = base_standing_rate
    f["pred_standing_rate_free_15m"] = (
        base_standing_rate * f["standing_reach_rate_multiplier"].to_numpy(float)
    )

    for fam in ("td", "ground"):
        mu = f[f"pred_{fam}_attempted"].to_numpy(float)
        p = physical_hurdle_probability(
            mu,
            models["hurdle_occurrence_models"][fam].predict_probability(x),
        )
        f[f"pred_{fam}_positive_probability"] = p
        f[f"pred_{fam}_conditional_attempts"] = np.divide(
            mu,
            p,
            out=np.zeros_like(mu),
            where=p > 0,
        )

    pair = build_pair_frame(f)
    p_control = models["control_occurrence_model"].predict_probability(
        pair[CONTROL_OCCURRENCE_FEATURES].to_numpy(float)
    )
    floor = np.clip(
        pair["pred_total_control"].to_numpy(float)
        / pair["duration"].to_numpy(float),
        0.0,
        0.999999,
    )
    p_control = np.clip(np.maximum(p_control, floor), 1e-8, 0.999999)
    pair["pred_control_any_probability"] = p_control
    pair["pred_control_conditional_total"] = (
        pair["pred_total_control"].to_numpy(float) / p_control
    )
    pair["pred_red_control_share"] = models["control_ownership_model"].predict_probability(
        pair[CONTROL_OWNERSHIP_FEATURES].to_numpy(float)
    )

    f["submission_clock_rate"] = float(submission_scale) * base_submission_rate(f)
    baseline = load_submission_baseline_v3() if submission_baseline is None else submission_baseline
    f = f.merge(baseline, on="fight_id", how="left", validate="many_to_one")
    if f["submission_conversion_baseline"].isna().any():
        raise RuntimeError("Missing FSR V3 submission conversion baseline for target fight")
    f["submission_conversion_probability"] = logistic(
        logit(clip_probability(f["submission_conversion_baseline"]))
        + float(conversion_offset)
    )
    return f, pair


def predict_target_v3(
    target_master: pd.DataFrame,
    fsr_v3: pd.DataFrame,
    models: dict,
    submission_scale: float,
    conversion_offset: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean-only target prediction using scheduled horizon and V3 features."""
    features = build_feature_rows_v3(
        target_master,
        fsr_v3,
        scheduled_duration=True,
    )
    expected = set(target_master["fight_id"].astype(str))
    got = set(features["fight_id"].astype(str))
    if expected != got:
        raise RuntimeError(
            "Unable to build V3 target features for: "
            + ", ".join(sorted(expected - got))
        )
    return predict_feature_frame_v3(
        features,
        models,
        submission_scale,
        conversion_offset,
    )
