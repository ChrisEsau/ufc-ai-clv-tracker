from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import build_feature_rows
from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import FAMILIES, POISSON_ALPHA, BINOMIAL_ALPHA, CONTROL_OCCURRENCE_ALPHA, CONTROL_AMOUNT_ALPHA, PoissonExposureRidge, BinomialRidge, ControlHurdle
from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import direct_feature_columns
from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import build_pair_frame, physical_hurdle_probability, CONTROL_OCCURRENCE_FEATURES, CONTROL_OWNERSHIP_FEATURES
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage7_budget_timeline import add_historical_free_time, fit_standing_free_time_model
from pipeline.simulation.event_clock_mc_v1.diagnostics_full_fight_predictive_replay_shadow_ko_kd import base_submission_rate
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11c_submission_conversion import load_submission_baseline, clip_probability, logistic, logit


def fit_inference_models(train):
    train = train.copy()
    cols = direct_feature_columns()
    x = train[cols].to_numpy(float)
    exposure = train["duration"].to_numpy(float) / 900.0
    att, comp = {}, {}
    for fam in FAMILIES:
        a, l = f"{fam}_attempted", f"{fam}_landed"
        att[fam] = PoissonExposureRidge(alpha=POISSON_ALPHA).fit(x, train[a].to_numpy(float), exposure)
        comp[fam] = BinomialRidge(alpha=BINOMIAL_ALPHA).fit(x, train[l].to_numpy(float), train[a].to_numpy(float))
    ctrl = ControlHurdle(occurrence_alpha=CONTROL_OCCURRENCE_ALPHA, amount_alpha=CONTROL_AMOUNT_ALPHA).fit(x, train["qualified_control_inflicted_seconds"].to_numpy(float), exposure)
    hurdle = {fam: BinomialRidge(alpha=BINOMIAL_ALPHA).fit(x, (train[f"{fam}_attempted"].to_numpy(float) > 0).astype(float), np.ones(len(train))) for fam in ("td", "ground")}
    t = train.copy()
    t["standing_attempted"] = t["distance_attempted"] + t["clinch_attempted"]
    t["standing_landed"] = t["distance_landed"] + t["clinch_landed"]
    t = add_historical_free_time(t)
    _, _, standing_model, standing_alpha = fit_standing_free_time_model(t, t.copy(), cols)
    pair = build_pair_frame(t)
    cocc = BinomialRidge(alpha=20.0).fit(pair[CONTROL_OCCURRENCE_FEATURES].to_numpy(float), (pair["actual_total_control"].to_numpy(float) > 0).astype(float), np.ones(len(pair)))
    pos = pair["actual_total_control"] > 0
    cown = BinomialRidge(alpha=20.0).fit(pair.loc[pos, CONTROL_OWNERSHIP_FEATURES].to_numpy(float), pair.loc[pos, "actual_red_control"].to_numpy(float) / 10.0, pair.loc[pos, "actual_total_control"].to_numpy(float) / 10.0)
    return {"feature_cols": cols, "attempt_models": att, "completion_models": comp, "control_direct_model": ctrl, "hurdle_occurrence_models": hurdle, "standing_model": standing_model, "standing_alpha": float(standing_alpha), "control_occurrence_model": cocc, "control_ownership_model": cown}


def predict_target(target_master, fsr_all, models, submission_scale, conversion_offset):
    f = build_feature_rows(target_master, fsr_all).copy()
    expected = set(target_master["fight_id"].astype(str))
    got = set(f["fight_id"].astype(str))
    if expected != got:
        raise RuntimeError("Unable to build target features for: " + ", ".join(sorted(expected - got)))
    f["fight_id"] = f["fight_id"].astype(str)
    f["duration"] = f["scheduled_rounds"].astype(float) * 300.0
    cols = models["feature_cols"]
    x = f[cols].to_numpy(float)
    exposure = f["duration"].to_numpy(float) / 900.0
    for fam in FAMILIES:
        mu = models["attempt_models"][fam].predict(x, exposure)
        p = models["completion_models"][fam].predict_probability(x)
        f[f"pred_{fam}_attempted"] = mu
        f[f"pred_{fam}_landed"] = mu * p
        # build_pair_frame expects historical columns, but forward prediction
        # never consumes them. Zero placeholders prevent outcome leakage.
        f[f"{fam}_attempted"] = 0.0
        f[f"{fam}_landed"] = 0.0
    control, _, _ = models["control_direct_model"].predict(x, exposure)
    f["pred_qualified_control_inflicted_seconds"] = np.minimum(control, f["duration"].to_numpy(float))
    f["qualified_control_inflicted_seconds"] = 0.0
    for _, g in f.groupby("fight_id"):
        idx = g.index
        total, duration = float(f.loc[idx, "pred_qualified_control_inflicted_seconds"].sum()), float(g["duration"].iloc[0])
        if total > duration:
            f.loc[idx, "pred_qualified_control_inflicted_seconds"] *= duration / total
    f["standing_attempted"] = 0.0
    f["standing_landed"] = 0.0
    f["pred_standing_attempted"] = f["pred_distance_attempted"] + f["pred_clinch_attempted"]
    f["pred_standing_landed"] = f["pred_distance_landed"] + f["pred_clinch_landed"]
    f["pred_standing_rate_free_15m"] = np.maximum(models["standing_model"].predict(f[cols]), 0.0)
    for fam in ("td", "ground"):
        mu = f[f"pred_{fam}_attempted"].to_numpy(float)
        p = physical_hurdle_probability(mu, models["hurdle_occurrence_models"][fam].predict_probability(x))
        f[f"pred_{fam}_positive_probability"] = p
        f[f"pred_{fam}_conditional_attempts"] = np.divide(mu, p, out=np.zeros_like(mu), where=p > 0)
    pair = build_pair_frame(f)
    p = models["control_occurrence_model"].predict_probability(pair[CONTROL_OCCURRENCE_FEATURES].to_numpy(float))
    floor = np.clip(pair["pred_total_control"].to_numpy(float) / pair["duration"].to_numpy(float), 0.0, 0.999999)
    p = np.clip(np.maximum(p, floor), 1e-8, 0.999999)
    pair["pred_control_any_probability"] = p
    pair["pred_control_conditional_total"] = pair["pred_total_control"].to_numpy(float) / p
    pair["pred_red_control_share"] = models["control_ownership_model"].predict_probability(pair[CONTROL_OWNERSHIP_FEATURES].to_numpy(float))
    f["submission_clock_rate"] = float(submission_scale) * base_submission_rate(f)
    base = load_submission_baseline()
    f = f.merge(base, on="fight_id", how="left", validate="many_to_one")
    if f["submission_conversion_baseline"].isna().any():
        raise RuntimeError("Missing FSR submission conversion baseline for target fight.")
    f["submission_conversion_probability"] = logistic(logit(clip_probability(f["submission_conversion_baseline"])) + float(conversion_offset))
    return f, pair
