"""Falsify persistence-only explanations for next-round TD policy signal.

Measurement only.  Reuses the leakage-safe examples and chronological event-date
split from policy_learnability.py, then adds previous-round information in blocks:

C  prefight fighter + opponent profile
P  C + fighter's own previous-round TD choice (game-plan persistence control)
O  P + outcome of that TD choice
R  O + opponent's previous-round TD behavior
S  R + non-TD previous-round state/context

If O/R/S materially improve OOS prediction after P, and especially within fixed
previous-choice strata, the dynamic signal cannot be explained by simple
"I wrestled last round, therefore I wrestle again" persistence alone.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.simulation.event_clock_mc_v2.diagnostics.policy_learnability import (
    EPS,
    build_examples,
    build_prefight,
    load_rounds,
    split,
)

OUT = Path("data/diagnostics/event_clock_mc_v2/policy_adaptation_falsification")


def score(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_prediction": float(np.mean(p)),
        "accuracy": float(accuracy_score(y, p >= .5)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def fit_predict(tr, te, cols):
    m = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=4000),
    )
    m.fit(tr[cols], tr.attempted_td_next_round)
    return m.predict_proba(te[cols])[:, 1]


def existing(ex, names):
    return [c for c in names if c in ex.columns]


def feature_blocks(ex):
    base = [
        "fighter_prior_fights", "fighter_prior_rounds", "fighter_td_round_rate",
        "fighter_td_attempts_per_round", "fighter_td_land_rate", "target_round",
        "opponent_prior_fights", "opponent_prior_rounds",
        "opponent_td_attempt_round_exposure", "opponent_td_attempts_faced_per_round",
        "opponent_td_allowed_rate", "opponent_td_denial_rate",
    ]
    persistence = existing(ex, [
        "prev_own_attempted_td", "prev_own_td_attempted",
    ])
    own_outcome = existing(ex, [
        "prev_own__td_landed", "prev_own_td_failed", "prev_own_td_success_rate",
    ])
    opponent_response = existing(ex, [
        "prev_opp_attempted_td", "prev_opp_td_attempted", "prev_opp__td_landed",
        "prev_opp_td_failed",
    ])
    state = existing(ex, [
        "prev_own__distance_attempted", "prev_opp__distance_attempted",
        "prev_own__distance_landed", "prev_opp__distance_landed",
        "prev_own__clinch_attempted", "prev_opp__clinch_attempted",
        "prev_own__clinch_landed", "prev_opp__clinch_landed",
        "prev_own__ground_attempted", "prev_opp__ground_attempted",
        "prev_own__ground_landed", "prev_opp__ground_landed",
        "prev_own__sig_str_attempted", "prev_opp__sig_str_attempted",
        "prev_own__sig_str_landed", "prev_opp__sig_str_landed",
        "prev_own__total_str_attempted", "prev_opp__total_str_attempted",
        "prev_own__total_str_landed", "prev_opp__total_str_landed",
        "prev_own__ctrl_sec", "prev_opp__ctrl_sec",
        "prev_own__kd", "prev_opp__kd",
        "prev_own__sub_attempted", "prev_opp__sub_attempted",
        "prev_diff__distance_landed", "prev_diff__sig_str_landed",
        "prev_diff__total_str_landed", "prev_diff__ctrl_sec", "prev_diff__kd",
    ])
    return base, persistence, own_outcome, opponent_response, state


def evaluate(ex):
    tr, te, meta = split(ex)
    y = te.attempted_td_next_round.to_numpy(int)
    base, persistence, own_outcome, opp_response, state = feature_blocks(ex)

    specs = [
        ("C_prefight", base),
        ("P_plus_own_previous_choice", base + persistence),
        ("O_plus_own_td_outcome", base + persistence + own_outcome),
        ("R_plus_opponent_previous_behavior", base + persistence + own_outcome + opp_response),
        ("S_plus_non_td_state", base + persistence + own_outcome + opp_response + state),
        ("X_state_without_own_choice", base + own_outcome + opp_response + state),
    ]

    rows = []
    pred = te[[
        "fight_id", "event_date", "fighter_id", "opponent_id", "round", "target_round",
        "attempted_td_next_round", "prev_own_attempted_td", "prev_own_td_attempted",
    ]].copy()
    for label, cols in specs:
        p = fit_predict(tr, te, cols)
        pred[f"p__{label}"] = p
        rows.append({"model": label, "feature_count": len(cols), **score(y, p)})

    metrics = pd.DataFrame(rows)
    c = metrics[metrics.model.eq("C_prefight")].iloc[0]
    p0 = metrics[metrics.model.eq("P_plus_own_previous_choice")].iloc[0]
    metrics["delta_auc_vs_C"] = metrics.auc - c.auc
    metrics["delta_brier_vs_C"] = metrics.brier - c.brier
    metrics["delta_log_loss_vs_C"] = metrics.log_loss - c.log_loss
    metrics["delta_auc_vs_P"] = metrics.auc - p0.auc
    metrics["delta_brier_vs_P"] = metrics.brier - p0.brier
    metrics["delta_log_loss_vs_P"] = metrics.log_loss - p0.log_loss

    # Sequential marginal increments isolate what each block adds after the prior block.
    seq = []
    ordered = [x[0] for x in specs[:5]]
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        a = metrics[metrics.model.eq(prev)].iloc[0]
        b = metrics[metrics.model.eq(cur)].iloc[0]
        seq.append({
            "from_model": prev, "to_model": cur,
            "delta_auc": b.auc - a.auc,
            "delta_brier": b.brier - a.brier,
            "delta_log_loss": b.log_loss - a.log_loss,
        })
    sequential = pd.DataFrame(seq)

    # Fixed-choice strata: persistence itself cannot separate examples inside a stratum.
    strata = []
    for label, mask in [
        ("previous_no_td", pred.prev_own_attempted_td.eq(0)),
        ("previous_td", pred.prev_own_attempted_td.eq(1)),
    ]:
        q = pred[mask].copy()
        yy = q.attempted_td_next_round.to_numpy(int)
        for model in ["C_prefight", "P_plus_own_previous_choice", "O_plus_own_td_outcome",
                      "R_plus_opponent_previous_behavior", "S_plus_non_td_state"]:
            strata.append({"stratum": label, "model": model, **score(yy, q[f"p__{model}"].to_numpy())})
    stratum_metrics = pd.DataFrame(strata)

    # Direct behavioral rates make direction and effect size interpretable.
    desc = []
    conditions = {
        "previous_no_td": ex.prev_own_attempted_td.eq(0),
        "previous_td": ex.prev_own_attempted_td.eq(1),
        "previous_td_failed_all": ex.prev_own_td_attempted.gt(0) & ex.get("prev_own__td_landed", pd.Series(0, index=ex.index)).eq(0),
        "previous_td_landed": ex.get("prev_own__td_landed", pd.Series(0, index=ex.index)).gt(0),
        "opponent_td_previous": ex.prev_opp_attempted_td.eq(1),
        "opponent_no_td_previous": ex.prev_opp_attempted_td.eq(0),
    }
    for label, mask in conditions.items():
        q = ex[mask]
        desc.append({
            "condition": label, "n": len(q),
            "next_td_rate": q.attempted_td_next_round.mean(),
            "next_td_attempts_mean": q.next_td_attempted.mean(),
        })

    meta.update({
        "train_examples": len(tr), "test_examples": len(te),
        "base_features": base, "persistence_features": persistence,
        "own_outcome_features": own_outcome, "opponent_response_features": opp_response,
        "state_features": state,
    })
    return metrics, sequential, stratum_metrics, pd.DataFrame(desc), pred, meta


def main():
    rd = load_rounds()
    prefight = build_prefight(rd)
    ex = build_examples(rd, prefight)
    metrics, sequential, strata, desc, pred, meta = evaluate(ex)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "nested_model_metrics.csv", index=False)
    sequential.to_csv(OUT / "sequential_increments.csv", index=False)
    strata.to_csv(OUT / "fixed_choice_strata.csv", index=False)
    desc.to_csv(OUT / "behavioral_rates.csv", index=False)
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame([meta]).to_json(OUT / "split_and_features.json", orient="records", indent=2)

    print("TD POLICY — PERSISTENCE VS ADAPTATION FALSIFICATION")
    print(f"examples={len(ex)} fights={meta['fights']} train={meta['train_examples']} test={meta['test_examples']}")
    print(f"split: train through {meta['train_end_date']} | test from {meta['test_start_date']}")
    print("\nNESTED MODEL METRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nSEQUENTIAL INCREMENTS")
    print(sequential.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print("\nFIXED PREVIOUS-CHOICE STRATA")
    print(strata.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nBEHAVIORAL RATES")
    print(desc.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
