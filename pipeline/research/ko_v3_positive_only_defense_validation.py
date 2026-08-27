"""Validate KO V3 where defender susceptibility can only add KO hazard.

Research only; production unchanged.

Architecture under test
-----------------------
Attacker KO offense is the base hazard. Defender KO-loss susceptibility contributes
only when its shrunk rate is ABOVE the chronological population prior. A defender
with a below-population/zero-KO-loss history cannot suppress demonstrated attacker
KO offense.

    logit(h) = logit(p_att) + max(0, logit(p_def) - logit(p0)) + age_delta

Attacker prior strength is fixed at O50. Defender strengths are swept to measure
sensitivity, with D50 the direct apples-to-apples architecture test. Age slopes are
fit strictly before each evaluation block: pre-2020 for selection 2020-2024 and
pre-2025 for untouched confirmation 2025-2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUTDIR = Path("data/research/ko_v3_positive_only_defense_validation")
O_STRENGTH = 50.0
D_STRENGTHS = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
EPS = 1e-9


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_chronological_population_prior(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    daily = (
        df[df["sig_landed"] > 0]
        .groupby("event_date", as_index=False)
        .agg(day_ko=("ko_win", "sum"), day_sig=("sig_landed", "sum"))
        .sort_values("event_date")
    )
    daily["prior_ko"] = daily["day_ko"].cumsum().shift(1, fill_value=0.0)
    daily["prior_sig"] = daily["day_sig"].cumsum().shift(1, fill_value=0.0)
    daily["population_ko_per_sig"] = np.where(
        daily["prior_sig"] > 0, daily["prior_ko"] / daily["prior_sig"], np.nan
    )
    return df.merge(
        daily[["event_date", "population_ko_per_sig"]],
        on="event_date", how="left", validate="many_to_one"
    )


def fit_age_slopes_before(frame: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[float, float, dict]:
    tr = frame[
        (frame["event_date"] < cutoff)
        & (frame["sig_landed"] > 0)
        & frame["attacker_age"].notna()
        & frame["defender_age"].notna()
    ].copy()
    xs = []
    ys = []
    ws = []
    for r in tr.itertuples(index=False):
        n = float(r.sig_landed)
        k = float(r.ko_win)
        x = [float(r.attacker_age) - 30.0, float(r.defender_age) - 30.0]
        if k > 0:
            xs.append(x); ys.append(1); ws.append(k)
        if n - k > 0:
            xs.append(x); ys.append(0); ws.append(n - k)
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    model.fit(np.asarray(xs, float), np.asarray(ys, int), sample_weight=np.asarray(ws, float))
    ba = float(model.coef_[0][0])
    bd = float(model.coef_[0][1])
    return ba, bd, {
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "fit_rows": int(len(tr)),
        "fit_ko_wins": int(tr["ko_win"].sum()),
        "fit_sig_landed": float(tr["sig_landed"].sum()),
        "attacker_age_logodds_per_year": ba,
        "defender_age_logodds_per_year": bd,
    }


def components(df: pd.DataFrame, d_strength: float):
    p0 = df["population_ko_per_sig"].to_numpy(float)
    att_k = df["prior_ko_wins"].to_numpy(float)
    att_n = df["prior_sig_landed"].to_numpy(float)
    def_k = df["opp_prior_ko_losses"].to_numpy(float)
    def_n = df["opp_prior_sig_absorbed"].to_numpy(float)
    p_att = (att_k + O_STRENGTH * p0) / (att_n + O_STRENGTH)
    p_def = (def_k + d_strength * p0) / (def_n + d_strength)
    d_delta = _logit(p_def) - _logit(p0)
    symmetric = _sigmoid(_logit(p_att) + d_delta)
    positive_only = _sigmoid(_logit(p_att) + np.maximum(0.0, d_delta))
    return p_att, p_def, symmetric, positive_only, d_delta


def add_age(hazard: np.ndarray, df: pd.DataFrame, beta_att: float, beta_def: float) -> np.ndarray:
    delta = (
        beta_att * (df["attacker_age"].to_numpy(float) - 30.0)
        + beta_def * (df["defender_age"].to_numpy(float) - 30.0)
    )
    return _sigmoid(_logit(hazard) + delta)


def metrics(df: pd.DataFrame, hazard: np.ndarray) -> dict:
    h = np.asarray(hazard, dtype=float)
    y = df["ko_win"].to_numpy(int)
    n = df["sig_landed"].to_numpy(float)
    hc = _clip(h)
    strike_ll = -float(np.sum(y * np.log(hc) + (n - y) * np.log(1.0 - hc)) / np.sum(n))
    expected = float(np.sum(h * n)); actual = float(np.sum(y))
    auc = float(roc_auc_score(y, h)) if len(np.unique(y)) > 1 else float("nan")

    tmp = df[["fight_id", "ko_win"]].copy(); tmp["hazard"] = h
    correct = []
    for _, g in tmp.groupby("fight_id"):
        if len(g) != 2 or int(g["ko_win"].sum()) != 1:
            continue
        wh = float(g.loc[g["ko_win"] == 1, "hazard"].iloc[0])
        lh = float(g.loc[g["ko_win"] == 0, "hazard"].iloc[0])
        correct.append(1.0 if wh > lh else (0.5 if wh == lh else 0.0))

    cutoff = np.quantile(h, 0.90); top = y[h >= cutoff]
    winners = h[y == 1]; non = h[y == 0]
    return {
        "rows": int(len(df)), "sig_landed": float(np.sum(n)), "ko_wins": int(np.sum(y)),
        "actual_ko_per_sig": float(actual / np.sum(n)),
        "expected_to_observed": float(expected / actual) if actual > 0 else None,
        "strike_weighted_log_loss": strike_ll,
        "fighter_row_auc": auc,
        "correct_side_ko_rate": float(np.mean(correct)) if correct else None,
        "correct_side_ko_fights": int(len(correct)),
        "top_decile_precision": float(np.mean(top)) if len(top) else None,
        "mean_hazard_ko_winners": float(np.mean(winners)) if len(winners) else None,
        "mean_hazard_non_ko": float(np.mean(non)) if len(non) else None,
        "winner_nonwinner_hazard_ratio": float(np.mean(winners) / np.mean(non)) if len(winners) and np.mean(non) > 0 else None,
        "p50_hazard": float(np.quantile(h, 0.50)),
        "p90_hazard": float(np.quantile(h, 0.90)),
        "p99_hazard": float(np.quantile(h, 0.99)),
    }


def period_frame(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    mask = (
        frame["event_date"].dt.year.between(start_year, end_year)
        & (frame["sig_landed"] > 0)
        & frame["population_ko_per_sig"].notna()
        & frame["attacker_age"].notna()
        & frame["defender_age"].notna()
    )
    return frame.loc[mask].copy().reset_index(drop=True)


def evaluate(df: pd.DataFrame, beta_att: float, beta_def: float) -> dict:
    result = {"defender_strength_curve": {}}
    for ds in D_STRENGTHS:
        p_att, p_def, symmetric, positive, d_delta = components(df, ds)
        offense_age = add_age(p_att, df, beta_att, beta_def)
        sym_age = add_age(symmetric, df, beta_att, beta_def)
        pos_age = add_age(positive, df, beta_att, beta_def)
        result["defender_strength_curve"][str(int(ds))] = {
            "offense_only": metrics(df, offense_age),
            "symmetric_defense": metrics(df, sym_age),
            "positive_only_defense": metrics(df, pos_age),
            "defender_delta_positive_fraction": float(np.mean(d_delta > 0.0)),
            "defender_delta_negative_fraction": float(np.mean(d_delta < 0.0)),
        }
    return result


def main():
    ff, audit = s1.load_raw_fighter_fights()
    frame = s1.build_matchup_frame(s1.build_prefight_states(ff)).copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame = add_chronological_population_prior(frame)

    selection = period_frame(frame, 2020, 2024)
    confirmation = period_frame(frame, 2025, 2026)
    sel_ba, sel_bd, sel_age = fit_age_slopes_before(frame, pd.Timestamp("2020-01-01"))
    con_ba, con_bd, con_age = fit_age_slopes_before(frame, pd.Timestamp("2025-01-01"))

    result = {
        "study": "KO V3 positive-only defender susceptibility",
        "architecture": "O50 attacker base + max(0, defender logit deviation) + age",
        "production_changed": False,
        "offense_prior_strength": O_STRENGTH,
        "defender_strengths": list(D_STRENGTHS),
        "selection_years": "2020-2024",
        "confirmation_years": "2025-2026",
        "age_fit_selection": sel_age,
        "age_fit_confirmation": con_age,
        "selection": evaluate(selection, sel_ba, sel_bd),
        "confirmation": evaluate(confirmation, con_ba, con_bd),
        "stage1_audit": audit,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nKO V3 POSITIVE-ONLY DEFENDER VALIDATION")
    print("=" * 118)
    print(f"O fixed at {int(O_STRENGTH)}")
    for label, block in (("SELECTION 2020-2024", result["selection"]), ("CONFIRMATION 2025-2026", result["confirmation"])):
        print(f"\n{label}")
        print("D | architecture       | strike_LL | E/O   | AUC    | correct | top10  | winner/non")
        for ds in D_STRENGTHS:
            arm = block["defender_strength_curve"][str(int(ds))]
            for name in ("offense_only", "symmetric_defense", "positive_only_defense"):
                m = arm[name]
                print(f"{int(ds):<3d}| {name:18s} | {m['strike_weighted_log_loss']:.8f} | {m['expected_to_observed']:.3f} | {m['fighter_row_auc']:.4f} | {m['correct_side_ko_rate']:.4f} | {m['top_decile_precision']:.4f} | {m['winner_nonwinner_hazard_ratio']:.3f}")
            print(f"   defender delta: + {arm['defender_delta_positive_fraction']:.3f} / - {arm['defender_delta_negative_fraction']:.3f}")
    print(f"\nWrote {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
