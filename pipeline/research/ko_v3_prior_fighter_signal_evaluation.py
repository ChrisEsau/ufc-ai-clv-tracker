"""Evaluate total-KO prior strength versus retained fighter signal.

Research only. No simulator or production changes.

Purpose
-------
The current KO V3 shadow shrinks attacker all-KO/Sig and defender all-KO-loss/Sig
rates toward a chronological population prior, then combines their *deviations*
from that prior on the logit scale. S=400 sig strikes was selected previously on
2020-2024 probability loss. This script evaluates whether that strength is
calibrating genuine signal or simply compressing fighter differences.

Primary metrics deliberately operate on the per-landed-significant-strike hazard:
- strike-weighted Bernoulli log loss
- expected KO events / observed KO events
- fighter-row AUC using the hazard itself (not realized current-fight exposure)
- correct-side hazard ranking within actual KO/TKO fights
- top-decile KO winner precision
- winner/non-winner hazard separation

It also ablates attacker offense, defender susceptibility, and both at S=400.
All fighter histories are the same-date-delayed Stage-1 prefight histories.
Population priors are also strictly before each event date.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUTDIR = Path("data/research/ko_v3_prior_fighter_signal_evaluation")
STRENGTHS = (25.0, 50.0, 100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1200.0)
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
    """Attach population KO/Sig using rows strictly before each event date."""
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
        daily["prior_sig"] > 0,
        daily["prior_ko"] / daily["prior_sig"],
        np.nan,
    )
    return df.merge(
        daily[["event_date", "population_ko_per_sig"]],
        on="event_date",
        how="left",
        validate="many_to_one",
    )


def components(df: pd.DataFrame, strength: float):
    p0 = df["population_ko_per_sig"].to_numpy(float)
    att_k = df["prior_ko_wins"].to_numpy(float)
    att_n = df["prior_sig_landed"].to_numpy(float)
    def_k = df["opp_prior_ko_losses"].to_numpy(float)
    def_n = df["opp_prior_sig_absorbed"].to_numpy(float)

    p_att = (att_k + strength * p0) / (att_n + strength)
    p_def = (def_k + strength * p0) / (def_n + strength)
    both = _sigmoid(_logit(p0) + (_logit(p_att) - _logit(p0)) + (_logit(p_def) - _logit(p0)))
    return p_att, p_def, both


def literal_raw(df: pd.DataFrame):
    att_n = df["prior_sig_landed"].to_numpy(float)
    def_n = df["opp_prior_sig_absorbed"].to_numpy(float)
    att = np.divide(df["prior_ko_wins"].to_numpy(float), att_n, out=np.zeros(len(df)), where=att_n > 0)
    deff = np.divide(df["opp_prior_ko_losses"].to_numpy(float), def_n, out=np.zeros(len(df)), where=def_n > 0)
    return 1.0 - (1.0 - att) * (1.0 - deff)


def metrics(df: pd.DataFrame, hazard: np.ndarray) -> dict:
    h = np.asarray(hazard, dtype=float)
    y = df["ko_win"].to_numpy(int)
    n = df["sig_landed"].to_numpy(float)
    hc = _clip(h)

    # Exact grouped Bernoulli likelihood for 0/1 KO event among n landed strikes.
    strike_ll = -float(np.sum(y * np.log(hc) + (n - y) * np.log(1.0 - hc)) / np.sum(n))
    expected = float(np.sum(h * n))
    actual = float(np.sum(y))
    auc = float(roc_auc_score(y, h)) if len(np.unique(y)) > 1 else float("nan")

    tmp = df[["fight_id", "ko_win"]].copy()
    tmp["hazard"] = h
    correct = []
    for _, g in tmp.groupby("fight_id"):
        if len(g) != 2 or int(g["ko_win"].sum()) != 1:
            continue
        win_h = float(g.loc[g["ko_win"] == 1, "hazard"].iloc[0])
        lose_h = float(g.loc[g["ko_win"] == 0, "hazard"].iloc[0])
        correct.append(1.0 if win_h > lose_h else (0.5 if win_h == lose_h else 0.0))

    cutoff = np.quantile(h, 0.90)
    top = y[h >= cutoff]
    winners = h[y == 1]
    non = h[y == 0]
    return {
        "rows": int(len(df)),
        "sig_landed": float(np.sum(n)),
        "ko_wins": int(np.sum(y)),
        "actual_ko_per_sig": float(actual / np.sum(n)),
        "mean_hazard_strike_weighted": float(np.sum(h * n) / np.sum(n)),
        "expected_ko_events": expected,
        "observed_ko_events": actual,
        "expected_to_observed": float(expected / actual) if actual > 0 else None,
        "strike_weighted_log_loss": strike_ll,
        "fighter_row_auc": auc,
        "correct_side_ko_rate": float(np.mean(correct)) if correct else None,
        "correct_side_ko_fights": int(len(correct)),
        "top_decile_precision": float(np.mean(top)) if len(top) else None,
        "mean_hazard_ko_winners": float(np.mean(winners)) if len(winners) else None,
        "mean_hazard_non_ko": float(np.mean(non)) if len(non) else None,
        "winner_nonwinner_hazard_ratio": float(np.mean(winners) / np.mean(non)) if len(winners) and np.mean(non) > 0 else None,
        "zero_hazards": int(np.sum(h <= 0.0)),
        "p50_hazard": float(np.quantile(h, 0.50)),
        "p90_hazard": float(np.quantile(h, 0.90)),
        "p99_hazard": float(np.quantile(h, 0.99)),
    }


def period_frame(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    mask = (
        frame["event_date"].dt.year.between(start_year, end_year)
        & (frame["sig_landed"] > 0)
        & frame["population_ko_per_sig"].notna()
    )
    return frame.loc[mask].copy().reset_index(drop=True)


def evaluate_period(df: pd.DataFrame) -> dict:
    out = {"literal_raw_union": metrics(df, literal_raw(df)), "strength_curve": {}, "s400_ablation": {}}
    for strength in STRENGTHS:
        p_att, p_def, both = components(df, strength)
        out["strength_curve"][str(int(strength))] = metrics(df, both)
        if strength == 400.0:
            out["s400_ablation"] = {
                "attacker_only": metrics(df, p_att),
                "defender_only": metrics(df, p_def),
                "both": metrics(df, both),
            }
    return out


def main():
    ff, audit = s1.load_raw_fighter_fights()
    states = s1.build_prefight_states(ff)
    frame = s1.build_matchup_frame(states).copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame = add_chronological_population_prior(frame)

    selection = period_frame(frame, 2020, 2024)
    confirmation = period_frame(frame, 2025, 2026)
    result = {
        "study": "KO V3 prior strength vs fighter signal",
        "architecture": "population-centered attacker KO/Sig + defender KO-loss/Sig logit deviations",
        "production_changed": False,
        "selection_years": "2020-2024",
        "confirmation_years": "2025-2026",
        "strengths_sig_strikes": list(STRENGTHS),
        "selection": evaluate_period(selection),
        "confirmation": evaluate_period(confirmation),
        "stage1_audit": audit,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    # Compact terminal table focused on the tradeoff.
    print("\nKO V3 PRIOR STRENGTH VS FIGHTER SIGNAL")
    print("=" * 100)
    for label, block in (("SELECTION 2020-2024", result["selection"]), ("CONFIRMATION 2025-2026", result["confirmation"])):
        print(f"\n{label}")
        print("strength | strike_LL | E/O | fighter_AUC | correct_side | top10_precision | winner/non ratio | zeros")
        raw = block["literal_raw_union"]
        print(f"raw      | {raw['strike_weighted_log_loss']:.8f} | {raw['expected_to_observed']:.3f} | {raw['fighter_row_auc']:.4f} | {raw['correct_side_ko_rate']:.4f} | {raw['top_decile_precision']:.4f} | {raw['winner_nonwinner_hazard_ratio']:.3f} | {raw['zero_hazards']}")
        for s in STRENGTHS:
            m = block["strength_curve"][str(int(s))]
            print(f"{int(s):<8d} | {m['strike_weighted_log_loss']:.8f} | {m['expected_to_observed']:.3f} | {m['fighter_row_auc']:.4f} | {m['correct_side_ko_rate']:.4f} | {m['top_decile_precision']:.4f} | {m['winner_nonwinner_hazard_ratio']:.3f} | {m['zero_hazards']}")
        print("\nS400 ABLATION")
        for arm, m in block["s400_ablation"].items():
            print(f"{arm:14s} LL={m['strike_weighted_log_loss']:.8f} E/O={m['expected_to_observed']:.3f} AUC={m['fighter_row_auc']:.4f} correct={m['correct_side_ko_rate']:.4f} ratio={m['winner_nonwinner_hazard_ratio']:.3f}")

    print(f"\nWrote {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
