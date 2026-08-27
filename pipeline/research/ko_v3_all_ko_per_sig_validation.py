"""Validate population-prior shrinkage for the simple all-KO-per-sig hazard.

Research only. Production simulator mechanics are unchanged.

Architecture held fixed:
    attacker history = prior all KO/TKO wins / prior sig strikes landed
    defender history = opponent prior all KO/TKO losses / prior sig strikes absorbed
    combined per-landed hazard = 1 - (1-attacker_rate)*(1-defender_rate)
    fight KO probability = 1 - (1-combined_per_landed)**current_fight_sig_landed

Only the rate estimator changes. Fighter histories are shrunk toward a population
KO-per-significant-strike prior computed strictly from fights BEFORE each event date:
    shrunk_rate = (events + S * population_rate) / (exposure + S)
where S is measured in significant-strike exposure units.

Selection: 2020-2024. Confirmation: 2025-2026.
Market data are never used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUT = Path("data/research/ko_v3_all_ko_per_sig_validation")
SELECTION_YEARS = tuple(range(2020, 2025))
CONFIRMATION_YEARS = (2025, 2026)
PRIOR_STRENGTHS = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0)


def combine(a, d):
    a = np.clip(np.asarray(a, float), 0.0, 1.0)
    d = np.clip(np.asarray(d, float), 0.0, 1.0)
    return 1.0 - (1.0 - a) * (1.0 - d)


def add_strict_population_prior(frame: pd.DataFrame) -> pd.DataFrame:
    """Add global KO/sig prior using only rows from dates strictly before each row."""
    x = frame.copy()
    x["event_date"] = pd.to_datetime(x["event_date"]).dt.normalize()
    daily = (
        x.groupby("event_date", as_index=False)
        .agg(day_ko_wins=("ko_win", "sum"), day_sig_landed=("sig_landed", "sum"))
        .sort_values("event_date")
    )
    daily["prior_population_ko_wins"] = daily["day_ko_wins"].cumsum().shift(1, fill_value=0.0)
    daily["prior_population_sig_landed"] = daily["day_sig_landed"].cumsum().shift(1, fill_value=0.0)
    daily["population_ko_per_sig"] = np.divide(
        daily["prior_population_ko_wins"].to_numpy(float),
        daily["prior_population_sig_landed"].to_numpy(float),
        out=np.full(len(daily), np.nan, dtype=float),
        where=daily["prior_population_sig_landed"].to_numpy(float) > 0,
    )
    return x.merge(
        daily[["event_date", "population_ko_per_sig", "prior_population_ko_wins", "prior_population_sig_landed"]],
        on="event_date",
        how="left",
        validate="many_to_one",
    )


def shrunk_rate(k, n, p0, strength):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p0 = np.asarray(p0, float)
    s = float(strength)
    if s == 0.0:
        return np.divide(k, n, out=p0.copy(), where=n > 0)
    return (k + s * p0) / (n + s)


def metrics(g: pd.DataFrame, p_fight: np.ndarray) -> dict:
    y = g["ko_win"].astype(int).to_numpy()
    p = np.clip(np.asarray(p_fight, float), 1e-9, 1.0 - 1e-9)
    return {
        "rows": int(len(g)),
        "ko_wins": int(y.sum()),
        "actual_ko_win_rate": float(y.mean()),
        "mean_predicted_ko_probability": float(p.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top_decile_precision": float(y[p >= np.quantile(p, 0.9)].mean()) if len(g) else np.nan,
        "extreme_false_positives_ge_50": int(((p >= 0.50) & (y == 0)).sum()),
        "mean_p_actual_ko_winners": float(p[y == 1].mean()) if y.sum() else np.nan,
        "mean_p_non_ko": float(p[y == 0].mean()) if (y == 0).sum() else np.nan,
    }


def add_candidate(frame: pd.DataFrame, strength: float) -> pd.DataFrame:
    x = frame.copy()
    p0 = x["population_ko_per_sig"].to_numpy(float)
    att = shrunk_rate(x["prior_ko_wins"], x["prior_sig_landed"], p0, strength)
    deff = shrunk_rate(x["opp_prior_ko_losses"], x["opp_prior_sig_absorbed"], p0, strength)
    valid = np.isfinite(att) & np.isfinite(deff) & x["sig_landed"].gt(0).to_numpy()
    x = x.loc[valid].copy()
    att = att[valid]
    deff = deff[valid]
    p_sig = combine(att, deff)
    n = x["sig_landed"].to_numpy(float)
    p_fight = 1.0 - np.power(1.0 - p_sig, n)
    x["att_ko_per_sig"] = att
    x["def_ko_loss_per_sig"] = deff
    x["combined_per_sig"] = p_sig
    x["p_fight"] = p_fight
    x["prior_strength"] = float(strength)
    return x


def score_variant(x: pd.DataFrame, years) -> dict:
    g = x[x.test_year.isin(years)].copy()
    return metrics(g, g["p_fight"].to_numpy(float))


def correct_side(frame: pd.DataFrame, years) -> dict:
    g = frame[frame.test_year.isin(years)].copy()
    rows = []
    for _, b in g.groupby("fight_id"):
        if len(b) != 2 or not bool(b.ko_win.any()):
            continue
        winner = b[b.ko_win].iloc[0]
        loser = b[~b.ko_win].iloc[0]
        rows.append(float(winner.p_fight) > float(loser.p_fight))
    return {"ko_fights": len(rows), "correct_side_rate": float(np.mean(rows)) if rows else np.nan}


def zero_hazard_audit(frame: pd.DataFrame, years) -> dict:
    g = frame[frame.test_year.isin(years)].copy()
    return {
        "rows": int(len(g)),
        "zero_per_sig_hazards": int((g.combined_per_sig <= 0.0).sum()),
        "zero_fight_probabilities": int((g.p_fight <= 0.0).sum()),
    }


def population_baseline(frame: pd.DataFrame, years) -> dict:
    g = frame[frame.test_year.isin(years) & frame.sig_landed.gt(0) & frame.population_ko_per_sig.notna()].copy()
    p0 = g.population_ko_per_sig.to_numpy(float)
    p = 1.0 - np.power(1.0 - p0, g.sig_landed.to_numpy(float))
    return metrics(g, p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ff, audit = s1.load_raw_fighter_fights(ROUND_STATS_PATH, MASTER_PATH)
    states = s1.build_prefight_states(ff)
    frame = add_strict_population_prior(s1.build_matchup_frame(states))

    variants = {}
    predictions = []
    for strength in PRIOR_STRENGTHS:
        x = add_candidate(frame, strength)
        key = f"s{int(strength)}"
        variants[key] = {
            "prior_strength_sig_strikes": float(strength),
            "selection": score_variant(x, SELECTION_YEARS),
            "confirmation": score_variant(x, CONFIRMATION_YEARS),
            "selection_correct_side": correct_side(x, SELECTION_YEARS),
            "confirmation_correct_side": correct_side(x, CONFIRMATION_YEARS),
            "selection_zero_hazard_audit": zero_hazard_audit(x, SELECTION_YEARS),
            "confirmation_zero_hazard_audit": zero_hazard_audit(x, CONFIRMATION_YEARS),
        }
        predictions.append(x[[
            "event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id",
            "ko_win", "sig_landed", "population_ko_per_sig", "prior_strength",
            "att_ko_per_sig", "def_ko_loss_per_sig", "combined_per_sig", "p_fight",
        ]])

    selected_key = min(variants, key=lambda k: variants[k]["selection"]["log_loss"])
    selected = variants[selected_key]
    raw = variants["s0"]
    report = {
        "study": "population-prior shrinkage of all KO/TKO per-sig attacker + defender hazard",
        "architecture_changed": False,
        "same_date_delayed_fighter_histories": True,
        "population_prior_strictly_before_event_date": True,
        "market_used": False,
        "changes_mc": False,
        "selection_years": list(SELECTION_YEARS),
        "confirmation_years": list(CONFIRMATION_YEARS),
        "prior_strength_grid_sig_strikes": list(PRIOR_STRENGTHS),
        "selection_rule": "minimum selection log_loss",
        "selected": {"key": selected_key, **selected},
        "selected_vs_raw_confirmation": {
            "log_loss_delta": float(selected["confirmation"]["log_loss"] - raw["confirmation"]["log_loss"]),
            "brier_delta": float(selected["confirmation"]["brier"] - raw["confirmation"]["brier"]),
            "auc_delta": float(selected["confirmation"]["auc"] - raw["confirmation"]["auc"]),
            "calibration_bias_abs_delta": float(abs(selected["confirmation"]["calibration_bias"]) - abs(raw["confirmation"]["calibration_bias"])),
            "correct_side_delta": float(selected["confirmation_correct_side"]["correct_side_rate"] - raw["confirmation_correct_side"]["correct_side_rate"]),
        },
        "population": {
            "selection": population_baseline(frame, SELECTION_YEARS),
            "confirmation": population_baseline(frame, CONFIRMATION_YEARS),
        },
        "variants": variants,
        "raw_audit": audit,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "shrinkage_predictions.csv", index=False)
    print("KO V3 ALL-KO PER-SIG SHRINKAGE VALIDATION")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
