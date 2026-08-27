"""Leakage-safe submission-finish architecture audit.

Research only. Production Event Clock mechanics are not modified.

Decomposes submission finishing into:
1) prefight attempt opportunity,
2) conversion given actual effective attempts, and
3) a static Poisson attempt x conversion overlay.

The static overlay is diagnostic only; it is not the causal Brain and does not
include terminal competition with KO/decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.fsr_v2.replay.engine import aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.mechanics.submission import SUBMISSION_OFFENSE_DEFENSE_BETA

OUTDIR = Path("data/research/submission_finish_architecture_audit")
EPS = 1e-9
BETA_STAGE11C = 1.0
BETA_CURRENT = float(SUBMISSION_OFFENSE_DEFENSE_BETA)
TARGETS = (
    ("Brendan Allen", "Edmen Shahbazyan"),
    ("Fares Ziam", "Tom Nolan"),
    ("Belal Muhammad", "Gabriel Bonfim"),
    ("Matt Schnell", "Alessandro Costa"),
    ("Bruno Silva", "Edgar Chairez"),
    ("Jordan Leavitt", "Joanderson Brito"),
    ("Ketlen Souza", "Ariane Carnelossi"),
    ("Karol Rosa", "Luana Santos"),
    ("Manel Kape", "Kyoji Horiguchi"),
)


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)


def logit(x):
    p = clip_prob(x)
    return np.log(p / (1.0 - p))


def logistic(x):
    z = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def at_least_one_success(per_attempt_p, attempts):
    p = np.clip(np.asarray(per_attempt_p, dtype=float), 0.0, 1.0)
    n = np.maximum(np.asarray(attempts, dtype=float), 0.0)
    return 1.0 - np.power(1.0 - p, n)


def poisson_any(mu):
    return 1.0 - np.exp(-np.maximum(np.asarray(mu, dtype=float), 0.0))


def build_frame() -> pd.DataFrame:
    fights = aggregate_fights(build_paired_rounds()).copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"]).dt.normalize()
    for col in ("fight_id", "fighter_id", "opponent_id"):
        fights[col] = fights[col].astype(str)

    snapshots = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    snapshots["event_date"] = pd.to_datetime(snapshots["event_date"]).dt.normalize()
    for col in ("fight_id", "fighter_id"):
        snapshots[col] = snapshots[col].astype(str)

    required = [
        "submission_tendency", "submission_suppression", "submission_offense",
        "submission_defense", "submission_conversion_baseline",
    ]
    missing = [c for c in required if c not in snapshots.columns]
    if missing:
        raise RuntimeError(f"FSR V3 snapshot missing submission fields: {missing}")

    own = snapshots[["event_date", "fight_id", "fighter_id", *required]].copy()
    frame = fights.merge(
        own, on=["event_date", "fight_id", "fighter_id"], how="inner", validate="one_to_one"
    )
    if frame.empty:
        raise RuntimeError("submission audit lost all rows joining FSR V3 snapshots")

    opp = own[[
        "event_date", "fight_id", "fighter_id", "submission_suppression", "submission_defense",
    ]].rename(columns={
        "fighter_id": "opponent_id",
        "submission_suppression": "opp_submission_suppression",
        "submission_defense": "opp_submission_defense",
    })
    frame = frame.merge(
        opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one"
    )
    if frame[["opp_submission_suppression", "opp_submission_defense"]].isna().any().any():
        raise RuntimeError("missing reciprocal submission traits after opponent join")

    numeric = required + [
        "opp_submission_suppression", "opp_submission_defense",
        "effective_submission_attempts", "submission_finish", "fight_elapsed_seconds",
    ]
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    if frame[numeric].isna().any().any():
        bad = [c for c in numeric if frame[c].isna().any()]
        raise RuntimeError(f"non-numeric/null submission audit fields: {bad}")

    frame["submission_finish"] = frame["submission_finish"].astype(int)
    frame["effective_submission_attempts"] = np.maximum(
        frame["effective_submission_attempts"].to_numpy(float),
        frame["submission_finish"].to_numpy(float),
    )
    frame["attempt_rate_tendency_only"] = np.maximum(
        frame["submission_tendency"].to_numpy(float), 0.0
    )
    frame["attempt_rate_with_suppression"] = (
        np.maximum(frame["submission_tendency"].to_numpy(float), 0.0)
        * np.maximum(frame["opp_submission_suppression"].to_numpy(float), 0.0)
    )
    frame["submission_edge"] = (
        frame["submission_offense"].to_numpy(float)
        - frame["opp_submission_defense"].to_numpy(float)
    )
    return frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def fit_attempt_scale(train: pd.DataFrame, rate_col: str) -> float:
    raw = train[rate_col].to_numpy(float) * train["fight_elapsed_seconds"].to_numpy(float)
    actual = train["effective_submission_attempts"].to_numpy(float)
    return float(actual.sum() / max(raw.sum(), 1e-12))


def attempt_metrics(df: pd.DataFrame, mu: np.ndarray) -> dict:
    y_count = df["effective_submission_attempts"].to_numpy(float)
    y_any = (y_count > 0).astype(int)
    mu = np.maximum(np.asarray(mu, float), 0.0)
    p_any = np.clip(poisson_any(mu), EPS, 1.0 - EPS)
    auc = float(roc_auc_score(y_any, p_any)) if np.unique(y_any).size == 2 else float("nan")
    pear = float(pearsonr(y_count, mu).statistic) if len(df) > 2 and np.std(mu) > 0 else float("nan")
    spear = float(spearmanr(y_count, mu).statistic) if len(df) > 2 and np.std(mu) > 0 else float("nan")
    actual = float(y_count.sum())
    expected = float(mu.sum())
    return {
        "rows": int(len(df)), "actual_attempts": actual, "expected_attempts": expected,
        "expected_to_observed": expected / actual if actual > 0 else None,
        "mean_actual_attempts": float(y_count.mean()), "mean_expected_attempts": float(mu.mean()),
        "mae_attempts": float(np.mean(np.abs(y_count - mu))),
        "pearson_count": pear, "spearman_count": spear,
        "actual_any_attempt_rate": float(y_any.mean()), "mean_pred_any_attempt": float(p_any.mean()),
        "any_attempt_auc": auc, "any_attempt_brier": float(brier_score_loss(y_any, p_any)),
        "any_attempt_log_loss": float(log_loss(y_any, p_any, labels=[0, 1])),
    }


def fit_conversion_offset(train: pd.DataFrame, beta: float) -> float:
    active = train[train["effective_submission_attempts"] > 0].copy()
    if active.empty:
        raise RuntimeError("no attempt-positive rows for conversion fit")
    y = active["submission_finish"].to_numpy(int)
    n = active["effective_submission_attempts"].to_numpy(float)
    base = logit(active["submission_conversion_baseline"].to_numpy(float))
    edge = active["submission_edge"].to_numpy(float)

    def objective(alpha):
        per = logistic(base + float(alpha) + float(beta) * edge)
        q = np.clip(at_least_one_success(per, n), EPS, 1.0 - EPS)
        return float(-np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))

    fit = minimize_scalar(objective, bounds=(-8.0, 8.0), method="bounded")
    if not fit.success:
        raise RuntimeError(f"conversion offset fit failed for beta={beta}")
    return float(fit.x)


def conversion_probabilities(df: pd.DataFrame, alpha: float, beta: float):
    per = logistic(
        logit(df["submission_conversion_baseline"].to_numpy(float))
        + float(alpha) + float(beta) * df["submission_edge"].to_numpy(float)
    )
    q = at_least_one_success(per, df["effective_submission_attempts"].to_numpy(float))
    return np.asarray(per, float), np.asarray(q, float)


def binary_metrics(df: pd.DataFrame, q: np.ndarray) -> dict:
    y = df["submission_finish"].to_numpy(int)
    q = np.clip(np.asarray(q, float), EPS, 1.0 - EPS)
    auc = float(roc_auc_score(y, q)) if np.unique(y).size == 2 else float("nan")
    actual = float(y.sum())
    expected = float(q.sum())
    return {
        "rows": int(len(df)), "actual_sub_wins": int(actual), "expected_sub_wins": expected,
        "expected_to_observed": expected / actual if actual > 0 else None,
        "mean_actual": float(y.mean()), "mean_predicted": float(q.mean()), "auc": auc,
        "brier": float(brier_score_loss(y, q)), "log_loss": float(log_loss(y, q, labels=[0, 1])),
        "winner_mean_p": float(np.mean(q[y == 1])) if np.any(y == 1) else None,
        "nonwinner_mean_p": float(np.mean(q[y == 0])) if np.any(y == 0) else None,
    }


def correct_side(df: pd.DataFrame, q: np.ndarray):
    tmp = df[["fight_id", "submission_finish"]].copy()
    tmp["q"] = np.asarray(q, float)
    values = []
    for _, g in tmp.groupby("fight_id"):
        if len(g) != 2 or int(g["submission_finish"].sum()) != 1:
            continue
        wp = float(g.loc[g["submission_finish"] == 1, "q"].iloc[0])
        lp = float(g.loc[g["submission_finish"] == 0, "q"].iloc[0])
        values.append(1.0 if wp > lp else 0.5 if wp == lp else 0.0)
    return (float(np.mean(values)) if values else None, int(len(values)))


def with_correct_side(df: pd.DataFrame, q: np.ndarray) -> dict:
    out = binary_metrics(df, q)
    side, n = correct_side(df, q)
    out["correct_side_sub_rate"] = side
    out["correct_side_sub_fights"] = n
    active = df["effective_submission_attempts"].to_numpy(float) > 0
    if np.any(active):
        out["attempt_positive"] = binary_metrics(df.loc[active].copy(), np.asarray(q)[active])
    return out


def overlay_probability(mu: np.ndarray, per_attempt_p: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(
        -np.maximum(np.asarray(mu, float), 0.0) * np.clip(per_attempt_p, 0.0, 1.0)
    )


def evaluate_period(frame: pd.DataFrame, start_year: int, end_year: int, train_cutoff: str) -> dict:
    cutoff = pd.Timestamp(train_cutoff)
    train = frame[frame["event_date"] < cutoff].copy()
    test = frame[frame["event_date"].dt.year.between(start_year, end_year)].copy()
    if len(train) < 100 or len(test) < 100:
        raise RuntimeError(f"insufficient rows train={len(train)} test={len(test)}")

    scales = {
        "tendency_only": fit_attempt_scale(train, "attempt_rate_tendency_only"),
        "tendency_x_suppression": fit_attempt_scale(train, "attempt_rate_with_suppression"),
    }
    mu, attempt = {}, {}
    for name, rate_col in (
        ("tendency_only", "attempt_rate_tendency_only"),
        ("tendency_x_suppression", "attempt_rate_with_suppression"),
    ):
        mu[name] = (
            scales[name] * test[rate_col].to_numpy(float)
            * test["fight_elapsed_seconds"].to_numpy(float)
        )
        attempt[name] = attempt_metrics(test, mu[name])

    alpha0 = fit_conversion_offset(train, 0.0)
    alpha1 = fit_conversion_offset(train, BETA_STAGE11C)
    alpha_current_joint = fit_conversion_offset(train, BETA_CURRENT)
    arms = {
        "baseline_only": (alpha0, 0.0),
        "stage11c_beta1_joint_offset": (alpha1, BETA_STAGE11C),
        "current_beta_literal_baseline_offset": (alpha0, BETA_CURRENT),
        "current_beta_joint_offset": (alpha_current_joint, BETA_CURRENT),
    }
    conversion, overlays = {}, {}
    for name, (alpha, beta) in arms.items():
        per, q_actual = conversion_probabilities(test, alpha, beta)
        conversion[name] = {
            "alpha": float(alpha), "beta": float(beta),
            "mean_per_attempt_probability": float(np.mean(per)),
            **with_correct_side(test, q_actual),
        }
        overlays[name] = with_correct_side(
            test, overlay_probability(mu["tendency_x_suppression"], per)
        )

    return {
        "train_cutoff": str(cutoff.date()), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "attempt_scales": scales, "attempt_models": attempt,
        "conversion_actual_attempts": conversion,
        "static_tendency_x_suppression_overlay": overlays,
    }


def nine_fight_diagnostic(frame: pd.DataFrame, confirmation: dict) -> list[dict]:
    scale = float(confirmation["attempt_scales"]["tendency_x_suppression"])
    conv = confirmation["conversion_actual_attempts"]
    alpha0 = float(conv["baseline_only"]["alpha"])
    alpha_joint = float(conv["current_beta_joint_offset"]["alpha"])
    rows = []
    for a, b in TARGETS:
        g = frame[
            frame["fighter_name"].astype(str).isin([a, b])
            & frame["opponent_name"].astype(str).isin([a, b])
            & frame["event_date"].dt.year.eq(2026)
        ].copy()
        if g.empty:
            continue
        g = g[g["event_date"].eq(g["event_date"].max())]
        if len(g) != 2:
            continue
        for r in g.itertuples(index=False):
            scheduled = 900.0
            raw_rate = float(r.attempt_rate_with_suppression)
            expected_attempts = scale * raw_rate * scheduled
            base_z = float(logit(float(r.submission_conversion_baseline)))
            edge = float(r.submission_edge)
            p_literal = float(logistic(base_z + alpha0 + BETA_CURRENT * edge))
            p_joint = float(logistic(base_z + alpha_joint + BETA_CURRENT * edge))
            rows.append({
                "fight_id": str(r.fight_id), "date": str(pd.Timestamp(r.event_date).date()),
                "matchup": f"{a} vs {b}", "fighter": str(r.fighter_name),
                "submission_tendency": float(r.submission_tendency),
                "opponent_submission_suppression": float(r.opp_submission_suppression),
                "base_attempt_rate_per_sec": raw_rate,
                "expected_attempts_scheduled_fight": expected_attempts,
                "submission_offense": float(r.submission_offense),
                "opponent_submission_defense": float(r.opp_submission_defense),
                "submission_edge": edge,
                "conversion_baseline": float(r.submission_conversion_baseline),
                "current_literal_per_attempt_p": p_literal,
                "current_joint_offset_per_attempt_p": p_joint,
                "current_literal_static_sub_probability": float(
                    overlay_probability(np.asarray([expected_attempts]), np.asarray([p_literal]))[0]
                ),
                "current_joint_static_sub_probability": float(
                    overlay_probability(np.asarray([expected_attempts]), np.asarray([p_joint]))[0]
                ),
                "actual_submission_finish": int(r.submission_finish),
            })
    return rows


def print_period(label: str, block: dict) -> None:
    print("\n" + "=" * 145)
    print(label)
    print("=" * 145)
    print("ATTEMPT OPPORTUNITY")
    print("arm                         | scale    | E/O   | mean actual/pred | any AUC | Brier  | LL")
    for arm, m in block["attempt_models"].items():
        print(
            f"{arm:27s} | {block['attempt_scales'][arm]:8.4f} | {m['expected_to_observed']:.3f} | "
            f"{m['mean_actual_attempts']:.3f}/{m['mean_expected_attempts']:.3f} | "
            f"{m['any_attempt_auc']:.4f} | {m['any_attempt_brier']:.4f} | {m['any_attempt_log_loss']:.4f}"
        )
    print("\nCONVERSION GIVEN ACTUAL EFFECTIVE ATTEMPTS")
    print("arm                                  | alpha    | beta    | E/O   | AUC    | correct | Brier  | LL")
    for arm, m in block["conversion_actual_attempts"].items():
        print(
            f"{arm:37s} | {m['alpha']:+8.4f} | {m['beta']:7.4f} | {m['expected_to_observed']:.3f} | "
            f"{m['auc']:.4f} | {m['correct_side_sub_rate']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f}"
        )
    print("\nSTATIC ATTEMPT x CONVERSION OVERLAY (not full MC)")
    print("arm                                  | E/O   | AUC    | correct | Brier  | LL")
    for arm, m in block["static_tendency_x_suppression_overlay"].items():
        print(
            f"{arm:37s} | {m['expected_to_observed']:.3f} | {m['auc']:.4f} | "
            f"{m['correct_side_sub_rate']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f}"
        )


def main() -> None:
    frame = build_frame()
    selection = evaluate_period(frame, 2020, 2024, "2020-01-01")
    confirmation = evaluate_period(frame, 2025, 2026, "2025-01-01")
    nine = nine_fight_diagnostic(frame, confirmation)
    result = {
        "study": "submission finish architecture audit", "production_changed": False,
        "current_runtime_beta": BETA_CURRENT, "selection_years": "2020-2024",
        "confirmation_years": "2025-2026", "frame_rows": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()), "selection": selection,
        "confirmation": confirmation, "nine_fight_diagnostic": nine,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    pd.DataFrame(nine).to_csv(OUTDIR / "nine_fight_diagnostic.csv", index=False)
    print("SUBMISSION FINISH ARCHITECTURE AUDIT")
    print(f"rows={len(frame):,} fights={frame['fight_id'].nunique():,} current beta={BETA_CURRENT:.10f}")
    print_period("SELECTION 2020-2024 (parameters fit strictly pre-2020)", selection)
    print_period("CONFIRMATION 2025-2026 (parameters fit strictly pre-2025)", confirmation)
    print("\n" + "=" * 145)
    print("NINE-FIGHT STATIC DIAGNOSTIC")
    print("=" * 145)
    if nine:
        cols = [
            "matchup", "fighter", "expected_attempts_scheduled_fight", "submission_edge",
            "current_literal_per_attempt_p", "current_joint_offset_per_attempt_p",
            "current_literal_static_sub_probability", "current_joint_static_sub_probability",
            "actual_submission_finish",
        ]
        print(pd.DataFrame(nine)[cols].to_string(index=False))
    print(f"\nWrote {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
