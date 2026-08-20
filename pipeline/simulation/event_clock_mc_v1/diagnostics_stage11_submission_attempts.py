from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    metrics,
    within_bout_direction,
)
from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import (
    BINOMIAL_ALPHA,
    POISSON_ALPHA,
    BinomialRidge,
    PoissonExposureRidge,
)
from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)
from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import (
    direct_feature_columns,
    draw_frailty,
    estimate_nb_alpha,
)
from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    physical_hurdle_probability,
)

FIGHTS = 500
PATHS = 20
SEED = 20260818

STAGE9_PATHS = Path(
    "data/diagnostics/event_clock_mc_v1/stage9_final_flow_paths_500x20.csv"
)
OUT = Path(
    "data/diagnostics/event_clock_mc_v1/stage11a_submission_attempts_500x20.csv"
)
PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/stage11a_submission_attempt_paths_500x20.csv"
)

CTX = [
    "ctx_self_td_landed",
    "ctx_opp_td_landed",
    "ctx_self_ground_attempted",
    "ctx_opp_ground_attempted",
    "ctx_self_control",
    "ctx_opp_control",
    "ctx_total_td_landed",
    "ctx_total_ground_attempted",
    "ctx_total_control",
    "ctx_self_control_share",
]


def norm_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.casefold()


def build_submission_targets() -> pd.DataFrame:
    """One historical submission-attempt/result row per fighter/fight."""
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    required = {
        "fight_id", "method", "winner", "winner_id",
        "r_id", "r_name", "r_sub_att",
        "b_id", "b_name", "b_sub_att",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(f"Master missing submission columns: {sorted(missing)}")

    master["fight_id"] = master["fight_id"].astype(str)
    winner_id = norm_text(master["winner_id"])
    winner_name = norm_text(master["winner"])
    red_id, blue_id = norm_text(master["r_id"]), norm_text(master["b_id"])
    red_name, blue_name = norm_text(master["r_name"]), norm_text(master["b_name"])

    red_win = ((winner_id != "") & (winner_id == red_id)) | (
        (winner_id == "") & (winner_name != "") & (winner_name == red_name)
    )
    blue_win = ((winner_id != "") & (winner_id == blue_id)) | (
        (winner_id == "") & (winner_name != "") & (winner_name == blue_name)
    )
    is_sub = norm_text(master["method"]).str.contains(
        "submission", regex=False, na=False
    )

    def attempts(col: str) -> pd.Series:
        return pd.to_numeric(master[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    red = pd.DataFrame({
        "fight_id": master["fight_id"],
        "side": "red",
        "submission_attempted": attempts("r_sub_att"),
        "submission_win": (is_sub & red_win).astype(int),
        "method": master["method"].astype(str),
    })
    blue = pd.DataFrame({
        "fight_id": master["fight_id"],
        "side": "blue",
        "submission_attempted": attempts("b_sub_att"),
        "submission_win": (is_sub & blue_win).astype(int),
        "method": master["method"].astype(str),
    })
    return pd.concat([red, blue], ignore_index=True)


def add_grappling_context(
    frame: pd.DataFrame,
    *,
    td_col: str,
    ground_col: str,
    control_col: str,
    path_level: bool = False,
) -> pd.DataFrame:
    """Create fighter/opponent grappling context from historical or Stage-9 rows."""
    out = frame.copy()
    keys = ["fight_id", "side"]
    if path_level:
        keys.insert(1, "path")

    out["ctx_self_td_landed"] = out[td_col].astype(float)
    out["ctx_self_ground_attempted"] = out[ground_col].astype(float)
    out["ctx_self_control"] = out[control_col].astype(float)

    opp = out[keys + [td_col, ground_col, control_col]].copy()
    opp["side"] = opp["side"].map({"red": "blue", "blue": "red"})
    opp = opp.rename(columns={
        td_col: "ctx_opp_td_landed",
        ground_col: "ctx_opp_ground_attempted",
        control_col: "ctx_opp_control",
    })
    out = out.merge(opp, on=keys, how="left", validate="one_to_one")

    out["ctx_total_td_landed"] = (
        out["ctx_self_td_landed"] + out["ctx_opp_td_landed"]
    )
    out["ctx_total_ground_attempted"] = (
        out["ctx_self_ground_attempted"] + out["ctx_opp_ground_attempted"]
    )
    out["ctx_total_control"] = out["ctx_self_control"] + out["ctx_opp_control"]
    out["ctx_self_control_share"] = np.where(
        out["ctx_total_control"] > 0,
        out["ctx_self_control"] / out["ctx_total_control"],
        0.5,
    )
    return out


class SubmissionAttemptHurdle:
    """Mean-count + occurrence hurdle with Gamma-Poisson positive-path extras."""

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]):
        self.feature_cols = list(feature_cols)
        x = frame[self.feature_cols].to_numpy(float)
        y = frame["submission_attempted"].to_numpy(float)
        exposure = np.maximum(frame["duration"].to_numpy(float) / 900.0, 1e-6)

        self.mean_model = PoissonExposureRidge(alpha=POISSON_ALPHA).fit(
            x, y, exposure
        )
        self.occurrence_model = BinomialRidge(alpha=BINOMIAL_ALPHA).fit(
            x,
            successes=(y > 0).astype(float),
            trials=np.ones(len(frame), dtype=float),
        )

        mu, p, conditional = self.predict(frame)
        positive = y > 0
        self.extra_alpha = estimate_nb_alpha(
            np.maximum(y[positive] - 1.0, 0.0),
            np.maximum(conditional[positive] - 1.0, 1e-6),
        )
        return self

    def predict(self, frame: pd.DataFrame):
        x = frame[self.feature_cols].to_numpy(float)
        exposure = np.maximum(frame["duration"].to_numpy(float) / 900.0, 1e-6)
        mu = self.mean_model.predict(x, exposure)
        p = self.occurrence_model.predict_probability(x)
        p = physical_hurdle_probability(mu, p)
        conditional = np.divide(mu, p, out=np.zeros_like(mu), where=p > 0)
        return mu, p, conditional

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        _, p, conditional = self.predict(frame)
        result = np.zeros(len(frame), dtype=int)
        for i in np.flatnonzero(rng.random(len(frame)) < p):
            extra_mean = max(float(conditional[i]) - 1.0, 0.0)
            extras = rng.poisson(extra_mean * draw_frailty(rng, self.extra_alpha))
            result[i] = 1 + int(extras)
        return result


def print_opportunity_audit(frame: pd.DataFrame) -> None:
    positive = frame[frame["submission_attempted"] > 0]
    print("\n" + "=" * 120)
    print("HISTORICAL SUBMISSION-ATTEMPT OPPORTUNITY AUDIT")
    print("=" * 120)
    print(
        f"positive fighter-fight rows: {len(positive)}/{len(frame)} "
        f"({len(positive) / max(len(frame), 1):.2%})"
    )
    if positive.empty:
        return
    checks = {
        "self TD landed > 0": positive["ctx_self_td_landed"] > 0,
        "self control > 0": positive["ctx_self_control"] > 0,
        "self ground attempts > 0": positive["ctx_self_ground_attempted"] > 0,
        "opponent control > 0": positive["ctx_opp_control"] > 0,
        "any Stage-9 grappling context": (
            (positive["ctx_total_td_landed"] > 0)
            | (positive["ctx_total_ground_attempted"] > 0)
            | (positive["ctx_total_control"] > 0)
        ),
    }
    for label, mask in checks.items():
        print(f"{label:<34}: {mask.mean():.2%}")


def occurrence_scores(actual, p):
    y = (np.asarray(actual, dtype=float) > 0).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    brier = brier_score_loss(y, p)
    ll = log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])
    return auc, brier, ll


def print_model(frame, expected_col: str, p_col: str, label: str) -> None:
    actual = frame["submission_attempted"]
    expected = frame[expected_col]
    _, rho, mae = metrics(actual, expected)
    side, n = within_bout_direction(frame, "submission_attempted", expected_col)
    auc, brier, ll = occurrence_scores(actual, frame[p_col])
    print("\n" + label)
    print("-" * 120)
    print(
        f"mean HIST={actual.mean():.4f} | PRED={expected.mean():.4f} | "
        f"positive HIST={(actual > 0).mean():.2%} | PRED={frame[p_col].mean():.2%}"
    )
    print(
        f"count rho={rho:+.4f} | MAE={mae:.4f} | correct-side={side:.2%} (N={n})"
    )
    print(f"any-attempt AUC={auc:.4f} | Brier={brier:.4f} | log loss={ll:.4f}")


def print_dist(label: str, values) -> None:
    x = np.asarray(values, dtype=float)
    print(
        f"{label:<20} | mean={x.mean():.3f} | std={x.std(ddof=1):.3f} | "
        f"zero={(x == 0).mean():.2%} | p50={np.quantile(x, .5):.2f} | "
        f"p90={np.quantile(x, .9):.2f} | p99={np.quantile(x, .99):.2f}"
    )


def main() -> None:
    print("=" * 130)
    print("EVENT CLOCK MC — STAGE 11A SUBMISSION ATTEMPT CALIBRATION")
    print("=" * 130)
    print(
        "FSR V2 submission profile -> attempt hurdle -> realized grappling context "
        "-> path-level submission-attempt budget"
    )

    train, test = prepare_direct_predictions()
    train["fight_id"] = train["fight_id"].astype(str)
    test["fight_id"] = test["fight_id"].astype(str)
    if test["fight_id"].nunique() != FIGHTS:
        raise RuntimeError(f"Expected {FIGHTS} fresh fights")

    targets = build_submission_targets()
    train = train.merge(targets, on=["fight_id", "side"], validate="one_to_one")
    test = test.merge(targets, on=["fight_id", "side"], validate="one_to_one")
    if test["fight_id"].nunique() != FIGHTS:
        raise RuntimeError("Submission target join lost fresh fights")

    train = add_grappling_context(
        train,
        td_col="td_landed",
        ground_col="ground_attempted",
        control_col="qualified_control_inflicted_seconds",
    )
    test = add_grappling_context(
        test,
        td_col="td_landed",
        ground_col="ground_attempted",
        control_col="qualified_control_inflicted_seconds",
    )
    print_opportunity_audit(test)

    static = direct_feature_columns()
    baseline = SubmissionAttemptHurdle().fit(train, static)
    context = SubmissionAttemptHurdle().fit(train, static + CTX)

    for frame in (train, test):
        base_mu, base_p, _ = baseline.predict(frame)
        ctx_mu, ctx_p, _ = context.predict(frame)
        frame["pred_submission_attempted_fsr_only"] = base_mu
        frame["pred_submission_positive_fsr_only"] = base_p
        frame["pred_submission_attempted_actual_context"] = ctx_mu
        frame["pred_submission_positive_actual_context"] = ctx_p

    if not STAGE9_PATHS.exists():
        raise RuntimeError(f"Stage-9 path file not found: {STAGE9_PATHS}")

    paths = pd.read_csv(STAGE9_PATHS, low_memory=False)
    paths["fight_id"] = paths["fight_id"].astype(str)
    if paths["fight_id"].nunique() != FIGHTS:
        raise RuntimeError(f"Stage-9 paths do not contain {FIGHTS} fights")

    paths = add_grappling_context(
        paths,
        td_col="sim_td_landed",
        ground_col="sim_ground_attempted",
        control_col="sim_control",
        path_level=True,
    )
    paths = paths.merge(
        test[["fight_id", "side", *static]],
        on=["fight_id", "side"],
        how="left",
        validate="many_to_one",
    )

    path_mu, path_p, path_cond = context.predict(paths)
    paths["pred_submission_attempted"] = path_mu
    paths["pred_submission_positive_probability"] = path_p
    paths["pred_submission_conditional_attempts"] = path_cond
    paths["sim_submission_attempted"] = context.sample(
        paths, np.random.default_rng(SEED)
    )

    sim_mean = paths.groupby(["fight_id", "side"], as_index=False).agg(
        sim_submission_attempted=("sim_submission_attempted", "mean"),
        sim_submission_positive_probability=(
            "pred_submission_positive_probability", "mean"
        ),
        sim_submission_expected_mean=("pred_submission_attempted", "mean"),
    )
    result = test.merge(
        sim_mean, on=["fight_id", "side"], how="left", validate="one_to_one"
    )

    print("\n" + "=" * 130)
    print("MODEL COMPARISON — FRESH 500")
    print("=" * 130)
    print_model(
        result,
        "pred_submission_attempted_fsr_only",
        "pred_submission_positive_fsr_only",
        "FSR-ONLY PREFIGHT SUBMISSION ATTEMPTS",
    )
    print_model(
        result,
        "pred_submission_attempted_actual_context",
        "pred_submission_positive_actual_context",
        "ACTUAL-CONTEXT UPPER BOUND",
    )
    print_model(
        result,
        "sim_submission_attempted",
        "sim_submission_positive_probability",
        "STAGE-9 SIMULATED-CONTEXT ATTEMPTS",
    )

    print("\n" + "=" * 130)
    print("SUBMISSION-ATTEMPT DISTRIBUTIONS")
    print("=" * 130)
    print_dist("HIST fighter-fight", result["submission_attempted"])
    print_dist("SIM path rows", paths["sim_submission_attempted"])
    print(f"FSR-only extra alpha: {baseline.extra_alpha:.4f}")
    print(f"context extra alpha:  {context.extra_alpha:.4f}")

    total_sub_wins = int(result["submission_win"].sum())
    total_attempts = float(result["submission_attempted"].sum())
    zero_attempt_sub_wins = int(
        (
            result.loc[result["submission_win"] == 1, "submission_attempted"]
            <= 0
        ).sum()
    )
    print("\n" + "=" * 130)
    print("STAGE 11B CONVERSION BASELINE")
    print("=" * 130)
    print(f"historical submission wins: {total_sub_wins}")
    print(f"historical recorded attempts: {total_attempts:.0f}")
    if total_attempts > 0:
        print(f"raw SUB wins / attempts: {total_sub_wins / total_attempts:.2%}")
    print(f"submission wins with zero recorded attempts: {zero_attempt_sub_wins}")
    print("older RFS reference: Beta(1,9) smoothed attempt conversion (10% prior mean)")

    result["absolute_sim_error"] = (
        result["submission_attempted"] - result["sim_submission_attempted"]
    ).abs()
    display = [
        "fight_id", "fighter_name", "opponent_name", "side",
        "submission_attempted", "submission_win",
        "pred_submission_attempted_fsr_only",
        "pred_submission_attempted_actual_context",
        "sim_submission_attempted",
        "sim_submission_positive_probability",
        "ctx_self_td_landed", "ctx_opp_td_landed",
        "ctx_self_ground_attempted", "ctx_opp_ground_attempted",
        "ctx_self_control", "ctx_opp_control",
        "self_submission_tendency", "opp_submission_suppression",
        "self_submission_offense", "opp_submission_defense",
    ]
    print("\n" + "=" * 170)
    print("LARGEST SIMULATED SUBMISSION-ATTEMPT MISSES")
    print("=" * 170)
    print(
        result.sort_values("absolute_sim_error", ascending=False)[display]
        .head(40)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    paths.to_csv(PATH_OUT, index=False)
    print(f"\nwrote: {OUT}")
    print(f"wrote: {PATH_OUT}")


if __name__ == "__main__":
    main()
