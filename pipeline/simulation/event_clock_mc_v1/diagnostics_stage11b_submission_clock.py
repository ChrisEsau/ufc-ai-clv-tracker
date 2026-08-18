from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import metrics, within_bout_direction
from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import prepare_direct_predictions
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11_submission_attempts import build_submission_targets

FIGHTS = 500
PATHS = 20
SEED = 20260818

STAGE9_PATHS = Path("data/diagnostics/event_clock_mc_v1/stage9_final_flow_paths_500x20.csv")
OUT = Path("data/diagnostics/event_clock_mc_v1/stage11b_submission_clock_500x20.csv")
PATH_OUT = Path("data/diagnostics/event_clock_mc_v1/stage11b_submission_clock_paths_500x20.csv")

# Top-control seconds are the reference opportunity exposure.
# The other values are fitted as top-control-equivalent exposure.
PARAM_NAMES = (
    "rate_scale",
    "bottom_control_weight",
    "td_window_seconds",
    "ground_event_seconds",
)
INITIAL = np.array([5.0, 0.50, 6.0, 1.5], dtype=float)
BOUNDS = (
    (0.10, 100.0),
    (0.05, 5.0),
    (0.25, 60.0),
    (0.05, 20.0),
)


def add_context(frame, td_att, td_lnd, ground_att, control, *, path_level=False):
    out = frame.copy()
    keys = ["fight_id", "side"]
    if path_level:
        keys.insert(1, "path")

    rename_self = {
        td_att: "clock_self_td_attempted",
        td_lnd: "clock_self_td_landed",
        ground_att: "clock_self_ground_attempted",
        control: "clock_self_control",
    }
    for src, dst in rename_self.items():
        out[dst] = pd.to_numeric(out[src], errors="coerce").fillna(0.0).clip(lower=0.0)

    opp = out[keys + [td_att, td_lnd, ground_att, control]].copy()
    opp["side"] = opp["side"].map({"red": "blue", "blue": "red"})
    opp = opp.rename(columns={
        td_att: "clock_opp_td_attempted",
        td_lnd: "clock_opp_td_landed",
        ground_att: "clock_opp_ground_attempted",
        control: "clock_opp_control",
    })
    out = out.merge(opp, on=keys, how="left", validate="one_to_one")

    for col in (
        "clock_opp_td_attempted",
        "clock_opp_td_landed",
        "clock_opp_ground_attempted",
        "clock_opp_control",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    out["clock_total_td_attempted"] = out["clock_self_td_attempted"] + out["clock_opp_td_attempted"]
    out["clock_total_td_landed"] = out["clock_self_td_landed"] + out["clock_opp_td_landed"]
    out["clock_total_ground_attempted"] = out["clock_self_ground_attempted"] + out["clock_opp_ground_attempted"]
    out["clock_total_control"] = out["clock_self_control"] + out["clock_opp_control"]
    out["clock_has_opportunity"] = (
        (out["clock_total_control"] > 0)
        | (out["clock_total_td_attempted"] > 0)
        | (out["clock_total_ground_attempted"] > 0)
    ).astype(int)
    return out


def add_fsr_rate(frame):
    out = frame.copy()
    tendency = pd.to_numeric(
        out["self_submission_tendency"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    suppression = pd.to_numeric(
        out["opp_submission_suppression"], errors="coerce"
    ).fillna(1.0).clip(lower=0.0)
    # Preserve the canonical FSR V2 matchup rate used by the old Event MC.
    out["fsr_submission_rate_per_second"] = tendency * suppression
    return out


def unpack(theta):
    values = np.exp(np.asarray(theta, dtype=float))
    return dict(zip(PARAM_NAMES, values))


def exposure_components(frame, params):
    return {
        "top_control": frame["clock_self_control"].to_numpy(float),
        "bottom_control": (
            frame["clock_opp_control"].to_numpy(float)
            * params["bottom_control_weight"]
        ),
        "wrestling": (
            frame["clock_total_td_attempted"].to_numpy(float)
            * params["td_window_seconds"]
        ),
        "ground_activity": (
            frame["clock_total_ground_attempted"].to_numpy(float)
            * params["ground_event_seconds"]
        ),
    }


def expected_attempts(frame, params):
    components = exposure_components(frame, params)
    rate = (
        frame["fsr_submission_rate_per_second"].to_numpy(float)
        * params["rate_scale"]
    )
    channel_mu = {name: rate * np.maximum(x, 0.0) for name, x in components.items()}
    total = np.zeros(len(frame), dtype=float)
    for value in channel_mu.values():
        total += value
    return total, channel_mu


def fit_clock(train):
    y = train["submission_attempted"].to_numpy(float)
    x0 = np.log(INITIAL)
    bounds = [(np.log(lo), np.log(hi)) for lo, hi in BOUNDS]

    def objective(theta):
        params = unpack(theta)
        mu, _ = expected_attempts(train, params)
        mu = np.maximum(mu, 1e-12)
        nll = np.sum(mu - y * np.log(mu))
        # Weak stabilization only; fresh-500 performance determines acceptance.
        penalty = 0.01 * np.sum((theta - x0) ** 2)
        return float(nll + penalty)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Submission clock fit failed: {result.message}")
    return unpack(result.x)


def run_clock(rate, exposure, rng):
    """Continuous-time Poisson clock over cumulative eligible exposure."""
    if rate <= 0 or exposure <= 0:
        return 0
    elapsed = 0.0
    count = 0
    while True:
        elapsed += float(rng.exponential(1.0 / rate))
        if elapsed > exposure:
            return count
        count += 1


def simulate_row(row, params, rng):
    one = row.to_frame().T
    components = exposure_components(one, params)
    rate = max(float(row["fsr_submission_rate_per_second"]), 0.0) * params["rate_scale"]

    result = {}
    total = 0
    for channel in ("top_control", "bottom_control", "wrestling", "ground_activity"):
        count = run_clock(rate, float(components[channel][0]), rng)
        result[f"sim_submission_{channel}"] = count
        total += count
    result["sim_submission_attempted"] = total
    return result


def occurrence_scores(actual, probability):
    y = (np.asarray(actual, dtype=float) > 0).astype(int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return (
        float(auc),
        float(brier_score_loss(y, p)),
        float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
    )


def print_model(frame, expected_col, probability_col, label):
    actual = frame["submission_attempted"]
    expected = frame[expected_col]
    _, rho, mae = metrics(actual, expected)
    side, n = within_bout_direction(frame, "submission_attempted", expected_col)
    auc, brier, ll = occurrence_scores(actual, frame[probability_col])

    print("\n" + label)
    print("-" * 132)
    print(
        f"mean HIST={actual.mean():.4f} | PRED={expected.mean():.4f} | "
        f"positive HIST={(actual > 0).mean():.2%} | PRED={frame[probability_col].mean():.2%}"
    )
    print(
        f"count rho={rho:+.4f} | MAE={mae:.4f} | "
        f"correct-side={side:.2%} (N={n})"
    )
    print(f"any-attempt AUC={auc:.4f} | Brier={brier:.4f} | log loss={ll:.4f}")


def print_dist(label, values):
    x = np.asarray(values, dtype=float)
    print(
        f"{label:<24} | mean={x.mean():.3f} | std={x.std(ddof=1):.3f} | "
        f"zero={(x == 0).mean():.2%} | p50={np.quantile(x,.5):.2f} | "
        f"p90={np.quantile(x,.9):.2f} | p99={np.quantile(x,.99):.2f}"
    )


def opportunity_audit(frame, label):
    positive = frame[frame["submission_attempted"] > 0]
    print(f"\n{label}")
    print("-" * 132)
    print(
        f"positive fighter-fight rows: {len(positive)}/{len(frame)} "
        f"({len(positive)/max(len(frame),1):.2%})"
    )
    if positive.empty:
        return
    off = positive["clock_has_opportunity"].eq(0)
    print(f"positive attempts with clock OFF: {off.mean():.2%} (N={int(off.sum())})")
    checks = (
        ("own control > 0", positive["clock_self_control"] > 0),
        ("opponent control > 0", positive["clock_opp_control"] > 0),
        ("any TD attempt", positive["clock_total_td_attempted"] > 0),
        ("any ground strike attempt", positive["clock_total_ground_attempted"] > 0),
    )
    for name, mask in checks:
        print(f"{name:<30}: {mask.mean():.2%}")


def main():
    print("=" * 142)
    print("EVENT CLOCK MC — STAGE 11B CONTEXTUAL SUBMISSION-ATTEMPT CLOCK")
    print("=" * 142)
    print(
        "FSR V2 tendency × opponent suppression -> opportunity exposure "
        "-> exponential submission-attempt clock"
    )

    train, test = prepare_direct_predictions()
    for frame in (train, test):
        frame["fight_id"] = frame["fight_id"].astype(str)

    targets = build_submission_targets()[
        ["fight_id", "side", "submission_attempted", "submission_win"]
    ]
    train = train.merge(targets, on=["fight_id", "side"], how="inner", validate="one_to_one")
    test = test.merge(targets, on=["fight_id", "side"], how="inner", validate="one_to_one")

    if test["fight_id"].nunique() != FIGHTS:
        raise RuntimeError("Submission target join lost fresh fights.")

    train = add_fsr_rate(add_context(
        train, "td_attempted", "td_landed", "ground_attempted",
        "qualified_control_inflicted_seconds"
    ))
    test = add_fsr_rate(add_context(
        test, "td_attempted", "td_landed", "ground_attempted",
        "qualified_control_inflicted_seconds"
    ))

    print("\n" + "=" * 142)
    print("OPPORTUNITY ELIGIBILITY")
    print("=" * 142)
    opportunity_audit(train, "TRAIN")
    opportunity_audit(test, "FRESH 500")

    # Reference: old FSR clock ran for every elapsed fight second.
    for frame in (train, test):
        frame["pred_old_fullfight_clock"] = (
            frame["fsr_submission_rate_per_second"] * frame["duration"]
        )
        frame["pred_old_fullfight_positive"] = 1.0 - np.exp(
            -frame["pred_old_fullfight_clock"]
        )

    params = fit_clock(train)

    print("\n" + "=" * 142)
    print("FITTED OPPORTUNITY-CLOCK PARAMETERS")
    print("=" * 142)
    for name in PARAM_NAMES:
        print(f"{name:<28}: {params[name]:.6f}")

    for frame in (train, test):
        mu, channel_mu = expected_attempts(frame, params)
        frame["pred_context_clock_attempted"] = mu
        frame["pred_context_clock_positive"] = 1.0 - np.exp(-mu)
        components = exposure_components(frame, params)
        frame["pred_clock_total_equivalent_seconds"] = sum(components.values())
        for channel in components:
            frame[f"pred_clock_exposure_{channel}"] = components[channel]
            frame[f"pred_clock_mu_{channel}"] = channel_mu[channel]

    print("\n" + "=" * 142)
    print("HISTORICAL-CONTEXT MODEL COMPARISON — FRESH 500")
    print("=" * 142)
    print_model(
        test, "pred_old_fullfight_clock", "pred_old_fullfight_positive",
        "OLD FSR FULL-FIGHT CLOCK"
    )
    print_model(
        test, "pred_context_clock_attempted", "pred_context_clock_positive",
        "CONTEXTUAL OPPORTUNITY CLOCK — ACTUAL CONTEXT"
    )

    if not STAGE9_PATHS.exists():
        raise RuntimeError(f"Stage-9 path file not found: {STAGE9_PATHS}")

    paths = pd.read_csv(STAGE9_PATHS, low_memory=False)
    paths["fight_id"] = paths["fight_id"].astype(str)
    if paths["fight_id"].nunique() != FIGHTS:
        raise RuntimeError(f"Expected {FIGHTS} fights in Stage-9 path file.")

    path_frame = add_context(
        paths,
        "sim_td_attempted",
        "sim_td_landed",
        "sim_ground_attempted",
        "sim_control",
        path_level=True,
    )

    static = test[
        [
            "fight_id", "side", "fighter_name", "opponent_name", "duration",
            "self_submission_tendency", "opp_submission_suppression",
        ]
    ].copy()

    path_frame = path_frame.merge(
        static,
        on=["fight_id", "side"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_static"),
    )

    for col in ("fighter_name", "opponent_name", "duration"):
        alt = f"{col}_static"
        if alt in path_frame:
            if col in path_frame:
                path_frame[col] = path_frame[col].fillna(path_frame[alt])
            else:
                path_frame[col] = path_frame[alt]

    path_frame = add_fsr_rate(path_frame)

    path_mu, path_channel_mu = expected_attempts(path_frame, params)
    path_frame["pred_submission_clock_attempted"] = path_mu
    path_frame["pred_submission_clock_positive"] = 1.0 - np.exp(-path_mu)

    components = exposure_components(path_frame, params)
    path_frame["clock_total_equivalent_seconds"] = sum(components.values())
    for channel in components:
        path_frame[f"clock_exposure_{channel}"] = components[channel]
        path_frame[f"clock_mu_{channel}"] = path_channel_mu[channel]

    sampled = []
    for i, row in path_frame.iterrows():
        rng = np.random.default_rng(SEED + int(i))
        sampled.append(simulate_row(row, params, rng))
    sampled = pd.DataFrame(sampled, index=path_frame.index)
    for col in sampled:
        path_frame[col] = sampled[col]

    sim_mean = path_frame.groupby(["fight_id", "side"], as_index=False).agg(
        sim_submission_attempted=("sim_submission_attempted", "mean"),
        sim_submission_positive_probability=("pred_submission_clock_positive", "mean"),
        sim_submission_expected=("pred_submission_clock_attempted", "mean"),
        sim_clock_equivalent_seconds=("clock_total_equivalent_seconds", "mean"),
        sim_submission_top_control=("sim_submission_top_control", "mean"),
        sim_submission_bottom_control=("sim_submission_bottom_control", "mean"),
        sim_submission_wrestling=("sim_submission_wrestling", "mean"),
        sim_submission_ground_activity=("sim_submission_ground_activity", "mean"),
    )

    result = test.merge(
        sim_mean, on=["fight_id", "side"], how="left", validate="one_to_one"
    )

    print("\n" + "=" * 142)
    print("STAGE-9 CONTEXTUAL CLOCK — FRESH 500")
    print("=" * 142)
    print_model(
        result, "sim_submission_expected", "sim_submission_positive_probability",
        "STAGE-9 CONTEXT — EXPECTED CLOCK"
    )
    print_model(
        result, "sim_submission_attempted", "sim_submission_positive_probability",
        "STAGE-9 CONTEXT — SAMPLED EXPONENTIAL CLOCK"
    )

    print("\n" + "=" * 142)
    print("SUBMISSION-ATTEMPT DISTRIBUTIONS")
    print("=" * 142)
    print_dist("HIST fighter-fight", result["submission_attempted"])
    print_dist("SIM path rows", path_frame["sim_submission_attempted"])

    total = max(float(path_frame["sim_submission_attempted"].sum()), 1.0)
    print("\n" + "=" * 142)
    print("SIMULATED SUBMISSION-ATTEMPT CLOCK SOURCES")
    print("=" * 142)
    for channel in ("top_control", "bottom_control", "wrestling", "ground_activity"):
        count = float(path_frame[f"sim_submission_{channel}"].sum())
        print(f"{channel:<24} | attempts={count:.0f} | share={count/total:.2%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    path_frame.to_csv(PATH_OUT, index=False)

    print(f"\nwrote: {OUT}")
    print(f"wrote: {PATH_OUT}")
    print(
        "\nNOTE: attempt clock only. Submission conversion remains disabled; "
        "if this passes, conversion becomes Stage 11C."
    )


if __name__ == "__main__":
    main()
